# backend/agents/order_agent.py
"""
Order Agent  ─  NovaCart AI  v2.0
══════════════════════════════════════════════════════════════════════
Multi-turn order flow with product catalog validation and payment confirmation.

Flow:
  TURN 1  — User says "order headphones"
    • LLM extracts product name + quantity from user message + short-term memory
    • search_product() does fuzzy lookup against the products table
    • If NOT FOUND   → politely inform user (item not available)
    • If FOUND       → show order summary card (product details + price + ETA)
                      → ask for payment method
    • State: pending_order = {product_id, product_name, quantity, price}

  TURN 2  — User replies "UPI" / "COD" / "2"
    • resolve_payment_method() maps input to canonical payment method
    • If unrecognised → re-ask for payment method (stays in pending state)
    • If recognised   → commit order to DB, clear pending_order
                       → show success message with Order ID

Why this design:
  • product_id stored in orders table for full traceability & analytics
  • price_per_unit + total_price stored for future spend analytics
  • payment_mode stored (column already existed in original schema)
  • LLM only does entity extraction — all business logic is in Python
"""

from __future__ import annotations
import json
import uuid

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from backend.db import get_connection
from backend.graph.state import ConversationState, get_last_human_message
from backend.memory import get_short_term_context
from backend.products import (
    search_product,
    format_product_card,
    resolve_payment_method,
    update_product_stock,
    check_stock,
)
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)


# ─────────────────────────────────────────────────────────────────────
# CUSTOM EXCEPTION  —  typed stock error with all needed context
# ─────────────────────────────────────────────────────────────────────

class InsufficientStockError(Exception):
    """Raised when requested quantity exceeds available stock."""
    def __init__(self, product_name: str, requested: int, available: int):
        self.product_name = product_name
        self.requested    = requested
        self.available    = available
        super().__init__(
            f"Insufficient stock for '{product_name}': "
            f"requested {requested}, available {available}"
        )


# ─────────────────────────────────────────────────────────────────────
# DB TOOL  —  creates the order row with full product + payment info
# ─────────────────────────────────────────────────────────────────────

def _commit_order(
    user_id:      str,
    product_id:   str,
    product_name: str,
    quantity:     int,
    price:        int,
    payment_mode: str,
) -> str:
    """
    Inserts a fully validated order into the database.
    Returns the generated order_id.
    Raises InsufficientStockError if stock is below requested quantity.
    """
    # ── BUG FIX 2: Stock validation BEFORE committing the order ──────
    # Previously, any quantity was accepted and only the DB stock value
    # was clamped to 0 (via MAX(0, stock - qty)). This meant an order
    # for 1000 laptops with only 30 in stock would still go through.
    # We now check stock first and raise a typed error if insufficient.
    is_sufficient, available = check_stock(product_id, quantity)
    if not is_sufficient:
        raise InsufficientStockError(product_name, quantity, available)

    order_id    = f"ORD-{uuid.uuid4().hex[:6].upper()}"
    total_price = price * quantity

    conn   = get_connection()
    cursor = conn.cursor()

    # Add price columns if they don't exist yet (safe migration)
    cursor.execute("PRAGMA table_info(orders)")
    existing_cols = {col[1] for col in cursor.fetchall()}

    if "price_per_unit" not in existing_cols:
        cursor.execute("ALTER TABLE orders ADD COLUMN price_per_unit INTEGER DEFAULT 0")
    if "total_price" not in existing_cols:
        cursor.execute("ALTER TABLE orders ADD COLUMN total_price INTEGER DEFAULT 0")

    cursor.execute(
        """
        INSERT INTO orders
            (order_id, user_id, product_id, product_name, quantity,
             price_per_unit, total_price, payment_mode, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (order_id, user_id, product_id, product_name, quantity,
         price, total_price, payment_mode, "PLACED"),
    )
    conn.commit()
    conn.close()

    # ✅ FIX 3: Decrement product stock after order is committed to DB
    update_product_stock(product_id, quantity)

    return order_id


# ─────────────────────────────────────────────────────────────────────
# LLM EXTRACTION PROMPT
# ─────────────────────────────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """\
You are an order extraction assistant for NovaCart, an e-commerce platform.

Your task: extract the product name and quantity the user wants to order.

You have access to recent conversation history. Use it to resolve references like:
  - "the same" or "same product" -> look at what was ordered or discussed before
  - "that item" / "those"        -> identify from context
  - "another one"                -> find the product from previous messages
  - a bare product name like "pizza" after "place another order for the same" -> pizza

Reasoning steps:
  1. Scan conversation history for any product already mentioned or ordered.
  2. Check the latest message for an explicit or implicit product name.
  3. Determine quantity -- default to 1 if not specified.
  4. If still no product can be identified, set product to null.

Output ONLY valid JSON. No markdown, no explanation:
{"product": "<product name or description>", "quantity": <number>}

If product cannot be identified:
{"product": null, "quantity": 1}
"""


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def _extract_product_and_qty(user_text: str, session_id: str):
    """Uses LLM + short-term memory to extract product name and quantity."""
    history  = get_short_term_context(session_id)
    messages = [SystemMessage(content=EXTRACTION_SYSTEM_PROMPT)]
    messages.extend(history)
    messages.append(HumanMessage(content=user_text))

    try:
        raw  = llm.invoke(messages).content.strip()
        raw  = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
    except Exception:
        data = {"product": None, "quantity": 1}

    product  = data.get("product")
    quantity = max(1, int(data.get("quantity") or 1))
    return product, quantity


def _order_success_message(
    order_id:     str,
    product_name: str,
    quantity:     int,
    price:        int,
    payment_mode: str,
) -> str:
    total = price * quantity
    return (
        "🎉 **Your Order Has Been Placed Successfully!**\n\n"
        f"🆔 Order ID     : **{order_id}**\n"
        f"🛍️  Product      : {product_name}\n"
        f"🔢 Quantity     : {quantity}\n"
        f"💰 Price/unit   : ₹{price:,}\n"
        f"💳 Total Paid   : ₹{total:,}\n"
        f"💵 Payment Mode : {payment_mode}\n\n"
        "You can track or return this order anytime using your Order ID!"
    )


# ─────────────────────────────────────────────────────────────────────
# MAIN AGENT
# ─────────────────────────────────────────────────────────────────────

def order_agent(state: ConversationState) -> ConversationState:
    """
    Handles two distinct sub-states:
      A) pending_order is set -> user is responding with a payment method
      B) pending_order is None -> fresh order request, extract + lookup product
    """
    user_text  = get_last_human_message(state["messages"])
    user_id    = state["user_id"]
    session_id = state["session_id"]

    # ══════════════════════════════════════════════════════════════
    # STATE A -- Payment method response (pending_order already set)
    # ══════════════════════════════════════════════════════════════
    if state.get("pending_order"):
        pending = state["pending_order"]

        payment_mode = resolve_payment_method(user_text)

        if not payment_mode:
            # Unrecognised input -- re-show options and stay pending
            state["messages"].append(
                AIMessage(
                    content=(
                        "Sorry, I didn't recognise that payment method. "
                        "Please choose one of:\n\n"
                        "   1️⃣  Cash on Delivery (COD)\n"
                        "   2️⃣  UPI\n"
                        "   3️⃣  Credit / Debit Card\n"
                        "   4️⃣  Net Banking\n\n"
                        "*Reply with the name or number (e.g. UPI or 2)*"
                    )
                )
            )
            return state

        # ── Commit the order ──────────────────────────────────────
        try:
            order_id = _commit_order(
                user_id      = user_id,
                product_id   = pending["product_id"],
                product_name = pending["product_name"],
                quantity     = pending["quantity"],
                price        = pending["price"],
                payment_mode = payment_mode,
            )
        except InsufficientStockError as e:
            # Stock was available when the card was shown but depleted by
            # the time the user confirmed payment (race condition / low stock).
            state["pending_order"] = None
            state["messages"].append(
                AIMessage(
                    content=(
                        f"⚠️ **Sorry, we can't complete your order.**\n\n"
                        f"By the time you confirmed, stock for "
                        f"**{e.product_name}** dropped to only "
                        f"**{e.available} unit(s)** — not enough to fill "
                        f"your order of **{e.requested}**.\n\n"
                        "Please try ordering a smaller quantity, "
                        "or check back later when stock is replenished."
                    )
                )
            )
            return state

        # Clear the pending state
        state["pending_order"] = None

        state["messages"].append(
            AIMessage(
                content=_order_success_message(
                    order_id     = order_id,
                    product_name = pending["product_name"],
                    quantity     = pending["quantity"],
                    price        = pending["price"],
                    payment_mode = payment_mode,
                )
            )
        )
        return state

    # ══════════════════════════════════════════════════════════════
    # STATE B -- Fresh order request: extract -> lookup -> show card
    # ══════════════════════════════════════════════════════════════

    # ── Step 1: LLM extracts product name + quantity ──────────────
    product_name_raw, quantity = _extract_product_and_qty(user_text, session_id)

    if not product_name_raw:
        state["messages"].append(
            AIMessage(
                content=(
                    "Sure 👍 I'd be happy to place an order!\n\n"
                    "Please tell me what product you'd like to order.\n\n"
                    "*Example: Order 1 wireless headphones*"
                )
            )
        )
        return state

    # ── Step 2: Fuzzy product catalog lookup ──────────────────────
    product = search_product(product_name_raw)

    if not product:
        state["messages"].append(
            AIMessage(
                content=(
                    f"😔 Sorry, **\"{product_name_raw}\"** is not available "
                    f"in our catalog right now.\n\n"
                    "We currently stock items across these categories:\n"
                    "   📱 Electronics\n"
                    "   🍳 Kitchen & Appliances\n"
                    "   👗 Clothing & Fashion\n"
                    "   📚 Books & Stationery\n"
                    "   🛒 Food & Grocery\n\n"
                    "Please try a different product and I'll be happy to help!"
                )
            )
        )
        return state

    # ── Step 3: Stock check BEFORE showing the order card ─────────
    # Prevents the user from entering the payment flow for a quantity
    # that can never be fulfilled. Better UX: fail early with a clear
    # message rather than at payment confirmation time.
    is_sufficient, available_stock = check_stock(product["product_id"], quantity)
    if not is_sufficient:
        state["messages"].append(
            AIMessage(
                content=(
                    f"⚠️ **Not enough stock for your order.**\n\n"
                    f"You requested **{quantity} unit(s)** of "
                    f"**{product['name']}**, but only "
                    f"**{available_stock} unit(s)** are currently available.\n\n"
                    f"Please try again with a quantity of **{available_stock} or less**."
                    + ("\n\n*This item is currently out of stock.*" if available_stock == 0 else "")
                )
            )
        )
        return state

    # ── Step 4: Show order summary card + ask for payment method ──
    state["pending_order"] = {
        "product_id":   product["product_id"],
        "product_name": product["name"],
        "quantity":     quantity,
        "price":        product["price"],
    }

    state["messages"].append(
        AIMessage(content=format_product_card(product, quantity))
    )
    return state