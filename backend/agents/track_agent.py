# backend/agents/track_agent.py
"""
Track Agent
════════════
Fetches order status and computes estimated delivery.

Business logic (dates, ETA) is computed in Python — not by the LLM.
The LLM is not used here since the response is fully structured data.

Key improvements over v1.0:
  • Per-function DB connection (thread safety fix).
  • Cleaner, structured response format.
"""

from langchain_core.messages import AIMessage
from backend.db import get_connection
from backend.graph.state import ConversationState
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()


# ─────────────────────────────────────────────────────────────────────
# DB TOOL
# ─────────────────────────────────────────────────────────────────────

def get_order_status(order_id: str, user_id: str):
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT status, product_name, quantity, order_date
           FROM orders
           WHERE order_id = ? AND user_id = ?""",
        (order_id, user_id)
    )
    row = cursor.fetchone()
    conn.close()
    return row


def add_business_days(start: datetime, days: int) -> datetime:
    """Add N business days (Mon–Fri) to a date."""
    current = start
    added   = 0
    while added < days:
        current += timedelta(days=1)
        if current.weekday() < 5:   # Monday=0 … Friday=4
            added += 1
    return current


# ─────────────────────────────────────────────────────────────────────
# AGENT
# ─────────────────────────────────────────────────────────────────────

def track_agent(state: ConversationState) -> ConversationState:
    user_id  = state["user_id"]
    order_id = state.get("active_order_id")

    if not order_id:
        state["messages"].append(
            AIMessage(
                content=(
                    "Sure 🙂 Please share your **Order ID** so I can track it.\n\n"
                    "*Example: Track ORD-XXXX*"
                )
            )
        )
        return state

    order = get_order_status(order_id, user_id)

    if not order:
        state["messages"].append(
            AIMessage(
                content=(
                    f"❌ No order with ID **{order_id}** was found on your account.\n\n"
                    "Please double-check the Order ID and try again."
                )
            )
        )
        return state

    status, product_name, quantity, order_date = order

    order_dt       = datetime.fromisoformat(order_date)
    order_date_str = order_dt.strftime("%d %b %Y")
    eta_start      = add_business_days(order_dt, 5)
    eta_end        = add_business_days(order_dt, 7)
    eta_str        = f"{eta_start.strftime('%d %b %Y')} – {eta_end.strftime('%d %b %Y')}"

    state["messages"].append(
        AIMessage(
            content=(
                f"📦 **Order Status — {order_id}**\n\n"
                f"🛍️ Product: {product_name}\n"
                f"🔢 Quantity: {quantity}\n"
                f"📅 Ordered On: {order_date_str}\n"
                f"📍 Status: **{status}**\n"
                f"🚚 Estimated Delivery: **{eta_str}**"
            )
        )
    )
    return state