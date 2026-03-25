# backend/graph/router.py
"""
Context-Aware Intent Router  ─  NovaCart v3.0
══════════════════════════════════════════════════
New capabilities in this version:

1. LONG-TERM MEMORY INJECTION
   user_context (past session summaries) from state is injected into
   the system prompt. The LLM can now reference the user's history:
   "I see you had a delivery issue last time — let me escalate this directly."

2. ANALYTICS INTENT
   New intent: user_analytics
   Triggered by questions like:
     - "which is my most ordered product?"
     - "how many tickets have I raised?"
     - "how many orders have I returned?"
     - "give me a summary of my activity"
   Handled directly in the router via SQL — no new agent needed.

3. SENTIMENT-AWARE ROUTING
   If past summaries show FRUSTRATED sentiment, the router is more
   likely to proactively offer escalation instead of repeating the
   same self-service flow.
"""

import re
import json

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from backend.graph.state import get_last_human_message
from backend.memory import get_short_term_context
from backend.db import (
    get_user_orders, get_user_tickets, get_user_returns,
    get_user_analytics, format_analytics_response,
    get_specific_analytics_response, get_orders_by_date,
    get_username,
)
from backend.products import format_catalog_response
from dotenv import load_dotenv

load_dotenv()

ORDER_ID_REGEX = r"(ORD-[A-Za-z0-9]+)"

# ─────────────────────────────────────────────────────────────────────
# RULE-BASED PRE-CHECKS
# These fire BEFORE the LLM so llama-3.1-8b-instant cannot misclassify
# obvious order-placement and product-list requests as faq.
# Same pattern already used for analytics / ordinal resolution.
# ─────────────────────────────────────────────────────────────────────

# Compiled once at module load for speed

# Order placement — covers every phrasing seen in production + the screenshot examples
_ORDER_PATTERNS = [
    # "order 2 laptops", "order a stool", "order me a chair"
    re.compile(r"^\s*order\s+(\d+|a|an|some|the|me|my|one|two|three|four|five)\b", re.I),
    # "place an order for / of / on"
    re.compile(r"place\s+an?\s+order", re.I),
    # "i want to order / buy / purchase", "i'd like to order", "i wanna buy"
    re.compile(r"i\s+(want|wanna|would\s*like|'d\s*like)\s+(to\s+)?(order|buy|purchase)", re.I),
    # "can you order / buy / place an order"
    re.compile(r"can\s+you\s+(order|buy|purchase|place)", re.I),
    # "buy a / buy 2 / buy some"
    re.compile(r"^\s*buy\s+(a|an|\d+|some|me|one|two|three)\b", re.I),
    # "i need to order / buy"
    re.compile(r"i\s+need\s+to\s+(order|buy|purchase)", re.I),
    # "get me a / get me 2"
    re.compile(r"get\s+me\s+(a|an|\d+|some)\b", re.I),
]

# Product catalog / list requests
_PRODUCTS_CATALOG_KEYWORDS = {
    "product list", "products list", "product catalog", "product catalogue",
    "item list", "items list", "catalog", "catalogue", "inventory",
    "available products", "available items", "what you sell", "what you have",
    "what do you sell", "what do you have", "what you offer",
    "show products", "show me products", "show your products",
    "list products", "list your products", "list all products",
    "your products", "all products", "see products", "view products",
    "show me the products", "what products", "what items",
}

# Words that cancel a pending order mid-flow ("stop", "never mind", "cancel")
CANCEL_WORDS = {
    "stop", "cancel", "nevermind", "never mind", "forget it", "forget this",
    "don't want", "dont want", "no thanks", "nope", "nah", "exit",
    "quit", "abort", "skip", "scratch that", "go back", "not now",
    "i changed my mind", "changed my mind", "discard", "drop it",
}

# Detect "do you know my name" / "what is my name" type queries
_NAME_PATTERNS = [
    re.compile(r"(do you know|what is|what'?s|tell me|remember) (my name|who i am)", re.I),
    re.compile(r"my name\??$", re.I),
    re.compile(r"who am i\??$", re.I),
]

def _is_cancel_pending(text: str) -> bool:
    """Returns True if the user wants to cancel/abandon the pending order."""
    lowered = text.strip().lower()
    return any(w in lowered for w in CANCEL_WORDS)

def _is_name_query(text: str) -> bool:
    """Returns True if user is asking the bot whether it knows their name."""
    return any(p.search(text.strip()) for p in _NAME_PATTERNS)


def _is_place_order_intent(text: str) -> bool:
    """Rule-based detection for unambiguous order placement requests.
    Bypasses LLM classification entirely — prevents faq misrouting.
    Uses a list of compiled patterns — any match is sufficient."""
    t = text.strip()
    return any(p.search(t) for p in _ORDER_PATTERNS)


def _is_view_products_intent(text: str) -> bool:
    """Rule-based detection for product catalog / product list requests.
    Checks against a set of known keyword phrases (case-insensitive)."""
    lowered = text.strip().lower()
    # Direct keyword set match (fast path)
    if any(kw in lowered for kw in _PRODUCTS_CATALOG_KEYWORDS):
        return True
    # Pattern: "(show|display|list|view|see) ... (products|items|catalog)"
    if re.search(
        r"(show|display|list|view|see|tell\s+me\s+about)\s+(me\s+)?(the\s+|all\s+|your\s+|our\s+|available\s+)?"
        r"(products?|items?|catalog|catalogue|stock|inventory)",
        lowered
    ):
        return True
    return False

router_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)


# ─────────────────────────────────────────────────────────────────────
# SIGNAL WORD GROUPS
# ─────────────────────────────────────────────────────────────────────

ORDINAL_TO_INDEX = {
    "first": 0, "second": 1, "third": 2, "fourth": 3,
    "fifth": 4, "sixth":  5, "seventh": 6, "eighth": 7,
    "ninth": 8, "tenth":  9,
    "1st": 0, "2nd": 1, "3rd": 2, "4th": 3,
    "5th": 4, "6th": 5, "7th": 6, "8th": 7,
    "9th": 8, "10th": 9,
}
LAST_WORDS     = {"last", "latest", "recent", "newest", "previous", "most recent"}
ORDER_SIGNALS  = {"order", "orders", "purchase", "purchases", "bought", "placed",
                  "delivery", "delivered", "shipment"}
TICKET_SIGNALS = {"ticket", "tickets", "complaint", "issue", "raised", "filed"}
RETURN_SIGNALS = {"return", "returns", "returned", "refund", "sent back", "exchange"}
ANALYTICS_SIGNALS = {
    "most ordered", "how many", "total orders", "total tickets",
    "total returns", "activity summary", "my stats", "order history",
    "how often", "how much", "summary of my", "times have i",
    "many times", "frequently", "most purchased",
    # date-based analytics
    "which date", "on which date", "what date", "placed most",
    "date i placed", "when do i order", "when did i",
    # natural count phrasings
    "how many orders", "how many returns", "how many tickets",
    "my activity", "give me a summary", "activity of my",
}


# ─────────────────────────────────────────────────────────────────────
# ORDINAL RESOLVER  (Python resolves position — NOT the LLM)
# ─────────────────────────────────────────────────────────────────────

def resolve_ordinal(message: str, records: list) -> tuple:
    if not records:
        return None, None

    lowered  = message.lower()
    words    = set(lowered.split())
    from datetime import date
    today_str = str(date.today())

    is_today = "today" in lowered or "of the day" in lowered
    working  = [r for r in records if
                (r.get("order_date") or r.get("created_date") or "").startswith(today_str)
               ] if is_today else records
    if not working:
        working = records

    if words & LAST_WORDS or "most recent" in lowered:
        return working[-1], len(working)

    for word, idx in ORDINAL_TO_INDEX.items():
        if word in words:
            rec = working[idx] if idx < len(working) else working[-1]
            pos = idx + 1 if idx < len(working) else len(working)
            return rec, pos

    m = re.search(r'\border\s+(\d+)\b|\bticket\s+(\d+)\b|\breturn\s+(\d+)\b', lowered)
    if m:
        n   = int(next(g for g in m.groups() if g is not None))
        idx = n - 1
        rec = working[idx] if idx < len(working) else working[-1]
        return rec, n if idx < len(working) else len(working)

    return None, None


# ─────────────────────────────────────────────────────────────────────
# CONTEXT DETECTION
# ─────────────────────────────────────────────────────────────────────

def _needs_db_context(lowered: str) -> tuple:
    words    = set(lowered.split())
    has_pos  = bool(words & set(ORDINAL_TO_INDEX.keys())) or \
               bool(words & LAST_WORDS) or \
               "most recent" in lowered or "today" in lowered

    has_order  = bool(words & ORDER_SIGNALS)
    has_ticket = bool(words & TICKET_SIGNALS)
    has_return = bool(words & RETURN_SIGNALS)

    # Detect when "order" is used as an object of returning, not as a purchase intent.
    # e.g. "which is my last order i have returned" — user means RETURN, not ORDER.
    is_return_sentence = any(p in lowered for p in [
        "have returned", "i returned", "orders returned", "returned order",
        "orders have i returned", "how many orders have i returned",
        "which order have i returned", "last order i have returned",
    ])

    fetch_returns = (has_pos and has_return) or any(p in lowered for p in [
        "my return", "last return", "first return", "return history",
        "have returned", "which return", "returned",
    ])

    # If user clearly means returns, disable order fetching to prevent wrong branch
    fetch_orders = False if is_return_sentence else (
        (has_pos and has_order) or any(p in lowered for p in [
            "my order", "my orders", "last order", "first order", "my last", "my first",
            "order history", "my purchase", "what did i order", "which order",
        ])
    )

    fetch_tickets = has_ticket or any(p in lowered for p in [
        "my ticket", "last ticket", "first ticket", "show ticket", "recent ticket",
    ])

    fetch_analytics = any(sig in lowered for sig in ANALYTICS_SIGNALS) or \
                      any(p in lowered for p in [
                          "most ordered product", "how many orders",
                          "how many tickets", "how many returns",
                          "my activity", "my stats", "frequently ordered",
                          "times have i ordered", "how many times",
                          "give me a summary", "summary of my activity",
                          "which date", "on which date", "placed most",
                          "most of my orders", "date i placed",
                      ])

    return fetch_orders, fetch_tickets, fetch_returns, fetch_analytics


# ─────────────────────────────────────────────────────────────────────
# FORMATTERS
# ─────────────────────────────────────────────────────────────────────

def _fmt_single_order(r, pos):
    return (f"Resolved order (position {pos}):\n"
            f"  Order ID: {r['order_id']} | Product: {r['product_name']} × {r['quantity']}"
            f" | Status: {r['status']} | Date: {(r['order_date'] or '')[:10]}")

def _fmt_single_ticket(r, pos):
    return (f"Resolved ticket (position {pos}):\n"
            f"  Ticket: {r['ticket_num']} | Order: {r.get('order_id','N/A')}"
            f" | Issue: {r['issue']} | Status: {r['status']}")

def _fmt_single_return(r, pos):
    return (f"Resolved return (position {pos}):\n"
            f"  Order: {r['order_id']} | Reason: {r['reason']}"
            f" | Status: {r['status']} | Date: {(r['created_date'] or '')[:10]}")

def _fmt_all(records, kind):
    if not records:
        return f"No {kind} found."
    lines = []
    for i, r in enumerate(records, 1):
        if kind == "orders":
            lines.append(f"{i}. {r['order_id']} | {r['product_name']} × {r['quantity']}"
                         f" | {r['status']} | {(r['order_date'] or '')[:10]}")
        elif kind == "tickets":
            lines.append(f"{i}. {r['ticket_num']} | {r.get('order_id','N/A')}"
                         f" | {r['issue']} | {r['status']}")
        else:
            lines.append(f"{i}. {r['order_id']} | {r['reason']}"
                         f" | {r['status']} | {(r['created_date'] or '')[:10]}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# DIRECT RESPONSE BUILDERS (view_tickets, view_returns — no agent needed)
# ─────────────────────────────────────────────────────────────────────

def _tickets_response(tickets):
    if not tickets:
        return ("📋 You don't have any support tickets yet.\n\n"
                "Raise one by saying: *Raise a ticket for ORD-XXXX — your issue*")
    lines = ["🎫 **Your Support Tickets**\n"]
    for i, t in enumerate(tickets, 1):
        order_part = f"Order {t['order_id']}" if t["order_id"] else "No order linked"
        lines.append(f"**{i}. Ticket {t['ticket_num']}**\n"
                     f"   📦 {order_part}\n"
                     f"   📝 Issue: {t['issue']}\n"
                     f"   📍 Status: **{t['status']}**\n"
                     f"   📅 Date: {(t['created_date'] or '')[:10]}\n")
    return "\n".join(lines)


def _returns_response(returns):
    if not returns:
        return ("📋 You don't have any return requests yet.\n\n"
                "Start one by saying: *Return ORD-XXXX — reason*")
    lines = ["↩️ **Your Return Requests**\n"]
    for i, r in enumerate(returns, 1):
        lines.append(f"**{i}. Order {r['order_id']}**\n"
                     f"   📝 Reason: {r['reason']}\n"
                     f"   📍 Status: **{r['status']}**\n"
                     f"   📅 Date: {(r['created_date'] or '')[:10]}\n")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────

def _build_system_prompt(resolved_ctx=None, user_context=None) -> str:

    # ── Long-term memory section ──────────────────────────────────
    history_section = ""
    if user_context:
        history_section = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USER PROFILE — PAST SESSION HISTORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{user_context}

Use this to:
  • Recognise returning users and their history
  • If past sentiment was FRUSTRATED — proactively offer escalation
    instead of making them go through the standard flow again
  • Reference unresolved issues: "I see your delivery issue last time
    wasn't resolved — let me handle this differently"
  • Personalise responses based on their history
Do NOT mention you are reading summaries. Make it feel natural.
"""

    # ── Pre-resolved DB data section ─────────────────────────────
    resolved_section = ""
    if resolved_ctx:
        resolved_section = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRE-RESOLVED DATA (from database)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{resolved_ctx}

Use the IDs from this record directly. Do NOT route to faq.
"""

    return f"""\
You are the intent classification engine for NovaCart, an e-commerce customer support AI.

Read the conversation history and the latest user message. Reason step by step, then output a JSON decision.
{history_section}{resolved_section}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE INTENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• place_order      — user wants to buy/order a product
• track_order      — check order status, or ask which order by position
• return_order     — return or exchange an order
• raise_ticket     — file a support complaint
• view_tickets     — see existing support tickets
• view_returns     — see existing return requests
• view_products    — user wants to see the product catalog / product list /
                     available items ("show me products", "what do you sell",
                     "show product list", "what items are available")
• user_analytics   — questions about counts, totals, most ordered product,
                     activity summary ("how many orders", "most ordered product",
                     "how many returns", "my activity summary", "my stats")
• faq              — general company/policy question

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REASONING STEPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1 — What context exists in conversation history?
Step 2 — What does the latest message mean in that context?
Step 3 — If PRE-RESOLVED DATA is above, extract the order_id / ticket directly.
Step 4 — Confirmation check:
         needs_confirmation = true ONLY when ALL of:
           a) Message is 1-3 words, no explicit prior context
           b) Intent is an irreversible action: place_order/return_order/raise_ticket/track_order
           c) No earlier confirmation in this conversation
         = false when user said "yes/sure/confirm/go ahead", full context is clear,
           or intent is view_tickets/view_returns/user_analytics/faq
Step 5 — Escalation: frustration, anger, "talk to human", repeated failure?
         Also check USER PROFILE above — if past sessions were FRUSTRATED,
         lower your threshold for offering escalation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT — ONLY valid JSON, no markdown
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{{
  "reasoning":          "<1-2 sentences>",
  "intent":             "<one of the 8 intents>",
  "order_id":           "<ORD-XXXXX>" or null,
  "needs_confirmation": true or false,
  "proposed_action":    "<short description>" or null,
  "escalation":         true or false,
  "escalation_reason":  "<reason>" or null
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{{"reasoning": "User asked how many tickets they have raised. Analytics question.", "intent": "user_analytics", "order_id": null, "needs_confirmation": false, "proposed_action": null, "escalation": false, "escalation_reason": null}}
{{"reasoning": "User asked most ordered product. Analytics.", "intent": "user_analytics", "order_id": null, "needs_confirmation": false, "proposed_action": null, "escalation": false, "escalation_reason": null}}
{{"reasoning": "Pre-resolved shows ORD-008F1E is first order. User wants to track it.", "intent": "track_order", "order_id": "ORD-008F1E", "needs_confirmation": false, "proposed_action": null, "escalation": false, "escalation_reason": null}}
{{"reasoning": "User said only 'return'. Short, ambiguous. Asking confirmation.", "intent": "return_order", "order_id": null, "needs_confirmation": true, "proposed_action": "initiate a return request", "escalation": false, "escalation_reason": null}}
{{"reasoning": "Past sessions show FRUSTRATED sentiment. User has same delivery issue again. Proactively escalating.", "intent": "raise_ticket", "order_id": null, "needs_confirmation": false, "proposed_action": null, "escalation": true, "escalation_reason": "Recurring delivery issue — user was frustrated in previous session"}}
"""


# ─────────────────────────────────────────────────────────────────────
# MAIN LLM ROUTE
# ─────────────────────────────────────────────────────────────────────

def llm_route(session_id: str, user_text: str, user_id: str, user_context: str = None) -> dict:

    lowered = user_text.lower()
    fetch_orders, fetch_tickets, fetch_returns, fetch_analytics = _needs_db_context(lowered)

    all_orders  = get_user_orders(user_id)  if fetch_orders  else []
    all_tickets = get_user_tickets(user_id) if fetch_tickets else []
    all_returns = get_user_returns(user_id) if fetch_returns else []

    # Python resolves ordinal position — not the LLM
    resolved_ctx   = None
    resolved_order = None

    if fetch_orders and all_orders:
        rec, pos = resolve_ordinal(user_text, all_orders)
        resolved_ctx   = _fmt_single_order(rec, pos) if rec else "User's orders:\n" + _fmt_all(all_orders, "orders")
        resolved_order = rec["order_id"] if rec else None

    elif fetch_tickets and all_tickets:
        rec, pos = resolve_ordinal(user_text, all_tickets)
        resolved_ctx = _fmt_single_ticket(rec, pos) if rec else "User's tickets:\n" + _fmt_all(all_tickets, "tickets")

    elif fetch_returns and all_returns:
        rec, pos = resolve_ordinal(user_text, all_returns)
        resolved_ctx = _fmt_single_return(rec, pos) if rec else "User's returns:\n" + _fmt_all(all_returns, "returns")

    system_prompt = _build_system_prompt(resolved_ctx, user_context)
    history       = get_short_term_context(session_id)

    messages = [SystemMessage(content=system_prompt)]
    messages.extend(history)
    messages.append(HumanMessage(content=user_text))

    try:
        raw    = router_llm.invoke(messages).content.strip()
        raw    = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)

        # Sanitise: only accept real ORD-XXXXX format — reject LLM placeholder strings
        raw_oid = resolved_order or result.get("order_id")
        if raw_oid and not re.match(r'^ORD-[A-Za-z0-9]+$', str(raw_oid)):
            raw_oid = None

        return {
            "intent":             result.get("intent", "faq"),
            "order_id":           raw_oid,
            "needs_confirmation": bool(result.get("needs_confirmation", False)),
            "proposed_action":    result.get("proposed_action"),
            "escalation":         bool(result.get("escalation", False)),
            "escalation_reason":  result.get("escalation_reason"),
            "_all_tickets":       all_tickets,
            "_all_returns":       all_returns,
            "_fetch_analytics":   fetch_analytics,
        }
    except Exception:
        # Sanitise resolved_order too in error path
        safe_resolved = resolved_order if (resolved_order and re.match(r'^ORD-[A-Za-z0-9]+$', str(resolved_order))) else None
        return {
            "intent": "faq", "order_id": safe_resolved,
            "needs_confirmation": False, "proposed_action": None,
            "escalation": False, "escalation_reason": None,
            "_all_tickets": all_tickets, "_all_returns": all_returns,
            "_fetch_analytics": fetch_analytics,
        }


# ─────────────────────────────────────────────────────────────────────
# ROUTER NODE
# ─────────────────────────────────────────────────────────────────────

def intent_router(state):
    user_text    = get_last_human_message(state["messages"]).strip()
    session_id   = state["session_id"]
    user_id      = state["user_id"]
    user_context = state.get("user_context")   # ← long-term memory from state
    username     = state.get("username") or get_username(user_id)

    # ── Pending order: cancel / escape check ─────────────────────────
    # Must come BEFORE the shortcut so "stop"/"cancel" clears the pending order
    # instead of being treated as an invalid payment method.
    if state.get("pending_order") and _is_cancel_pending(user_text):
        state["pending_order"] = None
        state["messages"].append(
            AIMessage(
                content=(
                    f"No problem, {username}! 😊 Your order has been cancelled.\n\n"
                    "Whenever you're ready, just tell me what you'd like to order!"
                )
            )
        )
        state["next_node"] = "END"
        return state

    # ── Pending order: view-products escape ───────────────────────
    # If user asks to see products while a payment is pending, cancel
    # the pending order silently and show the catalog — don't trap them.
    if state.get("pending_order") and _is_view_products_intent(user_text):
        state["pending_order"] = None
        state["messages"].append(AIMessage(content=format_catalog_response()))
        state["next_node"] = "END"
        return state

    # ── Pending order shortcut (payment confirmation turn) ─────────
    # If order_agent is waiting for a payment method, bypass all intent
    # classification and send directly to order_agent.
    if state.get("pending_order"):
        state["next_node"] = "order_agent"
        return state

    # ── Rule-based: view products catalog ───────────────────────────────
    # Catches: "show me products", "what do you sell", "product list", etc.
    # Handled entirely in Python — no LLM, no RAG, reads directly from
    # the products table. Same pattern as view_tickets / view_returns.
    if _is_view_products_intent(user_text):
        state["messages"].append(AIMessage(content=format_catalog_response()))
        state["next_node"] = "END"
        return state

    # ── Rule-based: place order ───────────────────────────────────────
    # Catches clear order requests BEFORE the LLM.
    # Guard: if the message ALSO asks to see products (compound intent like
    # "i want to order earbuds, can you show which products you have"),
    # show the catalog first — don't blindly jump into ordering.
    if _is_place_order_intent(user_text) and not _is_view_products_intent(user_text):
        state["next_node"] = "order_agent"
        return state

    if user_text.isdigit():
        state["messages"].append(
            AIMessage(content="I couldn't understand that. Could you describe your query in a sentence?")
        )
        state["next_node"] = "END"
        return state

    # ── Rule-based: return request with explicit order ID ─────────
    # Fires when message contains both a valid ORD-XXXX id AND return keywords.
    # This prevents the faq→track_order override from hijacking return requests.
    _RETURN_KEYWORDS = {
        "return", "refund", "exchange", "send back", "damaged",
        "wrong item", "defective", "broken", "not working",
    }
    _early_return_match = re.search(ORDER_ID_REGEX, user_text, re.IGNORECASE)
    if _early_return_match and any(kw in user_text.lower() for kw in _RETURN_KEYWORDS):
        state["active_order_id"] = _early_return_match.group(1).upper()
        state["next_node"] = "return_agent"
        return state

    # ── Lone order ID handler ─────────────────────────────────────
    # When user sends just an order ID (nothing else), ask what they want.
    # This avoids routing to FAQ which gives irrelevant RAG responses.
    lone_id_match = re.fullmatch(r"\s*(ORD-[A-Za-z0-9]+)\s*", user_text, re.IGNORECASE)
    if lone_id_match:
        oid = lone_id_match.group(1).upper()
        state["active_order_id"] = oid
        state["messages"].append(
            AIMessage(
                content=(
                    f"I see Order ID **{oid}**. What would you like to do with it?\n\n"
                    f"• **Track** — *Track {oid}*\n"
                    f"• **Return** — *Return {oid} — reason*\n"
                    f"• **Raise a ticket** — *Ticket for {oid} — issue*"
                )
            )
        )
        state["next_node"] = "END"
        return state

    # ── Short / ambiguous word handler ────────────────────────────
    # Single words like "ok", "sure", "fine", "noted" with no actionable
    # intent go to faq_agent which retrieves random RAG docs.
    # Instead we gently re-prompt the user.
    FILLER_WORDS = {
        "ok", "okay", "sure", "alright", "fine", "noted", "got it",
        "hmm", "hm", "ah", "oh", "hi", "hello", "hey", "k", "cool",
    }
    GOODBYE_WORDS = {
        "thanks", "thank you", "thankyou", "thx", "ty", "great", "nice",
        "perfect", "awesome", "bye", "goodbye", "see you", "see ya",
        "take care", "cheers",
    }
    lowered_text = user_text.lower().strip()

    if lowered_text in GOODBYE_WORDS:
        state["messages"].append(
            AIMessage(
                content=(
                    f"You're welcome, {username}! 😊 "
                    "Have a wonderful day! Feel free to come back anytime — "
                    "I'm always here to help. 🛒"
                )
            )
        )
        state["next_node"] = "END"
        return state

    if lowered_text in FILLER_WORDS:
        state["messages"].append(
            AIMessage(
                content=(
                    f"Happy to help, {username}! 😊 What would you like to do?\n\n"
                    "• 🛍️ **Place an order** — *Order 1 laptop*\n"
                    "• 📦 **Track an order** — *Track ORD-XXXX*\n"
                    "• ↩️ **Return an order** — *Return ORD-XXXX*\n"
                    "• 🎫 **Raise a ticket** — *Ticket for ORD-XXXX*\n"
                    "• ❓ **Policy questions** — *What is your return policy?*"
                )
            )
        )
        state["next_node"] = "END"
        return state

    # ── Name query ────────────────────────────────────────────────
    # "do you know my name?" / "what is my name?" — answer directly
    if _is_name_query(user_text):
        state["messages"].append(
            AIMessage(
                content=(
                    f"Of course I do! 😊 You're **{username}**. "
                    "How can I assist you today?"
                )
            )
        )
        state["next_node"] = "END"
        return state

    # ── BUG FIX 1A: Clear stale active_order_id before each fresh routing turn ──
    # Without this, a hallucinated or wrong order_id from a previous turn
    # persists in state and bleeds into the next agent call (e.g. return/track
    # showing [ORDER-ID-ERROR] from a prior bad LLM response).
    state["active_order_id"] = None

    result            = llm_route(session_id, user_text, user_id, user_context)
    intent            = result["intent"]
    order_id          = result["order_id"]
    needs_confirmation = result["needs_confirmation"]
    proposed_action   = result["proposed_action"]
    escalation        = result["escalation"]
    escalation_reason = result["escalation_reason"]

    # Regex always wins for explicit order IDs in message
    match = re.search(ORDER_ID_REGEX, user_text)
    if match:
        order_id = match.group(1)

    # ✅ FIX 1: Reject any placeholder/garbage strings — must be real ORD-XXXXX format
    if order_id and not re.match(r'^ORD-[A-Za-z0-9]+$', str(order_id)):
        order_id = None

    if order_id:
        state["active_order_id"] = order_id
    else:
        # BUG FIX 1B: If no valid order_id was resolved this turn, explicitly
        # clear active_order_id. Without this, a hallucinated [ORDER-ID-ERROR]
        # from a previous LLM response persists in state and is passed to
        # return_agent / track_agent on the very next turn, producing the
        # "Order ID: [ORDER-ID-ERROR]" bug seen in return/track responses.
        state["active_order_id"] = None

    # ── Ordinal-resolution override ───────────────────────────────
    # When Python resolved an order ID (ordinal or explicit) but LLM said
    # "faq" (because "details", "tell me" sound informational), override.
    # ✅ FIX 2: Also check return keywords — don't blindly route to track_order.
    _RETURN_KW = {"return", "refund", "exchange", "send back", "damaged",
                  "wrong item", "defective", "broken", "not working"}
    if order_id and intent == "faq":
        if any(kw in user_text.lower() for kw in _RETURN_KW):
            intent = "return_order"
        else:
            intent = "track_order"

    # ── Escalation ────────────────────────────────────────────────
    if escalation:
        state["escalation_reason"] = escalation_reason or "User requested escalation"
        state["next_node"] = "ticket_agent"
        return state

    # ── View tickets ──────────────────────────────────────────────
    # Checked BEFORE analytics so "how many tickets" routed to view_tickets
    # (LLM decides view_tickets) takes priority over the analytics fallback.
    if intent == "view_tickets":
        state["messages"].append(AIMessage(content=_tickets_response(result["_all_tickets"])))
        state["next_node"] = "END"
        return state

    # ── View returns ──────────────────────────────────────────────
    if intent == "view_returns":
        state["messages"].append(AIMessage(content=_returns_response(result["_all_returns"])))
        state["next_node"] = "END"
        return state

    # ── View products catalog ─────────────────────────────────────
    # Handled here as a safety net if LLM classifies it (rule-based
    # check above catches it first in most cases).
    if intent == "view_products":
        state["messages"].append(AIMessage(content=format_catalog_response()))
        state["next_node"] = "END"
        return state

    # ── Analytics ─────────────────────────────────────────────────
    # Fires when:
    #   a) LLM explicitly says user_analytics, OR
    #   b) _fetch_analytics=True AND LLM said faq
    #      (LLM sometimes misclassifies analytics as faq — this is the fallback)
    # The `and intent == "faq"` guard prevents analytics from overriding
    # view_tickets / view_returns which are checked above.
    if intent == "user_analytics" or (result["_fetch_analytics"] and intent == "faq"):
        analytics = get_user_analytics(user_id)
        response  = get_specific_analytics_response(user_text, analytics)
        state["messages"].append(AIMessage(content=response))
        state["next_node"] = "END"
        return state

    # ── Confirmation gate ─────────────────────────────────────────
    if needs_confirmation and proposed_action:
        state["messages"].append(
            AIMessage(
                content=(
                    f"Just to confirm — would you like me to **{proposed_action}**?\n\n"
                    "Please reply with **yes** to proceed, or let me know what you'd like to do."
                )
            )
        )
        state["next_node"] = "END"
        return state

    # ── Order ID required but missing ─────────────────────────────
    NEEDS_ORDER_ID = ("track_order", "return_order", "raise_ticket")

    if intent in NEEDS_ORDER_ID and not order_id:
        ask_map = {
            "track_order":  "Sure 🙂 Please share your **Order ID** so I can track it.\n\n*Example: Track ORD-XXXX*",
            "return_order": "Sure 👍 Please share your **Order ID** to start the return.\n\n*Example: Return ORD-XXXX*",
            "raise_ticket": "Sure 📝 Please share your **Order ID** and describe the issue.\n\n*Example: Raise a ticket for ORD-XXXX — item arrived damaged*",
        }
        state["messages"].append(AIMessage(content=ask_map[intent]))
        state["next_node"] = "END"
        return state

    # ── Route to agent ────────────────────────────────────────────
    INTENT_MAP = {
        "place_order":  "order_agent",
        "track_order":  "track_agent",
        "return_order": "return_agent",
        "raise_ticket": "ticket_agent",
        "faq":          "faq_agent",
    }
    state["next_node"] = INTENT_MAP.get(intent, "faq_agent")
    return state


def route_by_next_node(state):
    return state.get("next_node", "END")