# backend/agents/return_agent.py
"""
Return Agent
═════════════
Handles product return requests.

Key improvements over v1.0:
  • Injects short-term memory to extract return reason from conversation context
    (e.g. user mentioned "broken product" 2 turns ago — agent now picks that up).
  • Uses json.loads() instead of eval() (security fix).
  • Per-function DB connection (thread safety fix).
  • Prevents duplicate return requests.
"""

import json

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from backend.db import get_connection
from backend.graph.state import ConversationState, get_last_human_message
from backend.memory import get_short_term_context
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)


# ─────────────────────────────────────────────────────────────────────
# DB TOOLS
# ─────────────────────────────────────────────────────────────────────

def validate_order(order_id: str, user_id: str):
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, status FROM orders WHERE order_id = ? AND user_id = ?",
        (order_id, user_id)
    )
    row = cursor.fetchone()
    conn.close()
    return row


def create_return_request(user_id: str, order_id: str, reason: str):
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE orders SET status = ?, return_reason = ? WHERE order_id = ?",
        ("RETURN_REQUESTED", reason, order_id)
    )
    cursor.execute(
        "INSERT INTO returns (user_id, order_id, reason, status) VALUES (?, ?, ?, ?)",
        (user_id, order_id, reason, "RETURN_REQUESTED")
    )
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────
# REASON EXTRACTION PROMPT  —  Short-term memory + chain-of-thought
# ─────────────────────────────────────────────────────────────────────

REASON_EXTRACTION_PROMPT = """\
You are a return processing assistant for NovaCart, an e-commerce platform.

Your task: identify the return reason from the conversation.

Reasoning steps:
  1. Scan the conversation history — has the user described any problem with the product?
     Examples: "it arrived broken", "wrong item", "not what I ordered", "bad quality"
  2. Check the latest message for an explicit reason.
  3. Combine context into a clear, concise return reason (one sentence).
  4. If no specific reason can be found, use: "Customer requested return"

Output ONLY valid JSON. No markdown, no explanation:
{"reason": "<return reason>"}
"""


# ─────────────────────────────────────────────────────────────────────
# AGENT
# ─────────────────────────────────────────────────────────────────────

def return_agent(state: ConversationState) -> ConversationState:
    order_id   = state.get("active_order_id")
    user_id    = state["user_id"]
    user_text  = get_last_human_message(state["messages"])
    session_id = state["session_id"]

    if not order_id:
        state["messages"].append(
            AIMessage(
                content=(
                    "Sure 👍 Please share your **Order ID** to start the return.\n\n"
                    "*Example: Return ORD-XXXX — item arrived damaged*"
                )
            )
        )
        return state

    # Validate order ownership first
    order = validate_order(order_id, user_id)

    if not order:
        state["messages"].append(
            AIMessage(
                content=(
                    f"❌ Order **{order_id}** was not found or does not belong to your account.\n\n"
                    "Please check the Order ID and try again."
                )
            )
        )
        return state

    _, order_status = order

    # Prevent duplicate return requests
    if order_status == "RETURN_REQUESTED":
        state["messages"].append(
            AIMessage(
                content=(
                    f"ℹ️ A return request for **Order {order_id}** is already in progress.\n\n"
                    "Our team is processing it. You will receive an update shortly."
                )
            )
        )
        return state

    # Extract return reason using short-term memory context
    history  = get_short_term_context(session_id)
    messages = [SystemMessage(content=REASON_EXTRACTION_PROMPT)]
    messages.extend(history)
    messages.append(HumanMessage(content=user_text))

    try:
        raw    = llm.invoke(messages).content.strip()
        raw    = raw.replace("```json", "").replace("```", "").strip()
        data   = json.loads(raw)
        reason = data.get("reason", "Customer requested return")
    except Exception:
        reason = "Customer requested return"

    if not reason:
        reason = "Customer requested return"

    create_return_request(user_id, order_id, reason)

    state["messages"].append(
        AIMessage(
            content=(
                "↩️ **Return Request Raised Successfully!**\n\n"
                f"📦 Order ID: **{order_id}**\n"
                f"📝 Reason: {reason}\n\n"
                "Our team will process your return and contact you shortly."
            )
        )
    )
    return state