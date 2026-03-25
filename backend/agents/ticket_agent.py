# backend/agents/ticket_agent.py
"""
Ticket Agent
═════════════
Creates support tickets and handles human escalation.

Key improvements over v1.0:
  • Escalation is now driven entirely by the router's LLM decision.
    No hardcoded keyword list here.
  • Injects short-term memory so issue is extracted from full conversation context.
  • Uses json.loads() instead of eval() (security fix).
  • Per-function DB connection (thread safety fix).
"""

import json
import uuid

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

def generate_ticket_id() -> str:
    return f"TCK-{uuid.uuid4().hex[:6].upper()}"


def create_ticket(ticket_num: str, user_id: str, order_id, issue: str, status: str):
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO support_tickets (ticket_num, user_id, order_id, issue, status)
           VALUES (?, ?, ?, ?, ?)""",
        (ticket_num, user_id, order_id, issue, status)
    )
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────
# ISSUE EXTRACTION PROMPT  —  Short-term memory + chain-of-thought
# ─────────────────────────────────────────────────────────────────────

ISSUE_EXTRACTION_PROMPT = """\
You are a support ticket assistant for NovaCart, an e-commerce platform.

Your task: extract the customer's issue clearly from the conversation.

Reasoning steps:
  1. Read the conversation history — what problem has the customer been trying to communicate?
  2. Read the latest message for any specific complaint or issue.
  3. Combine both into a clear, concise issue description (1-2 sentences).
  4. If the issue is genuinely unclear even with full context, return null.

Output ONLY valid JSON. No markdown, no explanation:
{"issue": "<clear issue description>" or null}
"""


# ─────────────────────────────────────────────────────────────────────
# AGENT
# ─────────────────────────────────────────────────────────────────────

def ticket_agent(state: ConversationState) -> ConversationState:
    user_id           = state["user_id"]
    order_id          = state.get("active_order_id")
    escalation_reason = state.get("escalation_reason")
    user_text         = get_last_human_message(state["messages"])
    session_id        = state["session_id"]
    ticket_num        = generate_ticket_id()

    # ── ESCALATION PATH ───────────────────────────────────────────────
    # Triggered when router LLM detected frustration / human request.
    # No keyword list needed here.
    if escalation_reason:
        create_ticket(
            ticket_num=ticket_num,
            user_id=user_id,
            order_id=order_id,
            issue=f"Escalated: {escalation_reason}",
            status="ESCALATED"
        )
        state["messages"].append(
            AIMessage(
                content=(
                    "👤 **Your request has been escalated to a human support agent.**\n\n"
                    f"🎫 Ticket Number: **{ticket_num}**\n"
                    f"📋 Reason: {escalation_reason}\n\n"
                    "Our team will review your conversation and get back to you shortly.\n\n"
                    "If you don't hear back within 12 hours, please reach us at:\n"
                    "📞 *+91 98765 43210 (8 AM – 10 PM IST)*\n"
                    "📧 *support@novacart.in*"
                )
            )
        )
        return state

    # ── NORMAL TICKET PATH ────────────────────────────────────────────
    if not order_id:
        state["messages"].append(
            AIMessage(
                content=(
                    "Sure 📝 Please share your **Order ID** along with the issue.\n\n"
                    "*Example: Raise a ticket for ORD-XXXX — item arrived damaged*"
                )
            )
        )
        return state

    # Extract issue using short-term memory context
    history  = get_short_term_context(session_id)
    messages = [SystemMessage(content=ISSUE_EXTRACTION_PROMPT)]
    messages.extend(history)
    messages.append(HumanMessage(content=user_text))

    try:
        raw   = llm.invoke(messages).content.strip()
        raw   = raw.replace("```json", "").replace("```", "").strip()
        data  = json.loads(raw)
        issue = data.get("issue")
    except Exception:
        issue = None

    if not issue or len(str(issue).strip()) < 5:
        state["messages"].append(
            AIMessage(
                content=(
                    "📝 Could you briefly describe the issue you're facing with your order?\n\n"
                    "*Example: The product arrived damaged*"
                )
            )
        )
        return state

    create_ticket(
        ticket_num=ticket_num,
        user_id=user_id,
        order_id=order_id,
        issue=issue,
        status="OPEN"
    )

    state["messages"].append(
        AIMessage(
            content=(
                "🎫 **Support Ticket Created Successfully!**\n\n"
                f"🆔 Order ID: {order_id}\n"
                f"🎫 Ticket Number: **{ticket_num}**\n"
                f"📝 Issue: {issue}\n\n"
                "Our support team will review this and respond to you soon."
            )
        )
    )
    return state