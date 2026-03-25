# backend/graph/state.py
from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()


class ConversationState(TypedDict):
    messages:    Annotated[List[BaseMessage], add_messages]
    intent:      str
    session_id:  str          # scopes short-term memory
    user_id:     str          # persistent identity across sessions

    # Set by router each turn — not persisted between requests
    active_order_id:   Optional[str]   # resolved order ID for this turn
    escalation_reason: Optional[str]   # set when router detects frustration
    next_node:         Optional[str]   # routing decision

    # Long-term memory context — loaded by main.py at session start
    # Contains formatted string of past session summaries + user sentiment history
    # Injected into router prompt so the bot "remembers" the user across sessions
    user_context:      Optional[str]

    # ── Pending order (multi-turn payment confirmation flow) ──────────────
    # Set by order_agent after product match + before payment is collected.
    # Structure:
    #   {
    #     "product_id":   str,   # e.g. "ELEC-001"
    #     "product_name": str,   # e.g. "Wireless Bluetooth Headphones"
    #     "quantity":     int,
    #     "price":        int,   # price per unit in INR
    #   }
    # Cleared by order_agent once the order is committed to DB.
    pending_order:     Optional[dict]

    # Display username — loaded once at session start, used for personalisation
    username:          Optional[str]

    # ── Observability fields (set by agents, read by main.py) ────────────
    # rag_context: raw RAG text retrieved by faq_agent — used for grounding score
    rag_context:       Optional[str]
    # agent_used: which agent actually handled this turn (set in main.py)
    agent_used:        Optional[str]
    # order_committed: True when order_agent successfully inserted a DB row
    order_committed:   Optional[bool]


def get_last_human_message(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""