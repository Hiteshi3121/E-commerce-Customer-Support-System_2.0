# backend/graph/workflow.py
"""
LangGraph Workflow
═══════════════════
Message saving is handled entirely by main.py:
  - HumanMessage saved BEFORE graph runs (router can read it as context)
  - AIMessage(s) saved AFTER graph completes (no duplicates)

persist_memory node is removed. Agents connect directly to END.
"""

from langgraph.graph import StateGraph, END
from backend.graph.state import ConversationState
from backend.graph.router import intent_router, route_by_next_node
from backend.agents.order_agent import order_agent
from backend.agents.track_agent import track_agent
from backend.agents.ticket_agent import ticket_agent
from backend.agents.return_agent import return_agent
from backend.rag.faq_agent import faq_llm
from dotenv import load_dotenv

load_dotenv()


def create_workflow():
    graph = StateGraph(ConversationState)

    # Entry point
    graph.set_entry_point("intent_router")

    # Nodes
    graph.add_node("intent_router", intent_router)
    graph.add_node("faq_llm",       faq_llm)
    graph.add_node("order_agent",   order_agent)
    graph.add_node("track_agent",   track_agent)
    graph.add_node("ticket_agent",  ticket_agent)
    graph.add_node("return_agent",  return_agent)

    # Conditional routing from router
    graph.add_conditional_edges(
        "intent_router",
        route_by_next_node,
        {
            "faq_agent":    "faq_llm",
            "order_agent":  "order_agent",
            "track_agent":  "track_agent",
            "ticket_agent": "ticket_agent",
            "return_agent": "return_agent",
            "END":          END,
        }
    )

    # All agents go directly to END — no persist_memory middleman
    for node in ["faq_llm", "order_agent", "track_agent", "ticket_agent", "return_agent"]:
        graph.add_edge(node, END)

    return graph.compile()