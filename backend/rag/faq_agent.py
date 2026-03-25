# backend/rag/faq_agent.py
"""
FAQ Agent (RAG)
════════════════
Answers company-related questions using NovaCart's document store via MCP.

Key improvements over v1.0:
  • Uses get_short_term_context() instead of load_memory() with no limit.
  • Improved system prompt — clearer grounding rules, better tone guidance.
  • Follow-up questions are resolved in context (e.g. "what about refunds?"
    after discussing delivery policy is understood correctly).
"""

import asyncio

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from backend.mcp_client import search_company_docs
from backend.graph.state import get_last_human_message
from backend.memory import get_short_term_context
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)


FAQ_SYSTEM_PROMPT = """\
You are NovaCart's friendly AI customer support assistant.

Your role is to answer customer questions about NovaCart using ONLY the reference information retrieved from NovaCart's official documents.

Guidelines:
  • Read the reference information carefully before composing your answer.
  • If the answer is present in the reference information, respond clearly and concisely (2-3 sentences).
  • If the answer is NOT in the reference information, reply warmly using the customer's name
    (provided below) and say you don't have that detail, e.g.:
    "I'm sorry {username}, I don't have that specific detail right now.
     For more help, please reach us at support@novacart.in."
  • Never guess, invent, or use general knowledge. Only use what's in the provided reference.
  • Use the conversation history to understand follow-up questions in context.
    Example: "what about refunds?" after a delivery policy discussion → answer the refund policy.
  • Keep tone warm, polite, and personal — always address the customer by name.
  • Do not copy text verbatim from the reference — summarize in your own words.
"""


def faq_llm(state):
    session_id    = state["session_id"]
    user_question = get_last_human_message(state["messages"])
    username      = state.get("username") or "there"

    # ── Step 1: Retrieve from RAG (MCP → ChromaDB) ────────────────
    tool_result = asyncio.run(search_company_docs(user_question))

    if not tool_result or not tool_result.strip():
        state["messages"].append(
            AIMessage(
                content=(
                    f"I'm sorry, {username}! I don't have specific information about that right now.\n"
                    "Please reach our support team at 📧 *support@novacart.in* for detailed assistance."
                )
            )
        )
        return state

    # Save raw RAG context so observability can compute grounding score
    state["rag_context"] = tool_result

    # ── Step 2: Build prompt with short-term memory context ───────
    history  = get_short_term_context(session_id)
    messages = [SystemMessage(content=FAQ_SYSTEM_PROMPT)]
    messages.extend(history)
    messages.append(
        HumanMessage(
            content=(
                f"Customer name: {username}\n"
                f"Customer question:\n{user_question}\n\n"
                f"Reference information from NovaCart documents:\n{tool_result}\n\n"
                "Answer using only the reference information. Address the customer by name."
            )
        )
    )

    # ── Step 3: Generate grounded answer ──────────────────────────
    response = llm.invoke(messages)
    state["messages"].append(AIMessage(content=response.content))
    return state