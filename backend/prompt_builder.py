# backend/prompt_builder.py
"""
Shared Prompt Builder
══════════════════════
Utility for building LLM message lists with short-term memory context.
Uses the sliding-window short-term memory (last 10 messages = 5 turns)
instead of loading the full history.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from backend.memory import get_short_term_context


def build_prompt(session_id: str, system_prompt: str, user_question: str, context: str = None) -> list:
    """
    Build a complete message list for an LLM call:
      [SystemMessage] + [short-term memory history] + [current HumanMessage]

    Args:
        session_id:    Session identifier for memory lookup.
        system_prompt: Role + instructions for the LLM.
        user_question: The current user message.
        context:       Optional RAG context to inject alongside the question.

    Returns:
        List of LangChain messages ready to pass to llm.invoke().
    """
    history  = get_short_term_context(session_id)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(history)

    if context:
        messages.append(
            HumanMessage(
                content=(
                    f"Customer question:\n{user_question}\n\n"
                    f"Reference information:\n{context}\n\n"
                    "Answer using only the reference information provided:"
                )
            )
        )
    else:
        messages.append(HumanMessage(content=user_question))

    return messages