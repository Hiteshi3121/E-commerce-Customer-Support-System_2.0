# backend/main.py
"""
FastAPI Entry Point  ─  NovaCart AI
═════════════════════════════════════════════════════════
Memory lifecycle:

  SESSION START  (/chat/session/start)
    1. If old_session_id provided → summarize that session (background)
    2. Load user's past session summaries → build user_context string
    3. Store user_context in session so it can be injected into state

  EACH MESSAGE  (/chat)
    1. Save HumanMessage BEFORE graph (router can read it immediately)
    2. Build state dict with user_context from session store
    3. Run LangGraph
    4. Save new AI messages AFTER graph (no duplicates)

  SESSION END  (implicit — when next session starts with old_session_id)
    → summarize_session() is called for the ending session
"""

from dotenv import load_dotenv
load_dotenv()

import uuid
import time
from typing import Optional

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from backend.graph.workflow import create_workflow
from backend.db import init_db, get_username
from backend.guardrails import check_input, check_output
from backend.observability import (
    setup_langsmith, init_observability_db,
    log_turn, get_metrics_summary, get_recent_turns,
)
from backend.products import init_products_db
from backend.memory import (
    init_memory_db,
    save_memory,
    summarize_session,
    build_user_context_string,
)
from backend.auth.auth_routes import router as auth_router


# ── App setup ──────────────────────────────────────────────────────
app = FastAPI(title="NovaCart Backend")
app.include_router(auth_router)

init_db()
init_products_db()      # seed product catalog (idempotent)
init_memory_db()
init_observability_db() # create turn_metrics table (idempotent)
setup_langsmith()       # enable LangSmith tracing if API key is set

graph = create_workflow()

# In-memory store: session_id → user_context string
# Loaded once at session start, reused for every message in the session
_session_context_store: dict[str, str | None] = {}

# In-memory store: session_id → pending_order dict (or None)
# Holds product+qty between the order summary step and payment confirmation step.
# Without this, 'COD' / 'UPI' on the next turn has no context and hits FAQ.
_pending_order_store: dict = {}

# In-memory store: session_id → display username (e.g. "BhumiRaj")
_session_username_store: dict = {}


# ── Models ─────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response:   str
    session_id: str


# ── Endpoints ──────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat/session/start")
def start_chat_session(
    user_id: str,
    old_session_id: Optional[str] = None,
    background_tasks: BackgroundTasks = None
):
    """
    Starts a new chat session.

    old_session_id: the session that just ended (passed by frontend).
    If provided, the old session is summarized in the background so it
    doesn't block the new session from starting.
    """
    # ── Step 1: Summarize the ending session (non-blocking) ───────
    if old_session_id and background_tasks:
        background_tasks.add_task(summarize_session, old_session_id, user_id)

    # ── Step 2: Load long-term memory for this user ───────────────
    user_context = build_user_context_string(user_id)

    # ── Step 3: Create new session ────────────────────────────────
    session_id = f"sess_{uuid.uuid4().hex[:8]}"

    # Store user_context and username for use in every /chat call this session
    _session_context_store[session_id]  = user_context
    _session_username_store[session_id] = get_username(user_id)

    return {
        "session_id":    session_id,
        "has_history":   user_context is not None,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, user_id: str, session_id: str):
    turn_start = time.time()
    username   = _session_username_store.get(session_id, "there")

    # ── Step 1: INPUT GUARDRAIL ────────────────────────────────────
    # Runs BEFORE anything else — catches injections, PII, hard abuse.
    # If blocked, skip the graph entirely and return immediately.
    guardrail_events: list[str] = []
    input_result = check_input(req.message, username)
    if input_result.event:
        guardrail_events.append(input_result.event)
    if input_result.warnings:
        guardrail_events.extend(input_result.warnings)

    if input_result.blocked:
        latency_ms = int((time.time() - turn_start) * 1000)
        log_turn(
            session_id=session_id, user_id=user_id, username=username,
            user_message=req.message, intent="blocked", agent_used="guardrail",
            latency_ms=latency_ms, guardrail_events=guardrail_events,
            input_blocked=True,
        )
        return ChatResponse(response=input_result.reply, session_id=session_id)

    # ── Step 2: Save user message BEFORE graph ─────────────────────
    # Router calls get_short_term_context() which will now see this message
    save_memory(session_id, [HumanMessage(content=req.message)])

    # ── Step 3: Build state with long-term memory context ──────────
    user_context = _session_context_store.get(session_id)

    state = {
        "messages":          [HumanMessage(content=req.message)],
        "intent":            "",
        "user_id":           user_id,
        "session_id":        session_id,
        "active_order_id":   None,
        "escalation_reason": None,
        "next_node":         None,
        "user_context":      user_context,        # long-term memory
        "pending_order":     _pending_order_store.get(session_id),  # payment flow
        "username":          username,             # personalisation
        "rag_context":       None,                 # filled by faq_agent
        "agent_used":        None,                 # filled below from next_node
        "order_committed":   False,                # set by order_agent on success
    }

    # ── Step 4: Run LangGraph workflow ─────────────────────────────
    result = graph.invoke(state)

    # ── Step 5: Extract intent + agent for observability ──────────
    # next_node tells us which agent ran this turn
    agent_used = result.get("next_node") or "router"
    # Derive a clean intent label from next_node
    _node_to_intent = {
        "order_agent":  "place_order",
        "track_agent":  "track_order",
        "return_agent": "return_order",
        "ticket_agent": "raise_ticket",
        "faq_llm":      "faq",
        "faq_agent":    "faq",
        "END":          "shortcut",
    }
    intent = _node_to_intent.get(agent_used, agent_used)

    # ── Step 6: Save only new AI messages (no duplicates) ──────────
    new_ai_messages = [
        msg for msg in result.get("messages", [])[1:]
        if isinstance(msg, AIMessage)
    ]
    if new_ai_messages:
        save_memory(session_id, new_ai_messages)

    # ── Step 7: Persist pending_order across turns ─────────────────
    updated_pending = result.get("pending_order")
    if updated_pending:
        _pending_order_store[session_id] = updated_pending
    else:
        _pending_order_store.pop(session_id, None)

    # ── Step 8: OUTPUT GUARDRAIL ───────────────────────────────────
    raw_reply = new_ai_messages[-1].content if new_ai_messages else (
        "I'm sorry, I didn't quite catch that. Could you try rephrasing?"
    )

    output_result = check_output(
        response    = raw_reply,
        intent      = intent,
        rag_context = result.get("rag_context") or "",
        username    = username,
    )
    if output_result.warnings:
        guardrail_events.extend(output_result.warnings)

    reply = output_result.modified_response or raw_reply

    # ── Step 9: Log turn metrics ───────────────────────────────────
    latency_ms = int((time.time() - turn_start) * 1000)
    log_turn(
        session_id          = session_id,
        user_id             = user_id,
        username            = username,
        user_message        = req.message,
        intent              = intent,
        agent_used          = agent_used,
        latency_ms          = latency_ms,
        guardrail_events    = guardrail_events,
        input_blocked       = False,
        output_modified     = output_result.modified_response is not None,
        order_committed     = bool(result.get("order_committed")),
        faq_grounding_score = (
            _extract_grounding_score(output_result.warnings)
            if intent == "faq" else -1.0
        ),
    )

    return ChatResponse(response=reply, session_id=session_id)


def _extract_grounding_score(warnings: list[str]) -> float:
    """Parses 'low_grounding_score: 0.21' from guardrail warnings."""
    for w in warnings:
        if w.startswith("low_grounding_score:"):
            try:
                return float(w.split(":")[1].strip())
            except (IndexError, ValueError):
                pass
    return -1.0


@app.get("/metrics")
def get_metrics():
    """Observability endpoint — returns aggregated turn metrics for the dashboard."""
    return get_metrics_summary()


@app.get("/metrics/recent")
def get_recent(n: int = 20):
    """Returns the last N turns for the live log table in the dashboard."""
    return get_recent_turns(n)