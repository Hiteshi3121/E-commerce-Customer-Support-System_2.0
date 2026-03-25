# backend/observability.py
"""
Observability  ─  NovaCart AI
═══════════════════════════════════════════════════════════════════════
Two-layer observability system:

  LAYER 1 — LangSmith (automatic LangGraph tracing)
  ─────────────────────────────────────────────────
  Enabled by setting LANGCHAIN_TRACING_V2=true in .env
  LangSmith auto-instruments every LangGraph node and records:
    • Node entry / exit state
    • LLM input + output + token counts
    • Node latency
    • Full trace tree per user message
  No code changes needed in agents — it's pure environment config.

  LAYER 2 — Custom Metrics (SQLite, persisted)
  ────────────────────────────────────────────
  Stored in the same orders.db for simplicity.
  One row per /chat turn. Tracks:
    • session_id, user_id, username
    • intent routed to
    • agent used (order_agent, faq_agent, etc.)
    • turn latency (ms)
    • estimated token count
    • guardrail events fired (JSON list)
    • order committed (bool)
    • faq_grounding_score (0–1)
    • timestamp

  Public API:
    setup_langsmith()           → call once at startup (reads .env)
    init_observability_db()     → create metrics table (idempotent)
    log_turn(...)               → called after every /chat turn
    get_metrics_summary()       → returns dict for dashboard
    get_recent_turns(n)         → last N turns for table display
"""

from __future__ import annotations
import json
import os
import sqlite3
import time
from typing import Optional

from backend.db import DB_NAME, get_connection


# ─────────────────────────────────────────────────────────────────────
# LANGSMITH SETUP
# ─────────────────────────────────────────────────────────────────────

def setup_langsmith() -> bool:
    """
    Enables LangSmith tracing if LANGCHAIN_API_KEY is present in env.

    Add these to your .env file to activate:
        LANGCHAIN_TRACING_V2=true
        LANGCHAIN_API_KEY=lsv2_pt_your_key_here
        LANGCHAIN_PROJECT=novacart-ai

    Returns True if tracing was successfully enabled, False otherwise.
    Called once at FastAPI startup before create_workflow().
    """
    api_key = os.getenv("LANGCHAIN_API_KEY", "")
    if not api_key:
        print("[Observability] LangSmith not configured — set LANGCHAIN_API_KEY in .env to enable tracing.")
        return False

    # These env vars are read by LangChain/LangGraph automatically
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
    os.environ["LANGCHAIN_PROJECT"]    = os.getenv("LANGCHAIN_PROJECT", "novacart-ai")
    os.environ["LANGCHAIN_ENDPOINT"]   = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

    print(f"[Observability] LangSmith tracing enabled → project: {os.environ['LANGCHAIN_PROJECT']}")
    return True


# ─────────────────────────────────────────────────────────────────────
# METRICS DB INIT
# ─────────────────────────────────────────────────────────────────────

def init_observability_db() -> None:
    """
    Creates the turn_metrics table in orders.db (idempotent).
    Called once at FastAPI startup alongside init_db().
    """
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS turn_metrics (
            id                  INTEGER   PRIMARY KEY AUTOINCREMENT,
            session_id          TEXT      NOT NULL,
            user_id             TEXT      NOT NULL,
            username            TEXT,
            user_message        TEXT,
            intent              TEXT,
            agent_used          TEXT,
            latency_ms          INTEGER,
            estimated_tokens    INTEGER,
            guardrail_events    TEXT      DEFAULT '[]',
            input_blocked       INTEGER   DEFAULT 0,
            output_modified     INTEGER   DEFAULT 0,
            order_committed     INTEGER   DEFAULT 0,
            faq_grounding_score REAL      DEFAULT -1,
            turn_timestamp      DATETIME  DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────
# LOG TURN
# ─────────────────────────────────────────────────────────────────────

def log_turn(
    session_id:          str,
    user_id:             str,
    username:            str,
    user_message:        str,
    intent:              str,
    agent_used:          str,
    latency_ms:          int,
    guardrail_events:    list[str]        = None,
    input_blocked:       bool             = False,
    output_modified:     bool             = False,
    order_committed:     bool             = False,
    faq_grounding_score: float            = -1.0,
) -> None:
    """
    Persists one turn's metrics to the turn_metrics table.
    Called at the end of every /chat endpoint invocation.
    Silently swallows errors so a metrics failure never crashes the chat.
    """
    try:
        events_json = json.dumps(guardrail_events or [])
        # Rough token estimate: 1 token ≈ 4 chars of user message + response combined
        estimated_tokens = max(1, len(user_message) // 4)

        conn   = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO turn_metrics
                (session_id, user_id, username, user_message, intent, agent_used,
                 latency_ms, estimated_tokens, guardrail_events, input_blocked,
                 output_modified, order_committed, faq_grounding_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id, user_id, username,
                user_message[:500],    # cap to avoid huge rows
                intent, agent_used, latency_ms, estimated_tokens,
                events_json,
                int(input_blocked), int(output_modified),
                int(order_committed), faq_grounding_score,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        print(f"[Observability] log_turn error (non-fatal): {exc}")


# ─────────────────────────────────────────────────────────────────────
# METRICS SUMMARY  (for dashboard)
# ─────────────────────────────────────────────────────────────────────

def get_metrics_summary() -> dict:
    """
    Aggregates all turn_metrics rows into a summary dict for the
    Streamlit observability dashboard.

    Returns:
        {
          total_turns, total_sessions, total_users,
          avg_latency_ms, p95_latency_ms,
          total_estimated_tokens,
          intent_distribution: {intent: count},
          agent_distribution:  {agent: count},
          guardrail_events:    {event_type: count},
          input_blocked_count, output_modified_count,
          order_committed_count,
          order_completion_rate,   # orders / order flows started
          faq_fallback_rate,       # faq turns / total turns
          avg_faq_grounding_score,
          langsmith_enabled,
          langsmith_project,
        }
    """
    try:
        conn   = get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM turn_metrics")
        total_turns = cursor.fetchone()[0]

        if total_turns == 0:
            conn.close()
            return _empty_summary()

        cursor.execute("SELECT COUNT(DISTINCT session_id) FROM turn_metrics")
        total_sessions = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT user_id) FROM turn_metrics")
        total_users = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(latency_ms), SUM(estimated_tokens) FROM turn_metrics")
        row = cursor.fetchone()
        avg_latency    = round(row[0] or 0)
        total_tokens   = row[1] or 0

        # p95 latency
        cursor.execute("SELECT latency_ms FROM turn_metrics ORDER BY latency_ms ASC")
        latencies = [r[0] for r in cursor.fetchall() if r[0] is not None]
        p95_latency = _percentile(latencies, 95) if latencies else 0

        # Intent distribution
        cursor.execute("SELECT intent, COUNT(*) FROM turn_metrics GROUP BY intent ORDER BY COUNT(*) DESC")
        intent_dist = {r[0]: r[1] for r in cursor.fetchall() if r[0]}

        # Agent distribution
        cursor.execute("SELECT agent_used, COUNT(*) FROM turn_metrics GROUP BY agent_used ORDER BY COUNT(*) DESC")
        agent_dist = {r[0]: r[1] for r in cursor.fetchall() if r[0]}

        # Guardrail events — unpack JSON arrays
        cursor.execute("SELECT guardrail_events FROM turn_metrics WHERE guardrail_events != '[]'")
        events_counter: dict[str, int] = {}
        for (events_json,) in cursor.fetchall():
            try:
                for ev in json.loads(events_json):
                    # ev may be "prompt_injection" or "low_grounding_score: 0.21"
                    key = ev.split(":")[0].strip()
                    events_counter[key] = events_counter.get(key, 0) + 1
            except Exception:
                pass

        # Blocked / modified counts
        cursor.execute("SELECT SUM(input_blocked), SUM(output_modified) FROM turn_metrics")
        r = cursor.fetchone()
        input_blocked_count   = r[0] or 0
        output_modified_count = r[1] or 0

        # Order metrics
        cursor.execute("SELECT SUM(order_committed) FROM turn_metrics")
        order_committed_count = cursor.fetchone()[0] or 0

        order_agent_turns = agent_dist.get("order_agent", 0)
        order_completion_rate = (
            round(order_committed_count / order_agent_turns * 100, 1)
            if order_agent_turns > 0 else 0.0
        )

        # FAQ grounding
        cursor.execute(
            "SELECT AVG(faq_grounding_score) FROM turn_metrics "
            "WHERE faq_grounding_score >= 0"
        )
        avg_grounding = cursor.fetchone()[0]
        avg_grounding = round(avg_grounding, 3) if avg_grounding is not None else None

        faq_turns        = intent_dist.get("faq", 0)
        faq_fallback_rate = round(faq_turns / total_turns * 100, 1) if total_turns > 0 else 0.0

        conn.close()

        return {
            "total_turns":            total_turns,
            "total_sessions":         total_sessions,
            "total_users":            total_users,
            "avg_latency_ms":         avg_latency,
            "p95_latency_ms":         p95_latency,
            "total_estimated_tokens": total_tokens,
            "intent_distribution":    intent_dist,
            "agent_distribution":     agent_dist,
            "guardrail_events":       events_counter,
            "input_blocked_count":    input_blocked_count,
            "output_modified_count":  output_modified_count,
            "order_committed_count":  order_committed_count,
            "order_completion_rate":  order_completion_rate,
            "faq_fallback_rate":      faq_fallback_rate,
            "avg_faq_grounding_score": avg_grounding,
            "langsmith_enabled":      bool(os.getenv("LANGCHAIN_API_KEY")),
            "langsmith_project":      os.getenv("LANGCHAIN_PROJECT", "novacart-ai"),
        }

    except Exception as exc:
        print(f"[Observability] get_metrics_summary error: {exc}")
        return _empty_summary()


def get_recent_turns(n: int = 20) -> list[dict]:
    """
    Returns the last N turns as a list of dicts for the dashboard table.
    """
    try:
        conn   = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT turn_timestamp, username, user_message, intent, agent_used,
                   latency_ms, input_blocked, output_modified, order_committed,
                   guardrail_events
            FROM   turn_metrics
            ORDER  BY id DESC
            LIMIT  ?
            """,
            (n,),
        )
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "timestamp":       r[0][:16] if r[0] else "",
                "user":            r[1] or "",
                "message":         (r[2] or "")[:60] + ("..." if len(r[2] or "") > 60 else ""),
                "intent":          r[3] or "",
                "agent":           r[4] or "",
                "latency_ms":      r[5] or 0,
                "input_blocked":   bool(r[6]),
                "output_modified": bool(r[7]),
                "order_committed": bool(r[8]),
                "guardrail_events": json.loads(r[9]) if r[9] else [],
            }
            for r in rows
        ]
    except Exception as exc:
        print(f"[Observability] get_recent_turns error: {exc}")
        return []


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def _percentile(data: list, p: int) -> int:
    if not data:
        return 0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]


def _empty_summary() -> dict:
    return {
        "total_turns": 0, "total_sessions": 0, "total_users": 0,
        "avg_latency_ms": 0, "p95_latency_ms": 0,
        "total_estimated_tokens": 0,
        "intent_distribution": {}, "agent_distribution": {},
        "guardrail_events": {},
        "input_blocked_count": 0, "output_modified_count": 0,
        "order_committed_count": 0, "order_completion_rate": 0.0,
        "faq_fallback_rate": 0.0, "avg_faq_grounding_score": None,
        "langsmith_enabled": False, "langsmith_project": "novacart-ai",
    }