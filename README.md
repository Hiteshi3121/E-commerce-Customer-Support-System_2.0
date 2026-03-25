# NovaCart AI Assistant

An agentic e-commerce customer support chatbot built with **LangGraph**, **FastAPI**, and **Streamlit**. NovaCart handles the full customer journey — browsing products, placing orders, tracking shipments, raising returns, and answering FAQs — through a multi-agent workflow with built-in guardrails and observability.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Running the Project](#running-the-project)
- [API Endpoints](#api-endpoints)
- [Agents & Routing](#agents--routing)
- [Memory System](#memory-system)
- [Product Catalog](#product-catalog)
- [Guardrails](#guardrails)
- [Observability](#observability)
- [Evaluation Dashboard](#evaluation-dashboard)
- [Database Schema](#database-schema)
- [Environment Variables](#environment-variables)

---

## Overview

NovaCart is a production-grade AI assistant that simulates a real e-commerce support system. A user can log in, browse a product list/catalog, place an order through a guided two-turn payment flow, track existing orders, raise return requests, and get answers to company policy questions — all through natural language.

The system is built around a **LangGraph state machine** that routes each message to the correct specialist agent. All routing decisions are made with a combination of deterministic rule-based pre-checks and an LLM fallback, so common intents never hit the model unnecessarily. xxxx

---

## Architecture

```
User Message
    │
    ▼
[Input Guardrail]  ──── blocked ──── polite reply + log_turn()
    │ safe
    ▼
[FastAPI /chat endpoint]
    │
    ▼
[LangGraph Workflow]
    │
    ├── intent_router  (rule-based pre-checks → LLM fallback)
    │       │
    │       ├──► order_agent   (two-turn: search → card → payment → commit)  xxxx
    │       ├──► track_agent   (order status + ETA, no LLM)
    │       ├──► return_agent  (LLM extracts reason → DB insert)
    │       ├──► ticket_agent  (LLM extracts issue → support ticket)
    │       └──► faq_llm       (RAG via MCP → ChromaDB → Groq)
    │
    ▼
[Output Guardrail]  (order ID check, length cap, grounding score)
    │
    ▼
[log_turn()]  →  SQLite turn_metrics
    │
    ▼
Reply to user
```

**LangSmith** auto-instruments every LangGraph node in parallel — no code changes in agents required.

---

## Project Structure

```
novacart/
├── backend/
│   ├── main.py                  # FastAPI entry point, guardrail + observability wiring
│   ├── db.py                    # SQLite connection, schema init, migrations
│   ├── products.py              # 42-product catalog, fuzzy search, payment resolver
│   ├── memory.py                # Short-term (sliding window) + long-term (summaries) memory
│   ├── guardrails.py            # Input + output safety checks
│   ├── observability.py         # LangSmith setup + SQLite turn_metrics
│   ├── prompt_builder.py        # Shared LLM prompt builder with memory injection
│   ├── mcp_client.py            # FastMCP client — calls ChromaDB RAG tool
│   │
│   ├── agents/
│   │   ├── order_agent.py       # Two-turn product search + payment state machine
│   │   ├── track_agent.py       # Order tracking (rule-based, no LLM)
│   │   ├── return_agent.py      # Return request handler
│   │   └── ticket_agent.py      # Support ticket creator + escalation
│   │
│   ├── graph/
│   │   ├── state.py             # ConversationState TypedDict
│   │   ├── router.py            # Intent router (rules + LLM)
│   │   └── workflow.py          # LangGraph StateGraph definition
│   │
│   ├── rag/
│   │   └── faq_agent.py         # RAG-grounded FAQ responder
│   │
│   ├── mcp_server/
│   │   ├── server.py            # FastMCP server exposing RAG tool
│   │   ├── rag_tool.py          # ChromaDB search tool
│   │   └── vectorstore.py       # Document ingestion + embedding
│   │
│   └── auth/
│       └── auth_routes.py       # /auth/signup and /auth/login endpoints
│
├── frontend/
│   ├── streamlit_app.py         # Chatbot UI + Evaluation + Observability dashboard
│   └── evaluation_matrix.py    # Evaluation metric functions (accuracy, rating, success)
│
├── rag_data_store/              # Persisted ChromaDB vector store
│   └── chroma.sqlite3
│
├── NovaCart_Ecommerce.pdf       # Source document for RAG (company FAQ)
├── orders.db                    # SQLite — orders, users, returns, tickets, turn_metrics
├── memory.db                    # SQLite — conversation_memory, session_summaries
├── .env                         # API keys (see Environment Variables)
└── README.md
```

---

## Features

### Conversational Agents

| Agent | What it does |
| **order_agent** | Two-turn flow: extracts product + quantity → shows order card → collects payment method → commits to DB |
| **track_agent** | Looks up order by ID or resolves last order, computes ETA — no LLM used |
| **return_agent** | Extracts return reason from conversation context, inserts return record |
| **ticket_agent** | Creates support tickets, handles escalation to human support |
| **faq_agent** | Retrieves context from ChromaDB via MCP, generates grounded answers using Groq |

### Intent Router

The router uses **rule-based pre-checks first**, falling back to LLM classification only when needed:

- **Compiled regex patterns** catch order intents: `"order 2 laptops"`, `"buy a headphone"`, `"place an order for..."`
- **Keyword sets** catch catalog requests: `"show me products"`, `"what do you sell"`
- **Multi-turn state shortcuts**: pending payment flow → straight to `order_agent`, bypassing LLM
- **Conversational shortcuts**: goodbye, filler messages, name queries — answered without any LLM call
- **LLM fallback** (Groq LLaMA 3.1-8b) for ambiguous intents: track, return, ticket, faq

### Product Catalog

- **42 products** across 5 categories: Electronics, Kitchen & Appliances, Clothing & Fashion, Books & Stationery, Food & Grocery
- **4-layer fuzzy search**: exact match → starts-with → full-phrase LIKE → noun-first keyword LIKE → difflib (cutoff 0.52)
- **Plural handling**: `"pens"` → searches `"pen"`, `"watches"` → searches `"watch"`
- **Payment resolver**: 18 aliases mapped to 4 canonical methods — COD, UPI, Credit/Debit Card, Net Banking

### Two-Turn Payment Flow

```
User: "order 2 laptops"
Bot:  [shows product card with price, ETA, payment options]   ← STATE B

User: "UPI"
Bot:  [confirms order, inserts to DB, returns Order ID]        ← STATE A
```

Pending order is preserved in server-side memory across turns, not in the LLM context.

### Memory System

Two-layer architecture inspired by AWS Bedrock memory:

- **Short-term**: sliding window of last 10 messages (5 turns), per session, stored in `memory.db`
- **Long-term**: LLM-generated structured summary of each past session, loaded at session start and injected into agent context — enables sentiment continuity, recognising returning frustrated users, and topic recall across sessions

### Guardrails

**Input guardrails** (run before the graph):
- Prompt injection detector — 10+ patterns including `"ignore all previous instructions"`, `"act as a different AI"`, `"jailbreak"`, `"forget everything"`
- PII detector — credit/debit card numbers, Aadhaar, OTPs, CVVs
- Hard off-topic block — explicit abuse patterns

**Output guardrails** (run after the graph):
- Malformed order ID redaction — `ORD-FAKE-XYZ` → `[ORDER-ID-ERROR]`
- Response length cap — truncates responses over 600 words
- FAQ hallucination score — keyword overlap between response and RAG context; low-grounding responses get a disclaimer appended

### Observability

- **LangSmith** (optional): auto-traces every LangGraph node — router, agent, LLM call — with full prompt, response, and token counts. Enabled by setting `LANGCHAIN_API_KEY` in `.env`
- **Custom SQLite metrics**: one row per turn in `turn_metrics` table — intent, agent, latency (ms), estimated tokens, guardrail events, order committed flag, FAQ grounding score
- **REST endpoints**: `GET /metrics` and `GET /metrics/recent` power the Streamlit dashboard
- **Streamlit Observability page**: KPI cards (avg latency, p95, token usage), business metrics (order completion %, FAQ fallback rate), guardrail event breakdown, colour-coded live turn log (red = blocked, orange = output modified)

### Authentication

- `/auth/signup` — creates a new user with hashed password in the `users` table
- `/auth/login` — returns `user_id` for session management
- Streamlit frontend handles login/signup flow before entering the chatbot

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq — `llama-3.1-8b-instant` |
| Agent framework | LangGraph (StateGraph) |
| LLM client | LangChain Groq (`langchain-groq`) |
| RAG vector store | ChromaDB (`langchain-chroma`) |
| RAG embeddings | HuggingFace `all-MiniLM-L6-v2` (`langchain-huggingface`) |
| Document loader | LangChain PDF loader (`langchain-community`) |
| Text splitter | RecursiveCharacterTextSplitter (`langchain-text-splitters`) |
| MCP server | FastMCP |
| Backend API | FastAPI + Uvicorn |
| Database | SQLite (orders.db + memory.db) |
| Frontend | Streamlit |
| Tracing | LangSmith (optional) |
| Environment | python-dotenv |
| Data | Pandas |

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com) (free tier available)
- Optional: [LangSmith API key](https://smith.langchain.com) for tracing

### 1. Clone and navigate

```bash
git clone <repo-url>
cd novacart
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create your `.env` file

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys (see [Environment Variables](#environment-variables)).

### 5. Ingest documents into ChromaDB (first time only)

```bash
python -m backend.mcp_server.vectorstore
```

This reads `NovaCart_Ecommerce.pdf`, splits it, embeds it with HuggingFace, and saves the vector store to `rag_data_store/`.

---

## Running the Project

You need **three terminals** running simultaneously.

### Terminal 1 — MCP Server (RAG)

```bash
python -m backend.mcp_server.server
```

Starts the FastMCP server on `http://127.0.0.1:8000/mcp`. The FAQ agent calls this to retrieve company document context.

### Terminal 2 — FastAPI Backend

```bash
uvicorn backend.main:app --reload --port 8001
```

On startup you will see:

```
[Observability] LangSmith tracing enabled → project: novacart-ai
```
or
```
[Observability] LangSmith not configured — set LANGCHAIN_API_KEY in .env to enable tracing.
```

API docs available at `http://localhost:8001/docs`

### Terminal 3 — Streamlit Frontend

```bash
streamlit run frontend/streamlit_app.py
```

Opens at `http://localhost:8501`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/auth/signup` | Register new user |
| `POST` | `/auth/login` | Login, returns user_id |
| `POST` | `/chat/session/start` | Start a new chat session, load long-term memory |
| `POST` | `/chat` | Send a message, get AI response |
| `GET` | `/metrics` | Aggregated observability metrics |
| `GET` | `/metrics/recent?n=25` | Last N turn logs for dashboard |

### Chat example

```bash
# Start session
curl -X POST "http://localhost:8001/chat/session/start?user_id=user_abc123"

# Send message
curl -X POST "http://localhost:8001/chat?user_id=user_abc123&session_id=sess_xyz" \
  -H "Content-Type: application/json" \
  -d '{"message": "order 2 laptops"}'
```

---

## Agents & Routing

### Intent classification order

The router checks these in sequence, stopping at the first match:

1. Cancel during pending order → clear pending, return polite response
2. View products during pending order → clear pending, show catalog
3. Pending order present → skip LLM, go straight to `order_agent`
4. View products keyword match → return catalog inline
5. Place order regex match (with compound intent guard) → `order_agent`
6. Name query → `"You're BhumiRaj!"` (no LLM)
7. Goodbye words → farewell message (no LLM)
8. Filler words → help menu (no LLM)
9. **LLM classification** → track / return / ticket / faq / escalate

### Compound intent guard

If a message matches both place-order and view-products patterns (e.g. `"I want to order earbuds, can you show me products first"`), the catalog is shown. The order intent is not triggered prematurely.

---

## Memory System

### Short-term memory

- Stored in `memory.db` → `conversation_memory` table
- Sliding window of **last 10 messages** per `session_id`
- Every agent retrieves this via `get_short_term_context(session_id)` before calling the LLM
- Enables follow-up understanding: `"what about its return policy?"` after a delivery question

### Long-term memory

- Stored in `memory.db` → `session_summaries` table
- At **session end**, LLM generates a compact structured summary (issue, resolution, sentiment, next-steps)
- At **session start**, last 5 summaries are loaded for the user and injected into state as `user_context`
- Agents can reference this to recognise returning users, past issues, or unresolved complaints

---

## Product Catalog

42 products seeded at startup (idempotent — `INSERT OR IGNORE`):

| Category | Example products |
|---|---|
| Electronics | Wireless Bluetooth Headphones, Laptop (15.6"), Smart LED TV, Wireless Earbuds |
| Kitchen & Appliances | Pressure Cooker, Air Fryer, Electric Kettle, Mixer Grinder |
| Clothing & Fashion | Men's Formal Shirt, Women's Kurti, Sports Sneakers |
| Books & Stationery | A4 Notebook, Ball Point Pens, Atomic Habits, Sticky Notes |
| Food & Grocery | Basmati Rice, Coconut Oil, Organic Green Tea, Pizza (Margherita) |

### Search algorithm (4 layers + difflib)

```
1. Exact match:          "Wireless Bluetooth Headphones" → ELEC-001
2. Starts-with:          "Wireless B..." → ELEC-001
3. Full-phrase LIKE:     "wireless earbuds" → ELEC-011 (not ELEC-001)
4. Noun-first keywords:  tries last word first (noun), then adjectives
   Plural stripping:     "pens" → tries "pen", "watches" → "watch"
5. difflib (cutoff 0.52): "earbudss" → ELEC-011, "hedphones" → ELEC-001
```

---

## Guardrails

All guardrail logic lives in `backend/guardrails.py` and returns a `GuardrailResult` dataclass.

### Input guardrails (`check_input`)

Called **before** the LangGraph workflow. If `result.blocked == True`, the graph never runs.

| Check | Trigger | Example |
|---|---|---|
| Prompt injection | Regex patterns | `"ignore all previous instructions"` |
| PII — card number | 13–19 digit sequence | `"my card is 4111 1111 1111 1111"` |
| PII — Aadhaar | 12-digit spaced format | `"my Aadhaar is 1234 5678 9012"` |
| PII — OTP | `"OTP is XXXXXX"` pattern | `"my OTP is 847291"` |
| PII — CVV | `"cvv is XXX"` pattern | `"cvv is 456"` |
| Hard abuse | Explicit patterns | (blocked silently) |

### Output guardrails (`check_output`)

Called **after** the graph returns. Does not block — modifies or appends to the response.

| Check | Trigger | Action |
|---|---|---|
| Malformed order ID | `ORD-` followed by non-6-char suffix | Replace with `[ORDER-ID-ERROR]` |
| Price sanity | `₹0` or `₹999,999+` in response | Warning logged |
| Length cap | Response > 600 words | Truncate + append support email note |
| FAQ grounding | Response keyword overlap < 25% with RAG context (only for >30 word responses) | Append `"I may not have complete information"` disclaimer |

---

## Observability

### LangSmith (Layer 1 — distributed tracing)

Set in `.env`:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_your_key_here
LANGCHAIN_PROJECT=novacart-ai
```

What you get per user message:
- Full trace tree: LangGraph → router → agent → LLM call
- Exact prompt sent to Groq (system, history, user question, RAG context)
- Token counts: input / output / total
- Latency per node

Blocked turns (guardrail fires) **do not** appear in LangSmith — the graph never runs for them.

### Custom SQLite metrics (Layer 2 — business KPIs)

Table: `turn_metrics` in `orders.db`

| Column | Description |
|---|---|
| `session_id` | Session identifier |
| `user_id` / `username` | Who sent the message |
| `user_message` | First 500 chars of message |
| `intent` | Classified intent label |
| `agent_used` | Which agent handled the turn |
| `latency_ms` | End-to-end turn latency |
| `estimated_tokens` | `len(message) / 4` estimate |
| `guardrail_events` | JSON list of event codes fired |
| `input_blocked` | 1 if input guardrail blocked turn |
| `output_modified` | 1 if output guardrail changed response |
| `order_committed` | 1 if an order was inserted to DB |
| `faq_grounding_score` | 0.0–1.0 grounding score (-1 if not FAQ) |

### Dashboard metrics

Accessible in Streamlit → **🔭 Observability** sidebar tab:

- KPI row: total turns, sessions, avg latency, p95 latency, estimated tokens
- Business row: orders placed, order completion rate, FAQ fallback rate, avg grounding score
- Guardrail breakdown: input blocked count, output modified count, event type frequency
- Routing charts: intent distribution, agent distribution
- Live turn log: last 25 turns with colour coding (red = blocked, orange = modified)

---

## Evaluation Dashboard

Streamlit → **📊 Evaluation Metrics** sidebar tab.

After each turn, the frontend logs: user query, bot response, computed confidence score.
The evaluator can then manually label each interaction:
- Was the intent correctly identified? (Yes / No)
- Response appropriateness rating (1–5 slider)
- Was the task completed successfully? (Yes / No)

Aggregated metrics:
- **Intent Accuracy** — % correctly identified intents
- **Average Response Rating** — mean of 1–5 ratings
- **Task Success Rate** — % of successfully completed tasks
- Confidence vs Intent Accuracy line chart
- CSV download of all evaluation logs

---

## Database Schema

### `orders.db`

```sql
users (user_id, username, password)

orders (
    order_id, user_id, product_id, product_name,
    quantity, price_per_unit, total_price,
    payment_mode, status, created_at
)

returns (
    return_id, order_id, user_id,
    reason, status, created_at
)

support_tickets (
    ticket_id, user_id, issue,
    status, created_at
)

products (
    product_id, name, category,
    price, stock, description
)

turn_metrics (
    id, session_id, user_id, username,
    user_message, intent, agent_used,
    latency_ms, estimated_tokens,
    guardrail_events, input_blocked,
    output_modified, order_committed,
    faq_grounding_score, turn_timestamp
)
```

### `memory.db`

```sql
conversation_memory (
    id, session_id, role, content, timestamp
)

session_summaries (
    id, user_id, session_id,
    summary, created_at
)
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
# Required
GROQ_API_KEY=gsk_your_groq_api_key_here

# Optional — enables LangSmith distributed tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_your_langsmith_key_here
LANGCHAIN_PROJECT=novacart-ai
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

Get your Groq API key at [console.groq.com](https://console.groq.com) (free tier, no credit card).  
Get your LangSmith API key at [smith.langchain.com](https://smith.langchain.com) (free tier).

---

## Common Issues

**"MCP connection refused"** — Make sure the MCP server (Terminal 1) is running before starting the FastAPI backend.

**"ChromaDB collection not found"** — Run the vectorstore ingestion script: `python -m backend.mcp_server.vectorstore`

**"LangSmith not configured"** — This is just an info message. The chatbot works fully without LangSmith. Add `LANGCHAIN_API_KEY` to `.env` to enable tracing.

**Products not found** — The fuzzy search has a difflib cutoff of 0.52. Very short queries like `"tv"` may not match `"Smart LED TV"` — try `"smart tv"` or `"LED TV"`.

**Streamlit login fails** — Make sure the FastAPI backend is running on port 8001 before opening the Streamlit app.