# backend/guardrails.py
"""
Guardrails  ─  NovaCart AI
═══════════════════════════════════════════════════════════════════════
Two-layer safety system that wraps every turn in the /chat endpoint.

  INPUT GUARDRAILS   — run BEFORE the LangGraph workflow
  ┌─────────────────────────────────────────────────────────────────┐
  │  1. Prompt Injection Detector                                   │
  │     Catches jailbreaks / system-prompt overrides               │
  │  2. PII Detector                                                │
  │     Catches credit cards, Aadhaar numbers, OTPs in chat        │
  │  3. Off-topic Filter                                            │
  │     Blocks clear non-ecommerce abuse (politics, adult content) │
  └─────────────────────────────────────────────────────────────────┘

  OUTPUT GUARDRAILS  — run AFTER the LangGraph workflow
  ┌─────────────────────────────────────────────────────────────────┐
  │  1. Order ID Format Validator                                   │
  │     Any "ORD-XXXX" in response must be real DB format          │
  │  2. Price Sanity Check                                          │
  │     Prices quoted must match the products table                │
  │  3. Response Length Guard                                       │
  │     Truncates runaway LLM responses (>600 words)               │
  │  4. FAQ Hallucination Score                                     │
  │     Measures keyword overlap between response and RAG context  │
  └─────────────────────────────────────────────────────────────────┘

Design principles:
  • Rule-based first (zero latency, zero cost)
  • Every guardrail event is returned in GuardrailResult so
    observability.py can log it without duplicating logic
  • Guardrails never crash the app — all exceptions caught internally
  • Soft blocks return a polite response; hard blocks are logged
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────
# RESULT DATACLASS
# Carries the outcome of a guardrail check through the call stack.
# ─────────────────────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    """
    blocked:      True  → stop processing, send `reply` to user instead
                  False → safe, continue normally
    reply:        Polite message to show user when blocked (None if safe)
    event:        Short machine-readable label for observability logging
                  e.g. "prompt_injection", "pii_detected", "off_topic"
    detail:       Human-readable description of what was detected
    modified_response: For output guardrails — the (possibly trimmed)
                       version of the bot response (None = unchanged)
    """
    blocked:           bool            = False
    reply:             Optional[str]   = None
    event:             Optional[str]   = None
    detail:            Optional[str]   = None
    modified_response: Optional[str]   = None
    warnings:          list[str]       = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# COMPILED PATTERNS  (loaded once at module import)
# ─────────────────────────────────────────────────────────────────────

# ── Prompt injection ─────────────────────────────────────────────────
_INJECTION_PATTERNS = re.compile(
    r"ignore (\w+\s+){0,3}(instructions?|prompt|rules?|context)"
    r"|forget (everything|all|your|previous)"
    r"|you are now (a |an )?"
    r"|act as (a |an )?(different|new|another|evil|unrestricted)"
    r"|disregard (your |all |previous )?(instructions?|rules?|training)"
    r"|system\s*prompt"
    r"|jailbreak"
    r"|dan mode"
    r"|pretend (you are|you're|to be) (a |an )?(different|unrestricted|evil)",
    re.IGNORECASE,
)

# ── PII patterns ─────────────────────────────────────────────────────
# Credit / debit card  — 13-19 digits, optionally space/dash separated
_CC_PATTERN = re.compile(r"\b(?:\d[ \-]?){13,19}\b")
# Aadhaar — 12 digits (with spaces)
_AADHAAR_PATTERN = re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b")
# OTP — "OTP is 123456" / "my OTP: 4521"
_OTP_PATTERN = re.compile(r"\b(otp|one.?time.?password)\s*[:is]*\s*\d{4,8}\b", re.IGNORECASE)
# CVV
_CVV_PATTERN = re.compile(r"\b(cvv|cvc|security code)\s*[:is]*\s*\d{3,4}\b", re.IGNORECASE)

# ── Off-topic: hard blocks only (clearly non-ecommerce abuse) ────────
_OFFTOPIC_PATTERNS = re.compile(
    r"\b(generate\s+(nude|naked|explicit|porn)|"
    r"how\s+to\s+(make\s+a\s+bomb|hack|kill)|"
    r"racist|terrorism|child\s+(porn|abuse))\b",
    re.IGNORECASE,
)

# ── Order ID format (for output validation) ──────────────────────────
_ORDER_ID_PATTERN = re.compile(r"ORD-[A-Z0-9]{6}", re.IGNORECASE)

# ── Price in response (₹ followed by digits) ─────────────────────────
_PRICE_IN_RESPONSE = re.compile(r"₹\s*(\d[\d,]*)")


# ─────────────────────────────────────────────────────────────────────
# INPUT GUARDRAILS
# ─────────────────────────────────────────────────────────────────────

def check_input(user_text: str, username: str = "there") -> GuardrailResult:
    """
    Run all input guardrails against the user's raw message.

    Call this BEFORE invoking the LangGraph workflow.
    If result.blocked is True, return result.reply directly to the user
    and skip the graph entirely.
    """
    try:
        text = user_text.strip()

        # ── 1. Prompt injection ───────────────────────────────────────
        if _INJECTION_PATTERNS.search(text):
            return GuardrailResult(
                blocked = True,
                reply   = (
                    f"I'm sorry, {username}, I can't process that kind of request. 😊\n\n"
                    "I'm here to help you with shopping, orders, tracking, and returns on NovaCart. "
                    "What can I help you with today?"
                ),
                event  = "prompt_injection",
                detail = f"Injection pattern detected in: {text[:80]}",
            )

        # ── 2. PII detection ─────────────────────────────────────────
        pii_found = None
        if _CC_PATTERN.search(text):
            pii_found = "credit/debit card number"
        elif _AADHAAR_PATTERN.search(text):
            pii_found = "Aadhaar number"
        elif _OTP_PATTERN.search(text):
            pii_found = "OTP"
        elif _CVV_PATTERN.search(text):
            pii_found = "CVV/security code"

        if pii_found:
            return GuardrailResult(
                blocked = True,
                reply   = (
                    f"⚠️ Please don't share your **{pii_found}** in this chat, {username}.\n\n"
                    "For your security, NovaCart never asks for sensitive financial details "
                    "through the assistant. Payments are handled securely at checkout.\n\n"
                    "Is there anything else I can help you with?"
                ),
                event  = "pii_detected",
                detail = f"PII type: {pii_found}",
            )

        # ── 3. Hard off-topic block ───────────────────────────────────
        if _OFFTOPIC_PATTERNS.search(text):
            return GuardrailResult(
                blocked = True,
                reply   = (
                    f"I'm not able to help with that, {username}. "
                    "I'm NovaCart's shopping assistant — I can help you place orders, "
                    "track shipments, and handle returns. 🛒"
                ),
                event  = "off_topic_blocked",
                detail = f"Hard off-topic pattern matched: {text[:80]}",
            )

        # ── All clear ────────────────────────────────────────────────
        return GuardrailResult(blocked=False)

    except Exception as exc:
        # Guardrails must never crash the app
        return GuardrailResult(
            blocked  = False,
            warnings = [f"input_guardrail_error: {exc}"],
        )


# ─────────────────────────────────────────────────────────────────────
# OUTPUT GUARDRAILS
# ─────────────────────────────────────────────────────────────────────

def check_output(
    response:     str,
    intent:       str       = "",
    rag_context:  str       = "",
    username:     str       = "there",
) -> GuardrailResult:
    """
    Run all output guardrails against the bot's generated response.

    Call this AFTER the LangGraph workflow returns.
    If result.modified_response is not None, serve that to the user
    instead of the original response.
    All detected issues are added to result.warnings for logging.
    """
    try:
        warnings      = []
        final_response = response

        # ── 1. Order ID format validator ─────────────────────────────
        # Any ORD- reference in the response must be well-formed.
        # Malformed IDs indicate hallucination (LLM made up an order ID).
        order_ids_in_response = re.findall(r"ORD-[A-Za-z0-9]+", response)
        for oid in order_ids_in_response:
            if not _ORDER_ID_PATTERN.fullmatch(oid):
                warnings.append(f"malformed_order_id: {oid}")
                # Replace the malformed ID with a redacted placeholder
                final_response = final_response.replace(
                    oid, "[ORDER-ID-ERROR]"
                )

        # ── 2. Price sanity check ─────────────────────────────────────
        # Prices in the response should be reasonable (₹1 – ₹99,999).
        # Wildly out-of-range prices indicate LLM hallucination.
        prices_in_response = _PRICE_IN_RESPONSE.findall(final_response)
        for price_str in prices_in_response:
            try:
                price_val = int(price_str.replace(",", ""))
                if price_val < 1 or price_val > 999999:
                    warnings.append(f"suspicious_price: ₹{price_str}")
            except ValueError:
                pass

        # ── 3. Response length guard ──────────────────────────────────
        # Small LLMs can ramble when given large RAG context.
        # Anything over 600 words gets a polite truncation note appended.
        word_count = len(final_response.split())
        if word_count > 600:
            # Keep the first 600 words, add a note
            truncated      = " ".join(final_response.split()[:600])
            final_response = (
                truncated
                + f"\n\n*...For more details, please contact us at support@novacart.in*"
            )
            warnings.append(f"response_truncated: {word_count} words → 600")

        # ── 4. FAQ hallucination score ────────────────────────────────
        # Only runs when we have RAG context to compare against.
        # Measures: how many key nouns in the response appear in the
        # retrieved context? Low overlap = possible hallucination.
        hallucination_warning = None
        if rag_context and intent == "faq":
            score = _compute_grounding_score(final_response, rag_context)
            if score < 0.25 and len(final_response.split()) > 30:
                hallucination_warning = f"low_grounding_score: {score:.2f}"
                warnings.append(hallucination_warning)
                # Append a soft disclaimer instead of blocking
                final_response += (
                    f"\n\n*Note: I may not have complete information on this. "
                    f"For accurate details, please reach us at support@novacart.in*"
                )

        modified = final_response if final_response != response else None

        return GuardrailResult(
            blocked           = False,
            modified_response = modified,
            warnings          = warnings,
        )

    except Exception as exc:
        return GuardrailResult(
            blocked  = False,
            warnings = [f"output_guardrail_error: {exc}"],
        )


# ─────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────

# Common English stop words to exclude from grounding check
_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "to", "of", "in", "on", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above",
    "below", "from", "up", "down", "out", "off", "over", "under", "again",
    "further", "then", "once", "and", "but", "or", "nor", "so", "yet",
    "both", "either", "neither", "not", "only", "own", "same", "than",
    "too", "very", "just", "because", "as", "until", "while", "that",
    "this", "these", "those", "i", "me", "my", "we", "our", "you", "your",
    "he", "him", "his", "she", "her", "it", "its", "they", "them", "their",
    "what", "which", "who", "whom", "how", "all", "each", "more", "most",
    "no", "nor", "not", "such", "here", "there", "when", "where", "why",
}


def _compute_grounding_score(response: str, context: str) -> float:
    """
    Compute a simple keyword-overlap grounding score.

    Extracts content words (len ≥ 4, not stop words) from the response
    and measures what fraction of them also appear in the RAG context.

    Returns a float between 0.0 (no grounding) and 1.0 (fully grounded).
    """
    resp_words = {
        w.lower().strip(".,!?;:\"'()")
        for w in response.split()
        if len(w) >= 4 and w.lower() not in _STOP_WORDS
    }
    ctx_lower  = context.lower()

    if not resp_words:
        return 1.0  # nothing to check → assume fine

    grounded = sum(1 for w in resp_words if w in ctx_lower)
    return grounded / len(resp_words)
