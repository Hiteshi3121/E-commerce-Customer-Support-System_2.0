# frontend/streamlit_app.py
import streamlit as st
import requests
import pandas as pd
import random

from evaluation_matrix import (
    intent_accuracy,
    average_response_rating,
    task_success_rate
)

API_BASE = "http://localhost:8001"

# ── Page Config ────────────────────────────────────────────────────
st.set_page_config(page_title="NovaCart AI", layout="centered")
st.title("🛒 NovaCart AI Assistant")

# ── Greeting Templates ─────────────────────────────────────────────
GREETING_MESSAGES = [
    "👋 Hi {username}! How can I assist you today?",
    "😊 Welcome back, {username}! What would you like to do today?",
    "🛒 Hey {username}! Ready to explore NovaCart?",
    "✨ Hello {username}! I'm here to help you with your orders.",
    "🙌 Hi {username}! How can I make your shopping easier today?"
]

# ── Sidebar Navigation ─────────────────────────────────────────────
page = st.sidebar.selectbox(
    "Select Page",
    ["💬 Chatbot", "📊 Evaluation Metrics"],
    key="nav_page"
)

# ── Confidence Score Helper ────────────────────────────────────────
def compute_confidence(bot_response: str) -> float:
    if not bot_response:
        return 0.0
    uncertainty_words = [
        "maybe", "might", "not sure", "cannot", "couldn't",
        "can not", "unable", "sorry", "later", "try again"
    ]
    penalty      = sum(1 for w in uncertainty_words if w in bot_response.lower())
    length_score = min(len(bot_response) / 200, 1.0)
    return round(max(0.0, length_score - (0.1 * penalty)), 2)


# ── Session State Initialization ───────────────────────────────────
# CHANGE 1: Added has_history to defaults
defaults = {
    "user_id":         None,
    "username":        None,
    "session_id":      None,   # kept across resets — needed for summarization
    "session_started": False,
    "greeted":         False,
    "chat_history":    [],
    "evaluation_logs": [],
    "has_history":     False,  # NEW: True if user has past session summaries
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── Health Check ───────────────────────────────────────────────────
try:
    res = requests.get(f"{API_BASE}/health", timeout=7)
    if res.status_code != 200:
        st.error("Backend not healthy ❌")
        st.stop()
except Exception as e:
    st.error(f"Backend not reachable ❌ ({e})")
    st.stop()


# ══════════════════════════════════════════════════════════════════════
# 💬 CHATBOT PAGE
# ══════════════════════════════════════════════════════════════════════
if page == "💬 Chatbot":

    # ── Authentication ─────────────────────────────────────────────
    if not st.session_state.user_id:
        st.subheader("🔐 Login / Signup")

        tab1, tab2 = st.tabs(["Login", "Signup"])

        with tab1:
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")

            if st.button("Login"):
                res = requests.post(
                    f"{API_BASE}/auth/login",
                    json={"username": username, "password": password}
                )
                if res.status_code == 200:
                    data = res.json()
                    st.session_state.user_id  = data["user_id"]
                    st.session_state.username = data.get("username", username)
                    st.success("Login successful ✅")
                    st.success(st.session_state.user_id)
                    st.rerun()
                else:
                    st.error("Login failed ❌")

        with tab2:
            su_username = st.text_input("Username", key="signup_user")
            su_password = st.text_input("Password", type="password", key="signup_pass")

            if st.button("Signup"):
                res = requests.post(
                    f"{API_BASE}/auth/signup",
                    json={"username": su_username, "password": su_password}
                )
                if res.status_code == 200:
                    data = res.json()
                    st.session_state.user_id  = data["user_id"]
                    st.session_state.username = data.get("username", su_username)
                    st.success("Signup successful ✅")
                    st.success(st.session_state.user_id)
                    st.rerun()
                else:
                    st.error("Signup failed ❌")

        st.stop()

    # ── Start Chat Session (ONCE per session) ──────────────────────
    # CHANGE 2: Pass old_session_id so backend summarizes the ending session
    if not st.session_state.session_started:

        params = {"user_id": st.session_state.user_id}

        # If a previous session exists, pass it so the backend can summarize it
        if st.session_state.session_id:
            params["old_session_id"] = st.session_state.session_id

        res = requests.post(
            f"{API_BASE}/chat/session/start",
            params=params
        )

        if res.status_code == 200:
            data = res.json()
            st.session_state.session_id      = data["session_id"]
            st.session_state.session_started = True
            st.session_state.has_history     = data.get("has_history", False)  # NEW
            st.toast("| 🟢 Chat session started |")

    # ── Personalized Greeting (ONCE) ───────────────────────────────
    # CHANGE 3: Shows memory hint if user has past session summaries
    if not st.session_state.greeted:
        username = st.session_state.get("username", "there")
        greeting = random.choice(GREETING_MESSAGES).format(username=username)

        if st.session_state.has_history:
            greeting += (
                "\n\n💡 *I remember our previous conversations! "
                "You can ask me things like \"what's my most ordered product?\" "
                "or \"how many tickets have I raised?\"*"
            )

        st.session_state.chat_history.append(("assistant", greeting))
        st.session_state.greeted = True

    # ── Chat UI ────────────────────────────────────────────────────
    st.divider()
    st.subheader("💬 Chat with NovaCart AI")

    for role, msg in st.session_state.chat_history:
        st.chat_message(role).write(msg)

    user_msg = st.chat_input("Type your message...")

    if user_msg:
        st.session_state.chat_history.append(("user", user_msg))

        res = requests.post(
            f"{API_BASE}/chat",
            params={
                "user_id":    st.session_state.user_id,
                "session_id": st.session_state.session_id
            },
            json={"message": user_msg}
        )

        try:
            data      = res.json()
            bot_reply = data.get("response", "❌ No response field")
        except Exception:
            bot_reply = f"❌ Backend Error:\n{res.text}"

        confidence_score = compute_confidence(bot_reply)

        if confidence_score < 0.2:
            bot_reply += (
                "\n\n⚠️ *I'm not fully sure, I recommend contacting a human support agent.*"
            )

        st.session_state.chat_history.append(("assistant", bot_reply))
        st.session_state.evaluation_logs.append({
            "user_query":       user_msg,
            "bot_response":     bot_reply,
            "confidence_score": confidence_score,
            "intent_correct":   None,
            "response_rating":  None,
            "task_success":     None
        })

        st.rerun()

    # ── Reset Chat ─────────────────────────────────────────────────
    # CHANGE 4: Removed /chat/session/clear call (endpoint no longer exists)
    # session_id is intentionally NOT cleared — it gets passed as old_session_id
    # to the next /chat/session/start call so the backend summarizes this session
    st.divider()
    if st.button("🔄 Start New Chat"):
        st.session_state.session_started = False
        st.session_state.chat_history    = []
        st.session_state.greeted         = False
        st.session_state.has_history     = False
        # session_id deliberately kept — used for summarization on next session start
        st.rerun()


# ══════════════════════════════════════════════════════════════════════
# 📊 EVALUATION METRICS PAGE
# ══════════════════════════════════════════════════════════════════════
if page == "📊 Evaluation Metrics":

    st.title("📊 Evaluation Metrics Dashboard")

    logs = st.session_state.evaluation_logs

    if not logs:
        st.info("No evaluation data yet.")
        st.stop()

    for i, log in enumerate(logs):
        with st.expander(f"Interaction {i + 1}"):
            st.write("**User Query:**",       log["user_query"])
            st.write("**Bot Response:**",     log["bot_response"])
            st.write("**Confidence Score:**", log.get("confidence_score"))

            log["intent_correct"] = st.selectbox(
                "Was the intent correctly identified?",
                ["Yes", "No"],
                index=0 if log["intent_correct"] is None
                      else ["Yes", "No"].index(log["intent_correct"]),
                key=f"intent_{i}"
            )
            log["response_rating"] = st.slider(
                "Response Appropriateness (1–5)", 1, 5,
                log["response_rating"] or 3,
                key=f"rating_{i}"
            )
            log["task_success"] = st.selectbox(
                "Was the task completed successfully?",
                ["Yes", "No"],
                index=0 if log["task_success"] is None
                      else ["Yes", "No"].index(log["task_success"]),
                key=f"task_{i}"
            )

    st.divider()
    st.subheader("📈 Metrics Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Intent Accuracy",     f"{intent_accuracy(logs) * 100:.1f}%")
    col2.metric("Avg Response Rating", f"{average_response_rating(logs):.2f} / 5")
    col3.metric("Task Success Rate",   f"{task_success_rate(logs) * 100:.1f}%")

    st.divider()
    st.subheader("📉 Confidence vs Intent Accuracy")

    df = pd.DataFrame(logs)
    if "confidence_score" in df and "intent_correct" in df:
        df["intent_binary"] = df["intent_correct"].map({"Yes": 1, "No": 0})
        st.line_chart(df[["confidence_score", "intent_binary"]])

    st.divider()
    st.download_button(
        "⬇️ Download Evaluation CSV",
        df.to_csv(index=False),
        "evaluation_results.csv",
        "text/csv"
    )