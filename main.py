import asyncio
import logging
import os

import streamlit as st
from agents import Runner
from openai import OpenAI

from agent import get_life_coach_agent, get_run_config, get_session

logger = logging.getLogger(__name__)

# ── UI Config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Life Coach", page_icon="🌱", layout="wide")
st.title("🌱 Life Coach")
st.markdown("---")

# ── Session state — agent and session only (no Runner instance, no messages list) ──
if "agent" not in st.session_state:
    st.session_state.agent = get_life_coach_agent()
if "session" not in st.session_state:
    st.session_state.session = get_session(session_id="user_session_1")

client = OpenAI()
VECTOR_STORE_ID = os.environ["VECTOR_STORE_ID"]

# ── Sidebar: Memory Viewer ──────────────────────────────────────────────────
with st.sidebar:
    st.header("🧠 Session Memory")
    st.caption("All messages stored in the SQLite session (as seen by the agent).")

    try:
        memory_items = asyncio.run(st.session_state.session.get_items())
    except Exception as e:
        memory_items = []
        st.warning(f"Could not load memory: {e}")

    if memory_items:
        st.markdown(f"**{len(memory_items)} item(s) stored**")
        st.markdown("---")
        for i, item in enumerate(memory_items, 1):
            role = item.get("role", "unknown") if isinstance(item, dict) else getattr(item, "role", "unknown")
            role_icon = "🧑" if role == "user" else "🤖" if role == "assistant" else "⚙️"
            with st.expander(f"{role_icon} [{i}] {role.capitalize()}", expanded=False):
                st.json(item if isinstance(item, dict) else vars(item) if hasattr(item, "__dict__") else str(item))
    else:
        st.info("No memory yet. Start a conversation!")

    st.markdown("---")

    if st.button("🗑️ Clear Memory", type="primary", use_container_width=True):
        try:
            asyncio.run(st.session_state.session.clear_session())
            st.success("Memory cleared!")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to clear memory: {e}")

# ── Status labels for raw OpenAI API event types ────────────────────────────
STATUS_MESSAGES = {
    "response.web_search_call.in_progress":  ("🔍 Starting web search...", "running"),
    "response.web_search_call.searching":    ("🔍 Searching the web...", "running"),
    "response.web_search_call.completed":    ("✅ Web search complete", "complete"),
    "response.file_search_call.in_progress": ("🗂️ Starting file search...", "running"),
    "response.file_search_call.searching":   ("🗂️ File search in progress...", "running"),
    "response.file_search_call.completed":   ("✅ File search complete", "complete"),
    "response.completed":                    (" ", "complete"),
}

def update_status(status_container, event_type: str):
    if event_type in STATUS_MESSAGES:
        label, state = STATUS_MESSAGES[event_type]
        status_container.update(label=label, state=state)

# ── Chat history (source of truth: SQLiteSession DB) ────────────────────────
async def paint_history():
    """Read all messages from SQLiteSession and render them in the chat."""
    messages = await st.session_state.session.get_items()
    for message in messages:
        if not isinstance(message, dict):
            continue

        # User and assistant messages have a "role" key
        if "role" in message:
            role = message["role"]
            with st.chat_message(role):
                if role == "user":
                    content = message.get("content", "")
                    if isinstance(content, str):
                        st.write(content)
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "input_text":
                                st.write(part.get("text", ""))
                elif role == "assistant":
                    if message.get("type") == "message":
                        content = message.get("content", [])
                        text = " ".join(
                            part.get("text", "") for part in content
                            if isinstance(part, dict) and part.get("type") == "output_text"
                        )
                        st.write(text.replace("$", "\\$"))

        # Tool call items have a "type" key but no "role" key
        elif "type" in message:
            if message["type"] == "web_search_call":
                with st.chat_message("assistant"):
                    st.write("🔍 Searched the web")
            elif message["type"] == "file_search_call":
                with st.chat_message("assistant"):
                    st.write("🗂️ Searched your files")

asyncio.run(paint_history())

# ── Streaming generator ──────────────────────────────────────────────────────
async def stream_agent(user_input: str, status_container):
    streamed_result = Runner.run_streamed(
        st.session_state.agent,
        input=user_input,
        session=st.session_state.session,
        max_turns=10,
        run_config=get_run_config(),
    )

    async for event in streamed_result.stream_events():
        if event.type == "raw_response_event":
            # Update the status container with live phase information
            update_status(status_container, event.data.type)
            # Yield only text output deltas — precise, no false positives
            if event.data.type == "response.output_text.delta":
                yield event.data.delta.replace("$", "\\$")

# ── Chat input ───────────────────────────────────────────────────────────────
if prompt := st.chat_input(
    "How can I help you today?",
    accept_file=True,
    file_type=["txt", "pdf"],
):
    # Handle file uploads first
    for file in prompt.files:
        if file.type.startswith("text/") or file.type == "application/pdf":
            with st.chat_message("assistant"):
                with st.status("⏳ Uploading file...") as status:
                    uploaded_file = client.files.create(
                        file=(file.name, file.getvalue()),
                        purpose="user_data",
                    )
                    status.update(label="⏳ Attaching to vector store...")
                    client.vector_stores.files.create(
                        vector_store_id=VECTOR_STORE_ID,
                        file_id=uploaded_file.id,
                    )
                    status.update(label="✅ File uploaded to your vault", state="complete")

    # Handle text message
    if prompt.text:
        with st.chat_message("user"):
            st.write(prompt.text)
        with st.chat_message("assistant"):
            status_container = st.status("⏳ Thinking...", expanded=False)
            st.write_stream(stream_agent(prompt.text, status_container))
        st.rerun()  # repaint history from DB and refresh sidebar memory count
