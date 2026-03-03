import asyncio
import logging

import streamlit as st

from agent import get_life_coach_agent, get_run_config, get_runner, get_session

logger = logging.getLogger(__name__)

# UI Config
st.set_page_config(page_title="Life Coach", page_icon="🌱", layout="wide" )
st.title("🌱 Life Coach")
st.markdown("---")

# Initialize session state — cache agent, runner and session for reuse across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = get_life_coach_agent()
if "runner" not in st.session_state:
    st.session_state.runner = get_runner()
if "session" not in st.session_state:
    st.session_state.session = get_session(session_id="user_session_1")

# ── Sidebar: Memory Viewer ──────────────────────────────────────────────────
with st.sidebar:
    st.header("🧠 Session Memory")
    st.caption("All messages stored in the SQLite session (as seen by the agent).")

    # Fetch current memory from SQLiteSession (get_items is async)
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

    # Clear memory button
    if st.button("🗑️ Clear Memory", type="primary", use_container_width=True):
        try:
            asyncio.run(st.session_state.session.clear_session())
            st.session_state.messages = []
            st.success("Memory cleared!")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to clear memory: {e}")

# ── Chat UI ─────────────────────────────────────────────────────────────────

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# ── Streaming generator ──────────────────────────────────────────────────────
async def stream_agent(user_input: str, tool_placeholder: st.delta_generator.DeltaGenerator):
    """
    Async generator that drives the SDK event stream and yields token deltas.
    Passed directly to st.write_stream() — Streamlit handles async-to-sync conversion.
    """
    streamed_result = st.session_state.runner.run_streamed(
        st.session_state.agent,
        input=user_input,
        session=st.session_state.session,
        max_turns=10,
        run_config=get_run_config(),
    )

    got_tokens = False
    fallback_text = ""

    async for event in streamed_result.stream_events():
        # Token-level streaming
        if event.type == "raw_response_event":
            try:
                delta = getattr(event.data, "delta", None)
                if delta:
                    if not got_tokens:
                        tool_placeholder.empty()  # clear any "Searching..." indicator
                    got_tokens = True
                    yield delta
            except Exception as e:
                logger.warning("Token delta error: %s", e)

        # Run item events (tool calls, complete message output)
        elif event.type == "run_item_stream_event":
            item_type = getattr(event.item, "type", None)

            if item_type == "tool_call_item":
                # Show a live indicator while the web search tool runs
                desc = getattr(event.item, "description", None) or "web"
                tool_placeholder.info(f"🔍 Searching the {desc}...")

            elif item_type == "message_output_item":
                # Fallback: use completed message text if no token deltas arrived
                if not got_tokens:
                    try:
                        raw_item = getattr(event.item, "raw_item", None)
                        if raw_item and hasattr(raw_item, "content"):
                            parts = [c.text for c in raw_item.content if hasattr(c, "text")]
                            if parts:
                                fallback_text = "".join(parts)
                    except Exception as e:
                        logger.warning("Message output item error: %s", e)

    # Clear any leftover status indicator
    tool_placeholder.empty()

    # Emit fallback text as a single yield if we never got token deltas
    if not got_tokens and fallback_text:
        yield fallback_text


# ── Chat input ───────────────────────────────────────────────────────────────
if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        tool_placeholder = st.empty()
        full_response = st.write_stream(stream_agent(prompt, tool_placeholder))

    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": str(full_response)})
        st.rerun()  # re-execute the script so the sidebar re-fetches and displays the latest memory
