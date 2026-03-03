# 🌱 Life Coach Agent

An AI-powered life coach chatbot built with [Streamlit](https://streamlit.io) and the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python). It provides evidence-based, actionable advice by always searching the web for the latest research before responding.

## Features

- 💬 **Streaming chat UI** — word-by-word response rendering powered by `st.write_stream()`
- 🔍 **Live web search** — always searches for the latest studies and research before giving advice
- 🧠 **Persistent memory** — conversations are stored in a local SQLite database and survive page reloads
- 📊 **Memory inspector** — sidebar shows all raw session data for debugging
- 🗑️ **One-click memory reset** — clear the conversation history at any time

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.13+ |
| Package Manager | [uv](https://github.com/astral-sh/uv) |
| Web UI | Streamlit ≥ 1.30.0 |
| Agent Framework | OpenAI Agents SDK (`openai-agents`) |
| LLM | gpt-4o-mini via OpenAI Responses API |
| Memory | SQLite via `SQLiteSession` |

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/shortdaddy0711/life-coach-agent.git
cd life-coach-agent
```

### 2. Set up your API key

Create a `.env` file in the project root:

```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

### 3. Install dependencies

```bash
uv sync
```

### 4. Run the app

```bash
uv run streamlit run main.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Project Structure

```
life-hacker/
├── main.py          # Streamlit UI — chat, history, sidebar, streaming
├── agent.py         # OpenAI Agents SDK setup — Agent, Session, RunConfig
├── pyproject.toml   # Project metadata and dependencies
├── uv.lock          # Locked dependency tree
└── .env             # Your OpenAI API key (never committed)
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | OpenAI API key for the Agents SDK and web search tool |
