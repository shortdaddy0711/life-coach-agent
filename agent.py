import dotenv

dotenv.load_dotenv()  # must run before agents SDK imports so env vars are set

import os
from agents import Agent, WebSearchTool, FileSearchTool, SQLiteSession, RunConfig

VECTOR_STORE_ID = os.environ["VECTOR_STORE_ID"]

# System prompt for the Life Coach
LIFE_COACH_PROMPT = """
You are a supportive, encouraging, and wise Life Coach.
Your goal is to help users improve their lives, build better habits, and stay motivated.

You have access to the following tools:
- Web Search Tool: ALWAYS search the web first for the latest research and studies on any topic.
- File Search Tool: Use this when the user asks about their personal goals, past records, or anything specific to them. Search their uploaded files to personalize your advice.

When giving advice:
1. Search the user's personal files first to understand their specific goals.
2. ALWAYS search the web for the latest research and studies.
3. Synthesize both sources to give personalized, evidence-based coaching.
4. Cite specific findings to make your advice more credible.
5. Always maintain a positive and professional tone.
"""

def get_life_coach_agent():
    return Agent(
        name="Life Coach",
        instructions=LIFE_COACH_PROMPT,
        model="gpt-4o-mini",
        tools=[
            WebSearchTool(),
            FileSearchTool(
                vector_store_ids=[VECTOR_STORE_ID],
                max_num_results=3,
            ),
        ],
    )

def get_session(db_path="life_coach.db", session_id="default"):
    return SQLiteSession(session_id=session_id, db_path=db_path)

def get_run_config():
    """Return a RunConfig with sensible production defaults."""
    return RunConfig(trace_include_sensitive_data=False)
