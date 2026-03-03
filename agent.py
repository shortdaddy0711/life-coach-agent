import dotenv

dotenv.load_dotenv()  # must run before agents SDK imports so env vars are set

from agents import Agent, WebSearchTool, SQLiteSession, RunConfig

# System prompt for the Life Coach
LIFE_COACH_PROMPT = """
You are a supportive, encouraging, and wise Life Coach.
Your goal is to help users improve their lives, build better habits, and stay motivated.
When giving advice, be empathetic but also practical and actionable:
1. ALWAYS search the web first for the latest research and studies.
2. Then synthesize the search results with your coaching expertise.
3. Cite specific findings to make your advice more credible.
4. Always maintain a positive and professional tone.
"""

def get_life_coach_agent():
    return Agent(
        name="Life Coach",
        instructions=LIFE_COACH_PROMPT,
        model="gpt-4o-mini",
        tools=[WebSearchTool()],
    )

def get_session(db_path="life_coach.db", session_id="default"):
    return SQLiteSession(session_id=session_id, db_path=db_path)

def get_run_config():
    """Return a RunConfig with sensible production defaults."""
    return RunConfig(trace_include_sensitive_data=False)
