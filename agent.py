from dotenv import load_dotenv
from agents import Agent, Runner, WebSearchTool, SQLiteSession, RunConfig

load_dotenv()

# System prompt for the Life Coach
LIFE_COACH_PROMPT = """
You are a supportive, encouraging, and wise Life Coach.
Your goal is to help users improve their lives, build better habits, and stay motivated.
When giving advice, be empathetic but also practical.
Use the web search tool to find the latest research, tips, and techniques for habit formation and self-improvement if needed.
Always maintain a positive and professional tone.
"""

def get_life_coach_agent():
    # Initialize the hosted WebSearchTool
    search_tool = WebSearchTool()

    # Create the Life Coach Agent with an explicit model
    agent = Agent(
        name="Life Coach",
        instructions=LIFE_COACH_PROMPT,
        model="gpt-4o",
        tools=[search_tool]
    )

    return agent

def get_runner():
    return Runner()

def get_session(db_path="life_coach.db", session_id="default"):
    return SQLiteSession(session_id=session_id, db_path=db_path)

def get_run_config():
    """Return a RunConfig with sensible production defaults."""
    return RunConfig(
        trace_include_sensitive_data=False,
    )
