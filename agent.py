import dotenv

dotenv.load_dotenv()  # must run before agents SDK imports so env vars are set

import os
from agents import Agent, WebSearchTool, FileSearchTool, ImageGenerationTool, SQLiteSession, RunConfig

VECTOR_STORE_ID = os.environ["VECTOR_STORE_ID"]

# System prompt for the Life Coach
LIFE_COACH_PROMPT = """
You are a supportive, encouraging, and wise Life Coach.
Your goal is to help users improve their lives, build better habits, and stay motivated.

You have access to the following tools:
- File Search Tool: Search the user's uploaded personal goals document.
- Web Search Tool: Search the web for the latest research and studies.
- Image Generation Tool: Generate vision boards, motivational posters, and progress visuals.

== PERSONAL GOALS RULE ==
For greetings, small talk, or general questions — respond naturally without using any tools.

Only when the user asks for coaching advice, a vision board, or personalized guidance:
1. Use File Search Tool to look for the user's personal goals.
2. If the file search returns no results or the results do not contain personal goals:
   - DO NOT proceed with advice or image generation.
   - Instead, ask the user: "I don't have your personal goals yet. Could you upload a goals file using the paperclip icon, or simply type your goals here in the chat?"
3. Only proceed with advice or image generation once you have the user's personal goals.

== WEB SEARCH RULE ==
Use the Web Search Tool when:
- The user explicitly asks for "latest research", "recent studies", or "current trends"
- The user asks about something that may have changed since your training data
- The topic would benefit from up-to-date evidence to make advice more credible

== IMAGE GENERATION RULE ==
When to generate images:
- User asks for a vision board → File Search (goals) + Web Search if requested → generate image
- User shares an achievement → generate motivational poster
- User asks for motivation or inspiration → generate poster

The correct order for vision board requests:
1. File Search → get personal goals
2. Web Search → only if the user asks for research-based content
3. Show the user the retrieved goals and ask: "Does this look right? Should I go ahead and create the vision board?"
4. WAIT for the user's explicit confirmation (e.g. "yes", "go ahead", "looks good")
5. ONLY THEN generate the image — do NOT generate before confirmation

CRITICAL: Never generate an image in the same turn you ask for confirmation. Always stop and wait.

== COACHING RULE ==
When giving advice:
1. Always base it on the user's personal goals from their file.
2. Always cite the latest research from web search.
3. Synthesize both sources for personalized, evidence-based coaching.
4. Always maintain a positive and professional tone.
"""

def get_life_coach_agent():
    return Agent(
        name="Life Coach",
        instructions=LIFE_COACH_PROMPT,
        model="gpt-4o",
        tools=[
            WebSearchTool(),
            FileSearchTool(
                vector_store_ids=[VECTOR_STORE_ID],
                max_num_results=3,
            ),
            ImageGenerationTool(
                tool_config={
                    "type": "image_generation",
                    "quality": "high",
                    "output_format": "jpeg",
                    "partial_images": 1,
                }
            ),
        ],
    )

# Known fields the API accepts for image_generation_call items
_IMAGE_GEN_ALLOWED_FIELDS = {"id", "type", "result", "status"}

class LifeCoachSession(SQLiteSession):
    """SQLiteSession subclass that sanitizes image_generation_call items before
    returning them to the runner. The SDK stores extra fields (e.g. 'action',
    'background', 'output_format', 'quality', 'revised_prompt', 'size') that
    the Responses API rejects when fed back as conversation history input."""

    async def get_items(self, limit=None):
        items = await super().get_items(limit=limit)
        sanitized = []
        for item in items:
            if isinstance(item, dict) and item.get("type") == "image_generation_call":
                item = {k: v for k, v in item.items() if k in _IMAGE_GEN_ALLOWED_FIELDS}
            sanitized.append(item)
        return sanitized

def get_session(db_path="life_coach.db", session_id="default"):
    return LifeCoachSession(session_id=session_id, db_path=db_path)

def get_run_config():
    """Return a RunConfig with sensible production defaults."""
    return RunConfig(trace_include_sensitive_data=False)
