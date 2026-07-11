"""
Router Node for WhatsApp Myth Buster Agent.
Decides which retrieval tool to use for the current claim:
  - fact_db: Local Qdrant hybrid search (known myths)
  - web_news: Tavily live news search (recent/political claims)
  - web_science: Tavily scientific search (health/medical claims)
On retry, avoids repeating the same tool used previously.
"""

import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from config import GROQ_API_KEY, GROQ_MODEL_FAST, LLM_TEMPERATURE
from memory.long_term import retrieve_past_verdict

logger = logging.getLogger(__name__)


# ── Pydantic schema ────────────────────────────────────────────────────────────
class ToolChoice(BaseModel):
    """Structured output schema for router decision."""
    tool: Literal["fact_db", "web_news", "web_science"] = Field(
        description="The retrieval tool to use for this claim."
    )
    reason: str = Field(
        description="One sentence explaining why this tool was chosen."
    )


# ── Prompt ─────────────────────────────────────────────────────────────────────
ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a routing agent for an Indian WhatsApp fact-checking system.

Given a claim, decide which retrieval tool to use:

- "fact_db": Use for common health myths, home remedies, viral WhatsApp health forwards,
  widely circulated Indian myths (e.g. "hot water cures cancer", "onions absorb viruses")
  
- "web_news": Use for claims about recent events, political statements, government actions,
  viral news, or anything that may have happened recently
  
- "web_science": Use for medical facts, scientific claims, nutritional claims,
  biological or chemical claims, vaccine-related claims

On retry: you are told which tool was used before. Choose a DIFFERENT tool.

Return JSON with "tool" and "reason"."""),
    ("human", """Claim: {claim}
Previously used tool (empty if first attempt): {previous_tool}""")
])


# ── Node function ──────────────────────────────────────────────────────────────
def route_claim(state: dict) -> dict:
    """
    LangGraph node: Decide which retrieval tool to use for the current claim.

    Reads:
        state["current_claim"]: The claim being processed.
        state["tool_choice"]: Previously used tool (empty on first attempt).
        state["retry_count"]: Number of retries so far.

    Writes:
        state["tool_choice"]: Selected tool name.
        state["retry_count"]: Incremented if this is a retry.
        state["reasoning_trail"]: Appended with routing decision.

    Args:
        state: LangGraph agent state dictionary.

    Returns:
        dict: Updated state with tool_choice populated.
    """
    claim = state.get("current_claim", "").strip()
    previous_tool = state.get("tool_choice", "")
    retry_count = state.get("retry_count", 0)

    if not claim:
        # Check long-term memory cache first (only on first attempt, not retries)
        if retry_count == 0:
            cached = retrieve_past_verdict(claim)
            if cached is not None:
                verdicts = state.get("verdicts", []) + [cached]
                reasoning = (
                    f"⚡ Cache hit for '{claim[:60]}' → {cached['verdict']} "
                    f"({cached['confidence']}% confidence) — skipping retrieval."
                )
                logger.info(
                    "Cache hit for claim: %s → %s", claim[:60], cached["verdict"]
                )
                return {
                    **state,
                    "tool_choice": "cached",
                    "verdicts": verdicts,
                    "reasoning_trail": state.get("reasoning_trail", []) + [reasoning],
                }
        logger.warning("route_claim called with empty claim.")
        return {
            **state,
            "tool_choice": "fact_db",
            "reasoning_trail": state.get("reasoning_trail", []) + [
                "⚠️ Empty claim — defaulting to fact_db."
            ],
        }

    try:
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL_FAST,
            temperature=LLM_TEMPERATURE,
        ).with_structured_output(ToolChoice)

        chain = ROUTER_PROMPT | llm
        result: ToolChoice = chain.invoke({
            "claim": claim,
            "previous_tool": previous_tool if retry_count > 0 else "",
        })

        # Increment retry count only when retrying
        new_retry_count = retry_count + 1 if previous_tool else retry_count

        reasoning = (
            f"🔀 Router → [{result.tool}]"
            + (f" (retry #{new_retry_count})" if previous_tool else "")
            + f" — {result.reason}"
        )

        logger.info(
            "Routed claim to [%s] (retry=%d): %s",
            result.tool,
            new_retry_count,
            claim[:60],
        )

        return {
            **state,
            "tool_choice": result.tool,
            "retry_count": new_retry_count,
            "reasoning_trail": state.get("reasoning_trail", []) + [reasoning],
        }

    except Exception as e:
        logger.error("route_claim failed: %s", e)
        return {
            **state,
            "tool_choice": "fact_db",
            "retry_count": retry_count,
            "reasoning_trail": state.get("reasoning_trail", []) + [
                f"❌ Routing failed: {e} — defaulting to fact_db."
            ],
        }