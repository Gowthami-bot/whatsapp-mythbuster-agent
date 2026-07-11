"""
Self Reflection Node for WhatsApp Myth Buster Agent.
Evaluates whether retrieved and reranked evidence is sufficient
to generate a reliable verdict for the current claim.
If insufficient and retry limit not reached, triggers re-routing.
"""

import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from config import GROQ_API_KEY, GROQ_MODEL_FAST, LLM_TEMPERATURE, MAX_RETRY_COUNT

logger = logging.getLogger(__name__)


# ── Pydantic schema ────────────────────────────────────────────────────────────
class ReflectionResult(BaseModel):
    """Structured output schema for evidence quality assessment."""
    result: Literal["sufficient", "insufficient"] = Field(
        description="Whether the evidence is sufficient to generate a verdict."
    )
    score: float = Field(
        description="Evidence quality score between 0.0 and 1.0.",
        ge=0.0,
        le=1.0,
    )
    reason: str = Field(
        description="One sentence explaining the assessment."
    )


# ── Prompt ─────────────────────────────────────────────────────────────────────
REFLECTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a critical evidence evaluator for an Indian WhatsApp fact-checking system.

Given a claim and retrieved evidence, assess whether the evidence is sufficient to generate a reliable verdict.

Mark as "sufficient" if:
- Evidence directly addresses the claim
- At least one credible source is present
- Evidence is clear enough to support True/False/Misleading/Unverifiable verdict

Mark as "insufficient" if:
- Evidence is irrelevant or only loosely related to the claim
- Evidence is too vague to support any verdict
- No credible sources are present
- Evidence is empty

Return JSON with:
- "result": "sufficient" or "insufficient"
- "score": float 0.0 to 1.0 (evidence quality)
- "reason": one sentence explanation"""),
    ("human", """Claim: {claim}

Evidence:
{evidence}""")
])


# ── Node function ──────────────────────────────────────────────────────────────
def reflect(state: dict) -> dict:
    """
    LangGraph node: Assess whether retrieved evidence is sufficient
    to generate a reliable verdict for the current claim.

    Reads:
        state["current_claim"]: The claim being evaluated.
        state["evidence"]: Reranked evidence list.
        state["retry_count"]: Current retry count.

    Writes:
        state["reflection_result"]: "sufficient" or "insufficient".
        state["reasoning_trail"]: Appended with reflection summary.

    Args:
        state: LangGraph agent state dictionary.

    Returns:
        dict: Updated state with reflection_result populated.
    """
    claim = state.get("current_claim", "").strip()
    evidence = state.get("evidence", [])
    retry_count = state.get("retry_count", 0)

    # No evidence — mark insufficient immediately
    if not evidence:
        logger.warning("reflect called with no evidence.")
        return {
            **state,
            "reflection_result": "insufficient",
            "reasoning_trail": state.get("reasoning_trail", []) + [
                "🤔 Reflection: INSUFFICIENT — no evidence retrieved."
            ],
        }

    # Max retries reached — force sufficient to avoid infinite loop
    if retry_count >= MAX_RETRY_COUNT:
        logger.info("Max retries reached — forcing sufficient.")
        return {
            **state,
            "reflection_result": "sufficient",
            "reasoning_trail": state.get("reasoning_trail", []) + [
                f"🤔 Reflection: Max retries ({MAX_RETRY_COUNT}) reached "
                f"— proceeding with available evidence."
            ],
        }

    # Format evidence for prompt
    evidence_text = "\n\n".join([
        f"Source: {e.get('source', 'Unknown')}\n{e.get('content', '')}"
        for e in evidence
    ])

    try:
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL_FAST,
            temperature=LLM_TEMPERATURE,
        ).with_structured_output(ReflectionResult)

        chain = REFLECTION_PROMPT | llm
        result: ReflectionResult = chain.invoke({
            "claim": claim,
            "evidence": evidence_text,
        })

        emoji = "✅" if result.result == "sufficient" else "🔄"
        reasoning = (
            f"🤔 Reflection: {result.result.upper()} "
            f"(score: {round(result.score, 2)}) — {result.reason}"
        )

        logger.info(
            "Reflection result: %s (score: %.2f) for claim: %s",
            result.result,
            result.score,
            claim[:60],
        )

        return {
            **state,
            "reflection_result": result.result,
            "reasoning_trail": state.get("reasoning_trail", []) + [reasoning],
        }

    except Exception as e:
        logger.error("reflect failed: %s", e)
        # On failure, mark sufficient to avoid blocking the agent
        return {
            **state,
            "reflection_result": "sufficient",
            "reasoning_trail": state.get("reasoning_trail", []) + [
                f"❌ Reflection failed: {e} — proceeding with available evidence."
            ],
        }