"""
Verdict Generator Node for WhatsApp Myth Buster Agent.
Generates a structured verdict for the current claim
based on reranked evidence using Groq LLM.
Produces: verdict label, confidence score, explanation, and sources.
"""

import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from config import GROQ_API_KEY, GROQ_MODEL_QUALITY, LLM_TEMPERATURE

logger = logging.getLogger(__name__)


# ── Pydantic schema ────────────────────────────────────────────────────────────
class VerdictOutput(BaseModel):
    """Structured output schema for claim verdict."""
    verdict: Literal["True", "False", "Misleading", "Unverifiable"] = Field(
        description="The verdict for the claim based on evidence."
    )
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0.",
        ge=0.0,
        le=1.0,
    )
    explanation: str = Field(
        description="1-2 sentence explanation of the verdict."
    )
    sources: list[str] = Field(
        description="List of source URLs or names used to reach this verdict."
    )


# ── Prompt ─────────────────────────────────────────────────────────────────────
VERDICT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert fact-checker for an Indian WhatsApp misinformation detection system.

Based on the provided evidence, generate a verdict for the claim.

Verdict types:
- "True": Claim is factually accurate based on evidence
- "False": Claim is factually incorrect based on evidence
- "Misleading": Claim has partial truth but is framed deceptively
- "Unverifiable": Insufficient or conflicting evidence to confirm or deny

Rules for Explanation:
- Explain the factual "why" behind the verdict using specific details from the evidence (e.g., "The central government has made no such announcement, and PIB Fact Check has flagged this claim as fake.").
- NEVER write generic statements like "The claim that X is false based on the evidence" or simply restate the claim. Provide the actual fact.
- Keep the explanation concise and direct (1-2 sentences maximum).

Rules for Confidence Score:
- High (0.90 to 1.00): There is direct, unambiguous debunking or confirming evidence from highly authoritative sources (e.g., PIB Fact Check, WHO, official government statements, or major fact-checkers like BoomLive/AltNews).
- Medium (0.70 to 0.89): Evidence is strong but circumstantial, or comes from general news sites without direct primary source citations.
- Low (0.00 to 0.69): Evidence is weak, conflicting, partial, or completely absent.

Return JSON matching the VerdictOutput schema."""),
    ("human", """Claim: {claim}

Evidence:
{evidence}""")
])

# ── Emoji map ──────────────────────────────────────────────────────────────────
_VERDICT_EMOJI = {
    "True": "✅",
    "False": "❌",
    "Misleading": "⚠️",
    "Unverifiable": "❓",
}


# ── Node function ──────────────────────────────────────────────────────────────
def generate_verdict(state: dict) -> dict:
    """
    LangGraph node: Generate a structured verdict for the current claim.

    Reads:
        state["current_claim"]: The claim to verdict.
        state["evidence"]: Reranked evidence list.
        state["verdicts"]: Existing verdicts list.

    Writes:
        state["verdicts"]: Appended with new verdict dict.
        state["reasoning_trail"]: Appended with verdict summary.

    Args:
        state: LangGraph agent state dictionary.

    Returns:
        dict: Updated state with new verdict appended.
    """
    claim = state.get("current_claim", "").strip()
    evidence = state.get("evidence", [])
    existing_verdicts = state.get("verdicts", [])

    # Format evidence for prompt
    if evidence:
        evidence_text = "\n\n".join([
            f"Source: {e.get('source', 'Unknown')}\n{e.get('content', '')}"
            for e in evidence
        ])
        sources = list({e.get("source", "Unknown") for e in evidence})
    else:
        evidence_text = "No evidence available."
        sources = []

    try:
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL_QUALITY,
            temperature=LLM_TEMPERATURE,
        ).with_structured_output(VerdictOutput)

        chain = VERDICT_PROMPT | llm
        result: VerdictOutput = chain.invoke({
            "claim": claim,
            "evidence": evidence_text,
        })

        # Build verdict dict matching frozen schema
        verdict_dict = {
            "claim": claim,
            "verdict": result.verdict,
            "confidence": int(round(result.confidence * 100)),
            "explanation": result.explanation,
            "sources": result.sources if result.sources else sources,
            "cached": False,
        }

        emoji = _VERDICT_EMOJI.get(result.verdict, "❓")
        reasoning = (
            f"{emoji} Verdict: {result.verdict} "
            f"({verdict_dict['confidence']}% confidence) — "
            f"{result.explanation}"
        )

        logger.info(
            "Verdict for '%s': %s (%d%% confidence)",
            claim[:60],
            result.verdict,
            verdict_dict["confidence"],
        )

        return {
            **state,
            "verdicts": existing_verdicts + [verdict_dict],
            "reasoning_trail": state.get("reasoning_trail", []) + [reasoning],
        }

    except Exception as e:
        logger.error("generate_verdict failed: %s", e)

        # Fallback verdict on LLM failure
        fallback_verdict = {
            "claim": claim,
            "verdict": "Unverifiable",
            "confidence": 0,
            "explanation": f"Verdict generation failed: {e}",
            "sources": sources,
            "cached": False,
        }

        return {
            **state,
            "verdicts": existing_verdicts + [fallback_verdict],
            "reasoning_trail": state.get("reasoning_trail", []) + [
                f"❌ Verdict generation failed: {e} — marked Unverifiable."
            ],
        }