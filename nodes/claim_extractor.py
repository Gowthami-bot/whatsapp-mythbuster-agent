"""
Claim Extractor Node for WhatsApp Myth Buster Agent.
Extracts individual verifiable factual claims from a WhatsApp message
using Groq LLM with structured output via Pydantic.
"""

import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from config import GROQ_API_KEY, GROQ_MODEL_FAST, LLM_TEMPERATURE

logger = logging.getLogger(__name__)


# ── Pydantic schema ────────────────────────────────────────────────────────────
class ClaimList(BaseModel):
    """Structured output schema for extracted claims."""
    claims: list[str] = Field(
        description="List of individual verifiable factual claims extracted from the message.",
        min_length=0,
    )


# ── Prompt ─────────────────────────────────────────────────────────────────────
CLAIM_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert fact-checker specializing in Indian WhatsApp misinformation.

Your job is to extract individual, verifiable factual claims from a WhatsApp message.

Rules:
- Extract only specific, checkable factual claims
- Each claim must be a single standalone statement
- Ignore forwarding instructions like "share this with 10 people"
- Ignore emotional appeals, blessings, or religious content
- Ignore greetings and closing statements
- Keep each claim concise and clear
- Return between 1 and 5 claims maximum
- If no verifiable claims exist, return an empty list"""),
    ("human", "WhatsApp message:\n{message}")
])


# ── Node function ──────────────────────────────────────────────────────────────
def extract_claims(state: dict) -> dict:
    """
    LangGraph node: Extract individual verifiable claims from a WhatsApp message.

    Reads:
        state["whatsapp_message"]: Raw WhatsApp message string.

    Writes:
        state["claims"]: List of extracted claim strings.
        state["current_claim"]: First claim to process.
        state["current_claim_index"]: Set to 0.
        state["reasoning_trail"]: Appended with extraction summary.

    Args:
        state: LangGraph agent state dictionary.

    Returns:
        dict: Updated state with claims populated.
    """
    message = state.get("whatsapp_message", "").strip()

    if not message:
        logger.warning("extract_claims called with empty message.")
        return {
            **state,
            "claims": [],
            "current_claim": "",
            "current_claim_index": 0,
            "reasoning_trail": state.get("reasoning_trail", []) + [
                "⚠️ Empty message received — no claims extracted."
            ],
        }

    try:
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL_FAST,
            temperature=LLM_TEMPERATURE,
        ).with_structured_output(ClaimList)

        chain = CLAIM_EXTRACTION_PROMPT | llm
        result: ClaimList = chain.invoke({"message": message})
        claims = result.claims

        logger.info("Extracted %d claim(s) from message.", len(claims))

        reasoning = (
            f"🔍 Extracted {len(claims)} claim(s): "
            + " | ".join(f'"{c}"' for c in claims)
            if claims
            else "🔍 No verifiable claims found in message."
        )

        return {
            **state,
            "claims": claims,
            "current_claim": claims[0] if claims else "",
            "current_claim_index": 0,
            "reasoning_trail": state.get("reasoning_trail", []) + [reasoning],
        }

    except Exception as e:
        logger.error("claim_extractor failed: %s", e)
        return {
            **state,
            "claims": [],
            "current_claim": "",
            "current_claim_index": 0,
            "reasoning_trail": state.get("reasoning_trail", []) + [
                f"❌ Claim extraction failed: {e}"
            ],
        }