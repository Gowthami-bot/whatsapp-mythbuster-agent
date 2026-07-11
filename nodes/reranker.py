"""
Reranker Node for WhatsApp Myth Buster Agent.
Uses BAAI/bge-reranker-base to rerank retrieved evidence
by relevance to the current claim.
Keeps top K results after reranking.
"""

import logging
import math
from config import RERANKER_MODEL, TOP_K_RERANKED
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# ── Singleton reranker ─────────────────────────────────────────────────────────

_reranker_instance: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    """
    Return the singleton CrossEncoder reranker instance.
    Loaded once on first call, reused on all subsequent calls.

    Returns:
        CrossEncoder: The BGE reranker instance.
    """
    global _reranker_instance

    if _reranker_instance is None:
        logger.info("Loading reranker model: %s", RERANKER_MODEL)
        try:
            _reranker_instance = CrossEncoder(
                RERANKER_MODEL,
                max_length=512
            )
            logger.info("Reranker model loaded successfully.")
        except Exception as e:
            logger.error("Failed to load reranker model: %s", e)
            raise

    return _reranker_instance


def rerank_evidence(state: dict) -> dict:
    """
    LangGraph node: Rerank retrieved evidence by relevance to the current claim.
    Uses BGE reranker to score each evidence chunk against the claim.
    Keeps top TOP_K_RERANKED results.

    Reads:
        state["current_claim"]: The claim to rerank evidence for.
        state["evidence"]: List of retrieved evidence dicts.

    Writes:
        state["evidence"]: Reranked and trimmed evidence list.
        state["reasoning_trail"]: Appended with reranking summary.

    Args:
        state: LangGraph agent state dictionary.

    Returns:
        dict: Updated state with reranked evidence.
    """
    claim = state.get("current_claim", "").strip()
    evidence = state.get("evidence", [])

    if not evidence:
        logger.warning("rerank_evidence called with no evidence.")
        return {
            **state,
            "evidence": [],
            "reasoning_trail": state.get("reasoning_trail", []) + [
                "⚠️ No evidence to rerank."
            ],
        }

    if not claim:
        logger.warning("rerank_evidence called with empty claim.")
        return {
            **state,
            "reasoning_trail": state.get("reasoning_trail", []) + [
                "⚠️ Empty claim — skipping reranking."
            ],
        }

    try:
        reranker = get_reranker()

        # Build [query, document] pairs for scoring
        texts = [e.get("content", "") for e in evidence]
        pairs = [(claim, text) for text in texts]
        # Score all pairs
        raw_scores = reranker.predict(pairs)
        scores: list[float] = [1 / (1 + math.exp(-s)) for s in raw_scores]

        # Sort by score descending, keep top K
        scored_evidence = sorted(
            zip(scores, evidence),
            key=lambda x: x[0],
            reverse=True,
        )
        top_evidence = [e for _, e in scored_evidence[:TOP_K_RERANKED]]
        top_scores = [round(s, 3) for s, _ in scored_evidence[:TOP_K_RERANKED]]

        reasoning = (
            f"📊 Reranked {len(evidence)} → kept top {len(top_evidence)} "
            f"(scores: {top_scores})"
        )

        logger.info(
            "Reranked %d evidence chunks → kept top %d for claim: %s",
            len(evidence),
            len(top_evidence),
            claim[:60],
        )

        return {
            **state,
            "evidence": top_evidence,
            "reasoning_trail": state.get("reasoning_trail", []) + [reasoning],
        }

    except Exception as e:
        logger.error("rerank_evidence failed: %s", e)
        # Return original evidence untouched on failure
        return {
            **state,
            "reasoning_trail": state.get("reasoning_trail", []) + [
                f"❌ Reranking failed: {e} — using original order."
            ],
        }