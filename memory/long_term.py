"""
Long-term memory module for WhatsApp Myth Buster Agent.
Stores and retrieves past verified claim verdicts
using Qdrant verdict_memory collection.
"""

import logging
from langchain_core.documents import Document
from utils.qdrant_helper import get_vector_store
from config import QDRANT_MEMORY_COLLECTION

logger = logging.getLogger(__name__)


def store_verdict(verdict: dict) -> None:
    """
    Store a verified verdict in Qdrant long-term memory.

    Args:
        verdict: Verdict dict matching the frozen verdict schema.
    """
    try:
        vector_store = get_vector_store(QDRANT_MEMORY_COLLECTION)

        doc = Document(
            page_content=verdict.get("claim", ""),
            metadata={
                "verdict": verdict.get("verdict", ""),
                "confidence": verdict.get("confidence", 0),
                "explanation": verdict.get("explanation", ""),
                "sources": ", ".join(verdict.get("sources", [])),
                "cached": False,
            }
        )

        vector_store.add_documents([doc])
        logger.info(
            "Stored verdict in long-term memory: %s → %s",
            verdict.get("claim", "")[:60],
            verdict.get("verdict", ""),
        )

    except Exception as e:
        logger.error("Failed to store verdict in long-term memory: %s", e)
        raise


def retrieve_past_verdict(claim: str) -> dict | None:
    """
    Check if a similar claim was already verified in long-term memory.
    Returns cached verdict if similarity score is high enough.

    Args:
        claim: The claim string to look up.

    Returns:
        dict | None: Cached verdict dict if found, None otherwise.
    """
    if not claim or not claim.strip():
        return None

    try:
        vector_store = get_vector_store(QDRANT_MEMORY_COLLECTION)
        results = vector_store.similarity_search_with_score(claim, k=1)

        if not results:
            return None

        doc, score = results[0]

        # Cosine similarity: higher = more similar
        # Threshold 0.90 = very high similarity only (near-identical claims)
        if score > 0.90:
            logger.info(
                "Cache hit for claim '%s' (similarity: %.4f)",
                claim[:60],
                score,
            )
            sources_str = doc.metadata.get("sources", "")
            return {
                "claim": doc.page_content,
                "verdict": doc.metadata.get("verdict", "Unverifiable"),
                "confidence": doc.metadata.get("confidence", 0),
                "explanation": doc.metadata.get("explanation", ""),
                "sources": (
                    sources_str.split(", ") if sources_str else []
                ),
                "cached": True,
            }

        logger.debug(
            "Cache miss for claim '%s' (similarity: %.4f)",
            claim[:60],
            score,
        )
        return None
    
    except Exception as e:
        logger.warning("Failed to retrieve past verdict: %s", e)
        return None