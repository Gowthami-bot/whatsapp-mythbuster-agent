"""
Hybrid search tool for WhatsApp Myth Buster Agent.
Combines BM25 (keyword) and Qdrant (semantic) retrieval
using LangChain EnsembleRetriever with RRF fusion.
Weights: BM25=0.4, Qdrant=0.6
"""

import logging
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from data.load_corpus import get_bm25_retriever
from utils.qdrant_helper import get_qdrant_retriever
from config import (
    QDRANT_FACT_COLLECTION,
    TOP_K_RETRIEVAL,
    BM25_WEIGHT,
    QDRANT_WEIGHT,
)

logger = logging.getLogger(__name__)

# ── Singleton EnsembleRetriever ────────────────────────────────────────────────
_ensemble_retriever: EnsembleRetriever | None = None


def get_ensemble_retriever() -> EnsembleRetriever:
    """
    Return the singleton EnsembleRetriever instance.
    Combines BM25Retriever and Qdrant retriever with RRF fusion.
    Must be called after build_fact_check_db() has been run.

    Returns:
        EnsembleRetriever: The hybrid retriever instance.

    Raises:
        RuntimeError: If BM25Retriever is not initialized yet.
    """
    global _ensemble_retriever

    if _ensemble_retriever is None:
        bm25 = get_bm25_retriever()
        if bm25 is None:
            raise RuntimeError(
                "BM25Retriever is not initialized. "
                "Call build_fact_check_db() before using hybrid search."
            )

        qdrant = get_qdrant_retriever(
            collection_name=QDRANT_FACT_COLLECTION,
            k=TOP_K_RETRIEVAL
        )

        _ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25, qdrant],
            weights=[BM25_WEIGHT, QDRANT_WEIGHT]
        )
        logger.info(
            "EnsembleRetriever initialized (BM25=%.1f, Qdrant=%.1f).",
            BM25_WEIGHT,
            QDRANT_WEIGHT,
        )

    return _ensemble_retriever


def hybrid_search(query: str) -> list[dict]:
    """
    Search the fact-check database using hybrid retrieval.
    Combines BM25 keyword search and Qdrant semantic search via RRF.

    Args:
        query: The claim or query string to search for.

    Returns:
        list[dict]: Retrieved evidence chunks with content, source, and metadata.
    """
    if not query or not query.strip():
        logger.warning("hybrid_search called with empty query.")
        return []

    try:
        retriever = get_ensemble_retriever()
        docs: list[Document] = retriever.invoke(query)

        results = [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "fact_check_db"),
                "origin": doc.metadata.get("origin", ""),
                "verdict": doc.metadata.get("verdict", ""),
                "published": doc.metadata.get("published", ""),
            }
            for doc in docs
        ]

        logger.info(
            "Hybrid search for '%s' returned %d results.",
            query[:60],
            len(results),
        )
        return results

    except RuntimeError as e:
        logger.error("Hybrid search failed — retriever not ready: %s", e)
        return []
    except Exception as e:
        logger.error("Hybrid search failed for query '%s': %s", query[:60], e)
        return []


def reset_ensemble_retriever() -> None:
    """
    Reset the singleton EnsembleRetriever.
    Useful for testing or after corpus reload.
    """
    global _ensemble_retriever
    _ensemble_retriever = None
    logger.info("EnsembleRetriever singleton reset.")