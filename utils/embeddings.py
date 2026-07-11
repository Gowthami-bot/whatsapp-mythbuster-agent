"""
Embeddings utility module for WhatsApp Myth Buster Agent.
Provides a singleton HuggingFace embedding model instance.
Uses BAAI/bge-small-en-v1.5 — outputs 384-dimensional vectors.
"""

import logging
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# ── Singleton instance ─────────────────────────────────────────────────────────
_embeddings_instance: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return the singleton HuggingFaceEmbeddings instance.
    Instantiated once on first call, reused on all subsequent calls.
    normalize_embeddings=True is required for BGE models.

    Returns:
        HuggingFaceEmbeddings: The embedding model instance.
    """
    global _embeddings_instance

    if _embeddings_instance is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        try:
            _embeddings_instance = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.error("Failed to load embedding model: %s", e)
            raise

    return _embeddings_instance


def embed_text(text: str) -> list[float]:
    """
    Embed a single text string into a vector.

    Args:
        text: The input string to embed.

    Returns:
        list[float]: 384-dimensional embedding vector.
    """
    try:
        embeddings = get_embeddings()
        vector = embeddings.embed_query(text)
        logger.debug("Embedded text of length %d", len(text))
        return vector
    except Exception as e:
        logger.error("Failed to embed text: %s", e)
        raise


def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of documents into vectors.

    Args:
        texts: List of strings to embed.

    Returns:
        list[list[float]]: List of 384-dimensional embedding vectors.
    """
    try:
        embeddings = get_embeddings()
        vectors = embeddings.embed_documents(texts)
        logger.info("Embedded %d documents.", len(texts))
        return vectors
    except Exception as e:
        logger.error("Failed to embed documents: %s", e)
        raise