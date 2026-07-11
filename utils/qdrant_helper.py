"""
Qdrant helper utility for WhatsApp Myth Buster Agent.
Provides singleton Qdrant client, collection management,
and LangChain vector store / retriever factory functions.
"""

import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from utils.embeddings import get_embeddings

from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_FACT_COLLECTION,
    QDRANT_MEMORY_COLLECTION,
    EMBEDDING_DIMENSION,
)

logger = logging.getLogger(__name__)

# ── Singleton Qdrant client ────────────────────────────────────────────────────
_qdrant_client: QdrantClient | None = None

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

def get_client() -> QdrantClient:
    """
    Return the singleton QdrantClient instance (cloud mode).
    Instantiated once on first call, reused on all subsequent calls.

    Returns:
        QdrantClient: The Qdrant client instance.
    """
    global _qdrant_client

    if _qdrant_client is None:
        logger.info("Initializing Qdrant client (cloud) at: %s", QDRANT_URL)
        try:
            _qdrant_client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
            )
            logger.info("Qdrant client initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize Qdrant client: %s", e)
            raise

    return _qdrant_client


def create_collection_if_not_exists(collection_name: str) -> None:
    """
    Create a Qdrant collection if it does not already exist.
    Uses Cosine distance and EMBEDDING_DIMENSION vector size.

    Args:
        collection_name: Name of the collection to create.
    """
    client = get_client()
    try:
        existing = [c.name for c in client.get_collections().collections]
        if collection_name not in existing:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=Distance.COSINE
                )
            )
            logger.info("Created Qdrant collection: %s", collection_name)
        else:
            logger.info("Collection already exists: %s", collection_name)
    except Exception as e:
        logger.error("Failed to create collection %s: %s", collection_name, e)
        raise


def initialize_collections() -> None:
    """
    Initialize both required Qdrant collections on startup.
    Creates fact_check_db and verdict_memory if they don't exist.
    """
    create_collection_if_not_exists(QDRANT_FACT_COLLECTION)
    create_collection_if_not_exists(QDRANT_MEMORY_COLLECTION)
    logger.info("All Qdrant collections initialized.")


def get_vector_store(collection_name: str) -> QdrantVectorStore:
    """
    Return a LangChain QdrantVectorStore for a given collection.

    Args:
        collection_name: The Qdrant collection to connect to.

    Returns:
        QdrantVectorStore: LangChain-compatible vector store.
    """
    try:
        embeddings: HuggingFaceEmbeddings = get_embeddings()
        vector_store = QdrantVectorStore(
            client=get_client(),
            collection_name=collection_name,
            embedding=embeddings
        )
        logger.debug("Vector store created for collection: %s", collection_name)
        return vector_store
    except Exception as e:
        logger.error("Failed to create vector store for %s: %s", collection_name, e)
        raise


def get_qdrant_retriever(collection_name: str, k: int = 10):
    """
    Return a LangChain retriever backed by a Qdrant collection.

    Args:
        collection_name: The Qdrant collection to search.
        k: Number of results to retrieve.

    Returns:
        VectorStoreRetriever: LangChain retriever instance.
    """
    try:
        vector_store = get_vector_store(collection_name)
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        logger.debug(
            "Qdrant retriever created for collection: %s, k=%d",
            collection_name, k
        )
        return retriever
    except Exception as e:
        logger.error(
            "Failed to create retriever for %s: %s", collection_name, e
        )
        raise


def collection_exists(collection_name: str) -> bool:
    """
    Check whether a Qdrant collection exists.

    Args:
        collection_name: Name of the collection to check.

    Returns:
        bool: True if collection exists, False otherwise.
    """
    try:
        client = get_client()
        existing = [c.name for c in client.get_collections().collections]
        return collection_name in existing
    except Exception as e:
        logger.error("Failed to check collection existence: %s", e)
        return False