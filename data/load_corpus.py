"""
Corpus loader for WhatsApp Myth Buster Agent.
Loads fact-check data from 5 sources:
  1. BoomLive RSS feed
  2. AltNews RSS feed
  3. PIB RSS feed
  4. WHO RSS feed
  5. IFND CSV dataset (local file)

Normalizes all sources into LangChain Documents,
stores them in Qdrant fact_check_db collection,
and initializes the BM25Retriever for hybrid search.
"""

import logging
import csv
import feedparser
import requests
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from utils.qdrant_helper import (
    get_vector_store,
    initialize_collections,
    collection_exists,
)
from config import (
    BOOMLIVE_RSS,
    ALTNEWS_RSS,
    PIB_RSS,
    WHO_RSS,
    IFND_CSV_PATH,
    QDRANT_FACT_COLLECTION,
    TOP_K_RETRIEVAL,
)

logger = logging.getLogger(__name__)

# ── Module-level BM25 retriever (singleton) ────────────────────────────────────
_bm25_retriever: BM25Retriever | None = None


def get_bm25_retriever() -> BM25Retriever | None:
    """
    Return the singleton BM25Retriever instance.
    Returns None if corpus has not been loaded yet.

    Returns:
        BM25Retriever | None: The BM25 retriever or None.
    """
    return _bm25_retriever


def _load_rss_feed(url: str, source_name: str) -> list[Document]:
    """
    Parse an RSS feed and return LangChain Documents.

    Args:
        url: RSS feed URL.
        source_name: Human-readable source label.

    Returns:
        list[Document]: Parsed documents from the feed.
    """
    docs: list[Document] = []
    try:
        feed = feedparser.parse(url)
        if feed.bozo:
            logger.warning("RSS feed may be malformed: %s", source_name)

        for entry in feed.entries:
            title = entry.get("title", "").strip()
            summary = entry.get("summary", "").strip()
            content = f"{title}. {summary}".strip()

            if len(content) < 20:
                continue

            doc = Document(
                page_content=content,
                metadata={
                    "source": entry.get("link", url),
                    "title": title,
                    "published": entry.get("published", ""),
                    "origin": source_name,
                    "verdict": "",
                }
            )
            docs.append(doc)

        logger.info("Loaded %d documents from %s", len(docs), source_name)

    except Exception as e:
        logger.error("Failed to load RSS feed %s: %s", source_name, e)

    return docs


def _load_ifnd_csv(filepath: str) -> list[Document]:
    """
    Load the IFND CSV dataset and return LangChain Documents.
    Expected columns: 'statement' (or 'text'), 'label' (fake/real or 0/1).

    Args:
        filepath: Path to the IFND CSV file.

    Returns:
        list[Document]: Parsed documents from the dataset.
    """
    docs: list[Document] = []
    try:
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle different column name conventions
                content = (
                    row.get("statement")
                    or row.get("text")
                    or row.get("claim")
                    or ""
                ).strip()

                if not content or len(content) < 10:
                    continue

                raw_label = str(
                    row.get("label", row.get("Label", ""))
                ).strip().lower()

                verdict = "False" if raw_label in ["fake", "0", "false"] else "True"

                doc = Document(
                    page_content=content,
                    metadata={
                        "source": row.get("source", "IFND"),
                        "origin": "IFND",
                        "verdict": verdict,
                        "published": row.get("date", ""),
                    }
                )
                docs.append(doc)

        logger.info("Loaded %d records from IFND dataset.", len(docs))

    except FileNotFoundError:
        logger.warning("IFND CSV not found at: %s — skipping.", filepath)
    except Exception as e:
        logger.error("Failed to load IFND CSV: %s", e)

    return docs


def _store_in_qdrant(documents: list[Document]) -> None:
    """
    Store documents in Qdrant fact_check_db collection.

    Args:
        documents: List of LangChain Documents to store.
    """
    try:
        vector_store = get_vector_store(QDRANT_FACT_COLLECTION)
        vector_store.add_documents(documents)
        logger.info(
            "Stored %d documents in Qdrant collection: %s",
            len(documents),
            QDRANT_FACT_COLLECTION,
        )
    except Exception as e:
        logger.error("Failed to store documents in Qdrant: %s", e)
        raise


def _init_bm25(documents: list[Document]) -> None:
    """
    Initialize the module-level BM25Retriever from documents.

    Args:
        documents: List of LangChain Documents to index.
    """
    global _bm25_retriever
    try:
        _bm25_retriever = BM25Retriever.from_documents(documents)
        _bm25_retriever.k = TOP_K_RETRIEVAL
        logger.info("BM25Retriever initialized with %d documents.", len(documents))
    except Exception as e:
        logger.error("Failed to initialize BM25Retriever: %s", e)
        raise


def build_fact_check_db(force_reload: bool = False) -> list[Document]:
    """
    Build the fact-check database from all corpus sources.
    Loads RSS feeds + IFND CSV, stores in Qdrant, initializes BM25.

    If Qdrant collection already has data and force_reload=False,
    skips Qdrant ingestion but still initializes BM25 from RSS data
    so hybrid search works correctly on restart.

    Args:
        force_reload: If True, re-ingests all data into Qdrant.

    Returns:
        list[Document]: All loaded documents.
    """
    logger.info("Starting corpus build...")

    # Ensure collections exist
    initialize_collections()

    all_docs: list[Document] = []

    # Load all sources
    all_docs += _load_rss_feed(BOOMLIVE_RSS, "BoomLive")
    all_docs += _load_rss_feed(ALTNEWS_RSS, "AltNews")
    all_docs += _load_rss_feed(PIB_RSS, "PIB")
    all_docs += _load_rss_feed(WHO_RSS, "WHO")
    all_docs += _load_ifnd_csv(IFND_CSV_PATH)

    if not all_docs:
        logger.warning("No documents loaded from any source.")
        return all_docs

    logger.info("Total documents loaded: %d", len(all_docs))

    # Store in Qdrant only if collection is empty or force_reload
    client_has_data = _qdrant_has_data()
    if force_reload or not client_has_data:
        _store_in_qdrant(all_docs)
    else:
        logger.info(
            "Qdrant collection already has data. Skipping ingestion. "
            "Use force_reload=True to re-ingest."
        )

    # Always initialize BM25 (in-memory, rebuilt on each restart)
    _init_bm25(all_docs)

    logger.info("Corpus build complete.")
    return all_docs


def _qdrant_has_data() -> bool:
    """
    Check if the fact_check_db collection already has documents.

    Returns:
        bool: True if collection has at least one document.
    """
    try:
        from utils.qdrant_helper import get_client
        client = get_client()
        result = client.count(collection_name=QDRANT_FACT_COLLECTION)
        has_data = result.count > 0
        logger.info(
            "Qdrant collection %s has %d documents.",
            QDRANT_FACT_COLLECTION,
            result.count,
        )
        return has_data
    except Exception as e:
        logger.warning("Could not check Qdrant data count: %s", e)
        return False


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    build_fact_check_db()