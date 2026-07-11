"""
Web search tool for WhatsApp Myth Buster Agent.
Uses Tavily API for live news and scientific source searches.
Acts as CRAG fallback when local fact-check DB is insufficient.
"""

import logging
from tavily import TavilyClient
from config import TAVILY_API_KEY, TOP_K_RETRIEVAL

logger = logging.getLogger(__name__)

# ── Singleton Tavily client ────────────────────────────────────────────────────
_tavily_client: TavilyClient | None = None


def get_tavily_client() -> TavilyClient:
    """
    Return the singleton TavilyClient instance.

    Returns:
        TavilyClient: The Tavily client instance.

    Raises:
        RuntimeError: If TAVILY_API_KEY is not set.
    """
    global _tavily_client

    if _tavily_client is None:
        if not TAVILY_API_KEY:
            raise RuntimeError(
                "TAVILY_API_KEY is not set. "
                "Add it to your .env file."
            )
        _tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        logger.info("Tavily client initialized.")

    return _tavily_client


def _parse_tavily_results(results: list[dict], tool_name: str) -> list[dict]:
    """
    Parse raw Tavily results into the standard evidence format.

    Args:
        results: Raw results from Tavily API.
        tool_name: Label for the search tool used.

    Returns:
        list[dict]: Formatted evidence chunks.
    """
    evidence = []
    for r in results:
        content = r.get("content", "").strip()
        if not content:
            continue
        evidence.append({
            "content": content,
            "source": r.get("url", tool_name),
            "origin": tool_name,
            "verdict": "",
            "published": r.get("published_date", ""),
        })
    return evidence


def search_news(query: str) -> list[dict]:
    """
    Search live Indian and international fact-check news sources.
    Used when claim relates to recent events or political statements.

    Args:
        query: The claim to fact-check via news search.

    Returns:
        list[dict]: Retrieved evidence from news sources.
    """
    if not query or not query.strip():
        logger.warning("search_news called with empty query.")
        return []

    try:
        client = get_tavily_client()
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=TOP_K_RETRIEVAL,
            include_domains=[
                "boomlive.in",
                "altnews.in",
                "factchecker.in",
                "thequint.com",
                "ndtv.com",
                "thehindu.com",
                "snopes.com",
                "factcheck.org",
                "reuters.com",
                "apnews.com",
            ]
        )
        results = response.get("results", [])
        evidence = _parse_tavily_results(results, "web_news")
        logger.info(
            "News search for '%s' returned %d results.",
            query[:60],
            len(evidence),
        )
        return evidence

    except RuntimeError as e:
        logger.error("News search failed — client not ready: %s", e)
        return []
    except Exception as e:
        logger.error("News search failed for query '%s': %s", query[:60], e)
        return []


def search_science(query: str) -> list[dict]:
    """
    Search scientific and medical sources for health-related claims.
    Used when claim involves medical, biological, or scientific facts.

    Args:
        query: The claim to fact-check via scientific search.

    Returns:
        list[dict]: Retrieved evidence from scientific sources.
    """
    if not query or not query.strip():
        logger.warning("search_science called with empty query.")
        return []

    try:
        client = get_tavily_client()
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=TOP_K_RETRIEVAL,
            include_domains=[
                "who.int",
                "cdc.gov",
                "pubmed.ncbi.nlm.nih.gov",
                "mayoclinic.org",
                "healthline.com",
                "webmd.com",
                "nih.gov",
                "nature.com",
                "science.org",
                "pib.gov.in",
            ]
        )
        results = response.get("results", [])
        evidence = _parse_tavily_results(results, "web_science")
        logger.info(
            "Science search for '%s' returned %d results.",
            query[:60],
            len(evidence),
        )
        return evidence

    except RuntimeError as e:
        logger.error("Science search failed — client not ready: %s", e)
        return []
    except Exception as e:
        logger.error("Science search failed for query '%s': %s", query[:60], e)
        return []


def reset_tavily_client() -> None:
    """
    Reset the singleton Tavily client.
    Useful for testing or re-initialization with a new key.
    """
    global _tavily_client
    _tavily_client = None
    logger.info("Tavily client singleton reset.")