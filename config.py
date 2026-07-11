"""
Configuration module for WhatsApp Myth Buster Agent.
Loads all environment variables and exposes typed constants.
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── LLM ───────────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
GROQ_MODEL_FAST: str = os.getenv("GROQ_MODEL_FAST", "llama-3.1-8b-instant")
GROQ_MODEL_QUALITY: str = os.getenv("GROQ_MODEL_QUALITY", "llama-3.3-70b-versatile")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# ── Web Search ─────────────────────────────────────────────────────────────────
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

# ── LangSmith Tracing ──────────────────────────────────────────────────────────
LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "whatsapp-myth-buster")

# ── Qdrant ─────────────────────────────────────────────────────────────────────
QDRANT_PATH: str = os.getenv("QDRANT_PATH", "./qdrant_storage")
QDRANT_FACT_COLLECTION: str = os.getenv("QDRANT_FACT_COLLECTION", "fact_check_db")
QDRANT_MEMORY_COLLECTION: str = os.getenv("QDRANT_MEMORY_COLLECTION", "verdict_memory")
QDRANT_URL: str = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")

# ── Models ─────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIMENSION: int = 384      # bge-small-en-v1.5 output size
RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

# ── Retrieval ──────────────────────────────────────────────────────────────────
TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "10"))
TOP_K_RERANKED: int = int(os.getenv("TOP_K_RERANKED", "3"))
BM25_WEIGHT: float = 0.4
QDRANT_WEIGHT: float = 0.6

# ── Agent ──────────────────────────────────────────────────────────────────────
MAX_RETRY_COUNT: int = int(os.getenv("MAX_RETRY_COUNT", "2"))
REFLECTION_THRESHOLD: float = float(os.getenv("REFLECTION_THRESHOLD", "0.4"))

# ── Corpus Sources ─────────────────────────────────────────────────────────────
BOOMLIVE_RSS: str = "https://www.boomlive.in/fact-check/feed"
ALTNEWS_RSS: str = "https://www.altnews.in/feed/"
PIB_RSS: str = "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3"
WHO_RSS: str = "https://www.who.int/rss-feeds/news-releases.xml"
IFND_CSV_PATH: str = os.getenv("IFND_CSV_PATH", "./data/raw/IFND.csv")

# ── Validation ─────────────────────────────────────────────────────────────────
def validate_config() -> bool:
    """
    Validate that all required environment variables are set.
    Returns True if valid, False otherwise.
    """
    missing = []

    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not TAVILY_API_KEY:
        missing.append("TAVILY_API_KEY")
    if not LANGCHAIN_API_KEY:
        missing.append("LANGCHAIN_API_KEY")

    if missing:
        logger.error("Missing required environment variables: %s", ", ".join(missing))
        return False

    logger.info("Configuration validated successfully.")
    return True


# ── Logging setup ──────────────────────────────────────────────────────────────
import logging
import sys

def setup_logging(level: str = "INFO") -> None:
    """
    Configure centralized logging for the entire application.
    Shows timestamp, level, module name, and message in terminal.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    logging.getLogger(__name__).info("Logging initialized at level: %s", level)