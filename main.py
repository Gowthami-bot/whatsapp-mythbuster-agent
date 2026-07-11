"""
Main entry point for WhatsApp Myth Buster Agent.
Assembles the LangGraph agent graph, wires all nodes,
defines conditional edges, and exposes the run function.
"""

import logging
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from nodes.claim_extractor import extract_claims
from nodes.router import route_claim
from nodes.retriever import hybrid_retrieve
from nodes.reranker import rerank_evidence
from nodes.self_reflection import reflect
from nodes.verdict_generator import generate_verdict
from nodes.memory_updater import update_memory, check_memory_cache
from nodes.response_aggregator import aggregate_responses
from config import validate_config, MAX_RETRY_COUNT

logger = logging.getLogger(__name__)


# ── Agent State ────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    """
    Frozen agent state schema.
    Every node reads from and writes to only these keys.
    """
    whatsapp_message: str
    claims: list[str]
    current_claim: str
    current_claim_index: int
    evidence: list[dict]
    reflection_result: str
    retry_count: int
    tool_choice: str
    verdicts: list[dict]
    reasoning_trail: list[str]


# ── Conditional edge functions ─────────────────────────────────────────────────
def should_retry_or_verdict(state: AgentState) -> str:
    """
    After self_reflection: decide whether to generate verdict
    or retry retrieval with a different tool.
    """
    reflection = state.get("reflection_result", "")
    retry_count = state.get("retry_count", 0)

    if reflection == "sufficient":
        logger.debug("Reflection sufficient — proceeding to verdict.")
        return "verdict_generator"

    if retry_count >= MAX_RETRY_COUNT:
        logger.info(
            "Max retries (%d) reached — forcing verdict.", MAX_RETRY_COUNT
        )
        return "verdict_generator"

    logger.info(
        "Reflection insufficient (retry %d) — re-routing.", retry_count
    )
    return "router"


def is_cache_hit(state: AgentState) -> str:
    """
    After memory_check: decide whether the claim was found
    in long-term memory (cache hit) or needs full retrieval.
    """
    verdicts = state.get("verdicts", [])
    if verdicts and verdicts[-1].get("cached"):
        logger.info("Cache hit — skipping retrieval pipeline.")
        return "memory_updater"

    logger.debug("Cache miss — proceeding to router.")
    return "router"


def has_more_claims(state: AgentState) -> str:
    """
    After memory_updater: decide whether to process next claim
    or aggregate all results.
    """
    claims = state.get("claims", [])
    current_index = state.get("current_claim_index", 0)

    if current_index < len(claims):
        logger.debug(
            "More claims remaining (%d/%d) — checking memory for next claim.",
            current_index,
            len(claims),
        )
        return "memory_check"

    logger.info("All claims processed — aggregating responses.")
    return "response_aggregator"


# ── Graph builder ──────────────────────────────────────────────────────────────
def build_graph() -> any:
    """
    Build and compile the LangGraph agent graph.
    """
    graph = StateGraph(AgentState)

    graph.add_node("claim_extractor", extract_claims)
    graph.add_node("memory_check", check_memory_cache)
    graph.add_node("router", route_claim)
    graph.add_node("hybrid_retriever", hybrid_retrieve)
    graph.add_node("reranker", rerank_evidence)
    graph.add_node("self_reflection", reflect)
    graph.add_node("verdict_generator", generate_verdict)
    graph.add_node("memory_updater", update_memory)
    graph.add_node("response_aggregator", aggregate_responses)

    graph.set_entry_point("claim_extractor")

    graph.add_edge("claim_extractor", "memory_check")
    graph.add_edge("router", "hybrid_retriever")
    graph.add_edge("hybrid_retriever", "reranker")
    graph.add_edge("reranker", "self_reflection")
    graph.add_edge("verdict_generator", "memory_updater")

    graph.add_conditional_edges(
        "memory_check",
        is_cache_hit,
        {
            "memory_updater": "memory_updater",
            "router": "router",
        }
    )

    graph.add_conditional_edges(
        "self_reflection",
        should_retry_or_verdict,
        {
            "verdict_generator": "verdict_generator",
            "router": "router",
        }
    )

    graph.add_conditional_edges(
        "memory_updater",
        has_more_claims,
        {
            "memory_check": "memory_check",
            "response_aggregator": "response_aggregator",
        }
    )

    graph.add_edge("response_aggregator", END)

    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)
    logger.info("LangGraph agent compiled successfully.")
    return compiled


# ── Singleton compiled graph ───────────────────────────────────────────────────
_graph = None


def get_graph():
    """Return the singleton compiled graph instance."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ── Public run function ────────────────────────────────────────────────────────
def run_myth_buster(
    message: str,
    thread_id: str = "default",
) -> dict:
    """
    Run the WhatsApp Myth Buster Agent on a message.

    Args:
        message: Raw WhatsApp message string to fact-check.
        thread_id: Conversation thread ID for short-term memory.

    Returns:
        dict: Final agent state containing verdicts and reasoning_trail.
    """
    if not message or not message.strip():
        logger.warning("run_myth_buster called with empty message.")
        return {
            "verdicts": [],
            "reasoning_trail": ["⚠️ Empty message provided."],
        }

    initial_state = AgentState(
        whatsapp_message=message.strip(),
        claims=[],
        current_claim="",
        current_claim_index=0,
        evidence=[],
        reflection_result="",
        retry_count=0,
        tool_choice="",
        verdicts=[],
        reasoning_trail=[],
    )

    config = {"configurable": {"thread_id": thread_id}}

    try:
        graph = get_graph()
        result = graph.invoke(initial_state, config=config)
        logger.info(
            "Agent completed: %d verdict(s) for message: %s",
            len(result.get("verdicts", [])),
            message[:60],
        )
        return result

    except Exception as e:
        logger.error("Agent run failed: %s", e)
        return {
            "verdicts": [],
            "reasoning_trail": [f"❌ Agent failed: {e}"],
        }


# ── Initialize function ────────────────────────────────────────────────────────
def initialize(log_level: str = "INFO") -> None:
    """
    Initialize the full application:
    - Setup logging
    - Validate config
    - Pre-load embedding model
    - Pre-load reranker model
    - Load corpus
    - Compile graph
    """
    from config import setup_logging
    setup_logging(log_level)

    logger.info("=" * 60)
    logger.info("WhatsApp Myth Buster Agent — Starting Up")
    logger.info("=" * 60)

    if not validate_config():
        raise RuntimeError("Configuration invalid. Check your .env file.")

    # Pre-load heavy models so first request is fast
    logger.info("Pre-loading embedding model (bge-small-en-v1.5)...")
    from utils.embeddings import get_embeddings
    get_embeddings()
    logger.info("Embeddings ready.")

    logger.info("Pre-loading reranker model (bge-reranker-base)...")
    from nodes.reranker import get_reranker
    get_reranker()
    logger.info("Reranker ready.")

    logger.info("Loading fact-check corpus...")
    from data.load_corpus import build_fact_check_db
    build_fact_check_db()

    logger.info("Compiling LangGraph agent...")
    get_graph()

    logger.info("=" * 60)
    logger.info("Agent ready — all systems loaded.")
    logger.info("=" * 60)


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    initialize()

    test_message = (
        "URGENT: Drinking hot water with lemon cures cancer! "
        "WHO has confirmed this. Share with 10 friends immediately!"
    )

    print(f"\nAnalyzing: {test_message}\n")
    result = run_myth_buster(test_message)

    for verdict in result.get("verdicts", []):
        emoji = {
            "True": "✅", "False": "❌",
            "Misleading": "⚠️", "Unverifiable": "❓"
        }.get(verdict["verdict"], "❓")
        print(
            f"{emoji} {verdict['verdict']} "
            f"({verdict['confidence']}%) — {verdict['claim']}"
        )
        print(f"   {verdict['explanation']}\n")