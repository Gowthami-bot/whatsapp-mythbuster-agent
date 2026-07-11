"""
Retriever Node for WhatsApp Myth Buster Agent.
Routes retrieval to the correct tool based on router's decision:
  - fact_db  → hybrid_search (BM25 + Qdrant)
  - web_news → search_news (Tavily news domains)
  - web_science → search_science (Tavily scientific domains)
"""

import logging
from tools.hybrid_search import hybrid_search
from tools.web_search import search_news, search_science

logger = logging.getLogger(__name__)

# ── Tool registry ──────────────────────────────────────────────────────────────
_TOOL_REGISTRY: dict[str, callable] = {
    "fact_db": hybrid_search,
    "web_news": search_news,
    "web_science": search_science,
}


def hybrid_retrieve(state: dict) -> dict:
    """
    LangGraph node: Retrieve evidence for the current claim
    using the tool selected by the router node.

    Reads:
        state["current_claim"]: The claim to retrieve evidence for.
        state["tool_choice"]: Which tool to use ("fact_db" | "web_news" | "web_science").

    Writes:
        state["evidence"]: List of retrieved evidence dicts.
        state["reasoning_trail"]: Appended with retrieval summary.

    Args:
        state: LangGraph agent state dictionary.

    Returns:
        dict: Updated state with evidence populated.
    """
    claim = state.get("current_claim", "").strip()
    tool_choice = state.get("tool_choice", "fact_db")

    if not claim:
        logger.warning("hybrid_retrieve called with empty claim.")
        return {
            **state,
            "evidence": [],
            "reasoning_trail": state.get("reasoning_trail", []) + [
                "⚠️ Empty claim — skipping retrieval."
            ],
        }

    tool_fn = _TOOL_REGISTRY.get(tool_choice)

    if tool_fn is None:
        logger.error("Unknown tool_choice: %s — falling back to fact_db.", tool_choice)
        tool_fn = hybrid_search
        tool_choice = "fact_db"

    try:
        evidence: list[dict] = tool_fn(claim)

        tool_labels = {
            "fact_db": "📚 Fact DB (Hybrid)",
            "web_news": "🌐 Web News",
            "web_science": "🔬 Web Science",
        }
        label = tool_labels.get(tool_choice, tool_choice)
        reasoning = (
            f"{label} search for: '{claim[:60]}' "
            f"→ {len(evidence)} result(s) found."
        )

        logger.info(
            "Retrieved %d result(s) using [%s] for claim: %s",
            len(evidence),
            tool_choice,
            claim[:60],
        )

        return {
            **state,
            "evidence": evidence,
            "reasoning_trail": state.get("reasoning_trail", []) + [reasoning],
        }

    except Exception as e:
        logger.error(
            "hybrid_retrieve failed for tool [%s], claim '%s': %s",
            tool_choice,
            claim[:60],
            e,
        )
        return {
            **state,
            "evidence": [],
            "reasoning_trail": state.get("reasoning_trail", []) + [
                f"❌ Retrieval failed using [{tool_choice}]: {e}"
            ],
        }