"""
Memory Updater Node for WhatsApp Myth Buster Agent.
After each verdict is generated:
  1. Stores the verdict in Qdrant verdict_memory collection (long-term memory)
  2. Advances state to the next claim
  3. Resets per-claim state fields for the next iteration
"""

import logging
from memory.long_term import store_verdict, retrieve_past_verdict
from config import MAX_RETRY_COUNT

logger = logging.getLogger(__name__)


def update_memory(state: dict) -> dict:
    """
    LangGraph node: Store the latest verdict in long-term memory
    and advance state to the next claim.

    Reads:
        state["verdicts"]: List of verdicts — uses the last one.
        state["claims"]: Full list of claims.
        state["current_claim_index"]: Index of claim just processed.

    Writes:
        state["current_claim_index"]: Incremented to next claim.
        state["current_claim"]: Set to next claim string.
        state["evidence"]: Reset to empty list.
        state["reflection_result"]: Reset to empty string.
        state["retry_count"]: Reset to 0.
        state["tool_choice"]: Reset to empty string.
        state["reasoning_trail"]: Appended with memory update summary.

    Args:
        state: LangGraph agent state dictionary.

    Returns:
        dict: Updated state ready for next claim or aggregation.
    """
    verdicts = state.get("verdicts", [])
    claims = state.get("claims", [])
    current_index = state.get("current_claim_index", 0)

    if not verdicts:
        logger.warning("update_memory called with no verdicts.")
        return {
            **state,
            "reasoning_trail": state.get("reasoning_trail", []) + [
                "⚠️ Memory update skipped — no verdicts to store."
            ],
        }

    latest_verdict = verdicts[-1]

    # Store in long-term memory
    try:
        store_verdict(latest_verdict)
        reasoning = (
            f"💾 Stored verdict for: '{latest_verdict.get('claim', '')[:60]}' "
            f"→ {latest_verdict.get('verdict', 'Unknown')}"
        )
        logger.info(
            "Stored verdict for claim: %s",
            latest_verdict.get("claim", "")[:60],
        )
    except Exception as e:
        logger.error("Failed to store verdict in memory: %s", e)
        reasoning = f"⚠️ Memory store failed: {e} — continuing."

    # Advance to next claim
    next_index = current_index + 1
    next_claim = claims[next_index] if next_index < len(claims) else ""

    return {
        **state,
        "current_claim_index": next_index,
        "current_claim": next_claim,
        # Reset per-claim fields for next iteration
        "evidence": [],
        "reflection_result": "",
        "retry_count": 0,
        "tool_choice": "",
        "reasoning_trail": state.get("reasoning_trail", []) + [reasoning],
    }


def check_memory_cache(state: dict) -> dict:
    """
    Check if the current claim already has a cached verdict
    in long-term memory. If found, inject cached verdict into state.

    This is a utility function — not a LangGraph node.
    Called optionally before routing to skip redundant retrieval.

    Args:
        state: LangGraph agent state dictionary.

    Returns:
        dict: State with cached verdict injected if found,
              otherwise state unchanged.
    """
    claim = state.get("current_claim", "").strip()

    if not claim:
        return state

    try:
        cached = retrieve_past_verdict(claim)
        if cached:
            logger.info(
                "Cache hit for claim: %s", claim[:60]
            )
            cached["cached"] = True
            existing_verdicts = state.get("verdicts", [])
            return {
                **state,
                "verdicts": existing_verdicts + [cached],
                "reasoning_trail": state.get("reasoning_trail", []) + [
                    f"💾 Cache hit — reusing past verdict for: '{claim[:60]}'"
                ],
            }
    except Exception as e:
        logger.warning("Cache check failed: %s", e)

    return state