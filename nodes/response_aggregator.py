"""
Response Aggregator Node for WhatsApp Myth Buster Agent.
Final node in the LangGraph pipeline.
Aggregates all per-claim verdicts into a structured final response
and appends a summary to the reasoning trail.
"""

import logging

logger = logging.getLogger(__name__)

# ── Verdict emoji map ──────────────────────────────────────────────────────────
_VERDICT_EMOJI = {
    "True": "✅",
    "False": "❌",
    "Misleading": "⚠️",
    "Unverifiable": "❓",
}


def aggregate_responses(state: dict) -> dict:
    """
    LangGraph node: Aggregate all claim verdicts into a final response.
    This is the terminal node — runs once after all claims are processed.

    Reads:
        state["verdicts"]: All collected verdicts.
        state["claims"]: Original list of extracted claims.
        state["reasoning_trail"]: Full reasoning trail.

    Writes:
        state["reasoning_trail"]: Appended with final summary.

    Args:
        state: LangGraph agent state dictionary.

    Returns:
        dict: Final state with complete verdicts and reasoning trail.
    """
    verdicts = state.get("verdicts", [])
    claims = state.get("claims", [])

    if not verdicts:
        logger.warning("aggregate_responses called with no verdicts.")
        return {
            **state,
            "reasoning_trail": state.get("reasoning_trail", []) + [
                "⚠️ No verdicts to aggregate."
            ],
        }

    # Build summary line per verdict
    summary_lines = []
    for v in verdicts:
        emoji = _VERDICT_EMOJI.get(v.get("verdict", ""), "❓")
        cached_label = " [cached]" if v.get("cached") else ""
        line = (
            f"{emoji} {v.get('verdict', 'Unknown')} "
            f"({v.get('confidence', 0)}%){cached_label} — "
            f"{v.get('claim', '')[:60]}"
        )
        summary_lines.append(line)

    summary = (
        f"✅ Analysis complete — {len(verdicts)}/{len(claims)} claim(s) processed.\n"
        + "\n".join(summary_lines)
    )

    logger.info(
        "Aggregation complete: %d verdict(s) for %d claim(s).",
        len(verdicts),
        len(claims),
    )

    return {
        **state,
        "reasoning_trail": state.get("reasoning_trail", []) + [summary],
    }