"""
Short-term memory module for WhatsApp Myth Buster Agent.
Short-term memory is handled by LangGraph MemorySaver in main.py.
This module provides helper utilities for conversation context formatting.
"""

import logging

logger = logging.getLogger(__name__)


def format_conversation_history(messages: list[dict]) -> str:
    """
    Format the last N conversation turns into a readable string
    for injection into LLM prompts.

    Args:
        messages: List of message dicts with "role" and "content" keys.

    Returns:
        str: Formatted conversation history string.
    """
    if not messages:
        return "No previous conversation."

    formatted = []
    for msg in messages[-6:]:  # last 3 turns
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "").strip()
        if content:
            formatted.append(f"{role}: {content}")

    return "\n".join(formatted) if formatted else "No previous conversation."