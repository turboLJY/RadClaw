"""
Context compaction for RadClaw.
Compresses conversation history when context grows too large.
Adapted from SearchClaw with medical imaging-specific preservation rules.
"""

from __future__ import annotations

import logging

from src.core.types import Message

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    return len(text) // 3


def should_compact(messages: list[Message], threshold_tokens: int = 50000) -> bool:
    total = sum(estimate_tokens(msg.text_content) for msg in messages)
    return total > threshold_tokens


async def compact_messages(
    messages: list[Message],
    threshold_tokens: int = 50000,
) -> list[Message]:
    # Phase 1: Microcompact — drop old tool results, keep first user + last N
    keep_last = 6
    if len(messages) <= keep_last + 1:
        return messages

    first_user = None
    for m in messages:
        if m.role == "user" and not m.metadata.get("_tag"):
            first_user = m
            break

    recent = messages[-keep_last:]

    compacted = []
    if first_user and first_user not in recent:
        compacted.append(first_user)

    compacted.append(Message(
        role="system",
        content="[Earlier conversation compacted to save context. Key information preserved in recent messages.]",
    ))
    compacted.extend(recent)

    logger.info(f"Compacted {len(messages)} messages to {len(compacted)}")
    return compacted