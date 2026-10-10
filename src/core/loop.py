"""
Agentic loop for RadClaw — single-agent execution engine.

This module provides ``query_loop``, a bidirectional async generator that
drives one agent through its tool-using conversation with the LLM.  The
same loop is reused for every agent in the pipeline; only the
``QueryParams`` configuration changes.

Key features:
- Streaming: yields ``StreamEvent`` objects as the LLM generates text and
  invokes tools.
- Vision support: the first message can include ``image_url`` content blocks
  for the VLM to process.
- Tool execution: runs tools in parallel (when concurrency-safe) or
  sequentially, with input validation.
- Stop hooks: optional quality gates that can inject feedback and request
  continuation.
- Compaction: strips tool messages when the conversation grows too long.
- Guard rail: forces a final text-only answer when ``max_turns`` is reached.

Adapted from SearchClaw's loop.py with vision-input extensions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncGenerator

from src.core.tool import Tool, ToolRegistry, ToolUseContext
from src.core.types import (
    ContentBlock,
    EventType,
    LoopState,
    Message,
    StreamEvent,
    ToolResult,
)
from src.llm.client import LLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query Parameters
# ---------------------------------------------------------------------------

@dataclass
class QueryParams:
    """Configuration for a single agentic loop invocation."""

    query: str
    system_prompt: str
    tool_registry: ToolRegistry
    llm_client: LLMClient
    history: list[Message] = field(default_factory=list)
    max_turns: int = 20
    compact_threshold_tokens: int = 50000
    session_id: str = ""
    hook_engine: object | None = None
    cache_dir: str = "./cache"
    temperature: float | None = None
    max_tokens: int | None = None
    # For vision: list of images to include in the first user message
    images: list[dict] | None = None  # [{"url": "data:image/png;base64,..."}]


# ---------------------------------------------------------------------------
# Agentic Loop
# ---------------------------------------------------------------------------

async def query_loop(
    params: QueryParams,
) -> AsyncGenerator[StreamEvent, str | None]:
    """Bidirectional async generator that drives one agent.

    Yields ``StreamEvent`` instances.  The caller may send a string back
    via ``.asend(answer)`` in response to a ``USER_QUESTION`` event (for
    interactive clarification loops); otherwise use ``.__anext__()``.

    Parameters
    ----------
    params : QueryParams
        Everything the loop needs: query, system prompt, tools, LLM client,
        and configuration knobs.

    Yields
    ------
    StreamEvent
        Events for text deltas, tool usage, tool results, status updates,
        errors, and the final ``DONE`` event.
    """

    # --- 1. Initialize state ------------------------------------------------

    state = LoopState(
        messages=list(params.history),  # shallow copy
    )

    # Build the first user message, optionally with vision content
    first_message = _build_first_message(params.query, params.images)
    state.messages.append(first_message)

    yield StreamEvent(
        type=EventType.STATUS,
        data={"message": "Starting agentic loop", "turn": 0},
    )

    # Build the tool-use context shared across all tool calls in this session
    from pathlib import Path

    extra_context = getattr(params, "_extra_context", {})
    tool_context = ToolUseContext(
        session_id=params.session_id or str(uuid.uuid4()),
        cache_dir=Path(params.cache_dir),
        extra=extra_context,
    )

    # Gather API schemas and concurrent-safe set once
    api_tools = params.tool_registry.get_api_schemas()
    concurrent_safe = params.tool_registry.get_concurrent_safe()

    # --- 2. Main loop -------------------------------------------------------

    while True:
        state.turn_count += 1

        # Guard: max turns exceeded → force final answer
        if state.turn_count > params.max_turns:
            logger.warning(
                "Max turns (%d) exceeded, forcing final answer", params.max_turns,
            )
            yield StreamEvent(
                type=EventType.STATUS,
                data={"message": "Max turns reached, generating final answer"},
            )
            final_text = await _final_answer(params, state)
            state.messages.append(
                Message(role="assistant", content=final_text)
            )
            yield StreamEvent(
                type=EventType.TEXT_DELTA,
                data={"text": final_text},
            )
            break

        # Convert messages to API format
        api_messages = [msg.to_api_dict() for msg in state.messages]

        # --- 2a. Call LLM with streaming ------------------------------------

        accumulated_text = ""
        tool_calls: list[dict] = []

        async for event in params.llm_client.stream(
            messages=api_messages,
            system_prompt=params.system_prompt,
            tools=api_tools if api_tools else None,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
        ):
            if event.type == EventType.TEXT_DELTA:
                accumulated_text += event.data.get("text", "")
                yield event

            elif event.type == EventType.TOOL_USE:
                tool_calls.append(event.data)
                yield event

            elif event.type == EventType.ERROR:
                yield event
                # On error, append what we have and break
                if accumulated_text:
                    state.messages.append(
                        Message(role="assistant", content=accumulated_text)
                    )
                yield StreamEvent(type=EventType.DONE, data={"reason": "error"})
                return

            elif event.type == EventType.STATUS:
                yield event

        # --- 2b. Process response -------------------------------------------

        if not tool_calls:
            # Empty response (no text, no tools) — likely thinking exhausted token budget
            if not accumulated_text.strip() and state.turn_count < params.max_turns:
                logger.warning(
                    "Empty LLM response on turn %d, injecting continuation prompt",
                    state.turn_count,
                )
                state.messages.append(
                    Message(role="assistant", content=accumulated_text or " ")
                )
                state.messages.append(
                    Message(
                        role="user",
                        content=(
                            "Your previous response was empty. Please continue your analysis. "
                            "Use the available tools to proceed with the next step."
                        ),
                    )
                )
                continue

            # Detect unparsed tool calls (model output XML that sglang didn't parse)
            if _contains_unparsed_tool_call(accumulated_text) and state.turn_count < params.max_turns:
                logger.warning(
                    "Detected unparsed tool call in text on turn %d, requesting retry",
                    state.turn_count,
                )
                state.messages.append(
                    Message(role="assistant", content=accumulated_text)
                )
                state.messages.append(
                    Message(
                        role="user",
                        content=(
                            "Your previous tool call was not properly formatted and could not be executed. "
                            "Please call the tool again using the correct function call format. "
                            "Do NOT output tool calls as XML or raw text."
                        ),
                    )
                )
                continue

            # Pure text response — the agent is done (or answering)
            state.messages.append(
                Message(role="assistant", content=accumulated_text)
            )

            # Run stop hooks (quality gates)
            hook_feedback = await _run_stop_hooks(params, state)
            if hook_feedback:
                # Hook says "not done yet" — inject feedback and continue
                yield StreamEvent(
                    type=EventType.STATUS,
                    data={"message": f"Stop hook feedback: {hook_feedback}"},
                )
                state.messages.append(
                    Message(role="user", content=hook_feedback)
                )
                continue

            # No feedback → loop is done
            break

        # --- 2c. Record assistant message with tool calls -------------------

        content_blocks: list[ContentBlock] = []
        if accumulated_text:
            content_blocks.append(
                ContentBlock(type="text", text=accumulated_text)
            )
        for tc in tool_calls:
            content_blocks.append(
                ContentBlock(
                    type="tool_use",
                    tool_name=tc["tool_name"],
                    tool_use_id=tc["tool_use_id"],
                    tool_input=tc.get("tool_input", {}),
                )
            )

        state.messages.append(
            Message(role="assistant", content=content_blocks)
        )

        # --- 2d. Execute tools ----------------------------------------------

        results = await _execute_tools(
            tool_calls=tool_calls,
            registry=params.tool_registry,
            context=tool_context,
            concurrent_safe=concurrent_safe,
        )

        # Inject tool results into conversation
        for tc, result in zip(tool_calls, results):
            yield StreamEvent(
                type=EventType.TOOL_RESULT,
                data={
                    "tool_use_id": tc["tool_use_id"],
                    "tool_name": tc["tool_name"],
                    "result": result.data[:2000],  # Truncate for event
                    "is_error": result.is_error,
                },
            )

            state.messages.append(
                Message(
                    role="tool",
                    content=result.data,
                    metadata={"tool_call_id": tc["tool_use_id"]},
                )
            )

        # --- 2e. Compaction check -------------------------------------------

        estimated_tokens = _estimate_tokens(state.messages)
        if estimated_tokens > params.compact_threshold_tokens:
            logger.info(
                "Token estimate %d exceeds threshold %d, compacting",
                estimated_tokens,
                params.compact_threshold_tokens,
            )
            state.messages = _condense_for_history(state.messages)
            state.compaction_count += 1
            yield StreamEvent(
                type=EventType.STATUS,
                data={
                    "message": "Conversation compacted",
                    "compaction_count": state.compaction_count,
                },
            )

    # --- 3. Done ------------------------------------------------------------

    yield StreamEvent(
        type=EventType.DONE,
        data={
            "reason": "complete",
            "turns": state.turn_count,
            "compactions": state.compaction_count,
            "final_text": state.last_assistant_message or "",
        },
    )


# ---------------------------------------------------------------------------
# First Message Builder
# ---------------------------------------------------------------------------

def _build_first_message(query: str, images: list[dict] | None) -> Message:
    """Build the first user message, optionally with image content blocks.

    For vision-capable models, images are included as ``image_url`` content
    blocks before the text query.
    """
    if not images:
        return Message(role="user", content=query)

    # Build multimodal content blocks
    blocks: list[ContentBlock] = []

    for img in images:
        url = img.get("url", "")
        if not url:
            continue
        blocks.append(
            ContentBlock(
                type="image_url",
                image_url={"url": url},
            )
        )

    # Add the text query after images
    blocks.append(ContentBlock(type="text", text=query))

    return Message(role="user", content=blocks)


# ---------------------------------------------------------------------------
# Tool Execution
# ---------------------------------------------------------------------------

async def _execute_tools(
    tool_calls: list[dict],
    registry: ToolRegistry,
    context: ToolUseContext,
    concurrent_safe: set[str],
) -> list[ToolResult]:
    """Execute a batch of tool calls, using parallelism where safe.

    Tools marked as concurrency-safe run in parallel via asyncio.gather.
    Others run sequentially to avoid race conditions on shared state.
    """
    if len(tool_calls) == 1:
        result = await _execute_single_tool(tool_calls[0], registry, context)
        return [result]

    # Separate concurrent-safe and sequential tools
    safe_calls: list[tuple[int, dict]] = []
    seq_calls: list[tuple[int, dict]] = []

    for idx, tc in enumerate(tool_calls):
        if tc["tool_name"] in concurrent_safe:
            safe_calls.append((idx, tc))
        else:
            seq_calls.append((idx, tc))

    # Pre-allocate results
    results: list[ToolResult | None] = [None] * len(tool_calls)

    # Run concurrent-safe tools in parallel
    if safe_calls:
        coros = [
            _execute_single_tool(tc, registry, context) for _, tc in safe_calls
        ]
        parallel_results = await asyncio.gather(*coros, return_exceptions=True)
        for (idx, _), result in zip(safe_calls, parallel_results):
            if isinstance(result, Exception):
                results[idx] = ToolResult(
                    data=json.dumps({"error": f"Tool execution failed: {result}"}),
                    is_error=True,
                )
            else:
                results[idx] = result

    # Run sequential tools one by one
    for idx, tc in seq_calls:
        try:
            results[idx] = await _execute_single_tool(tc, registry, context)
        except Exception as exc:
            logger.exception("Sequential tool %s failed", tc["tool_name"])
            results[idx] = ToolResult(
                data=json.dumps({"error": f"Tool execution failed: {exc}"}),
                is_error=True,
            )

    # Ensure no None slots (defensive)
    return [
        r if r is not None else ToolResult(data='{"error": "unknown execution failure"}', is_error=True)
        for r in results
    ]


async def _execute_single_tool(
    tool_call: dict,
    registry: ToolRegistry,
    context: ToolUseContext,
) -> ToolResult:
    """Execute a single tool call with input validation and error handling."""
    tool_name = tool_call["tool_name"]
    tool_input = tool_call.get("tool_input", {})
    tool_use_id = tool_call.get("tool_use_id", "unknown")

    tool = registry.get(tool_name)
    if tool is None:
        logger.warning("Tool '%s' not found in registry", tool_name)
        return ToolResult(
            data=json.dumps({
                "error": f"Tool '{tool_name}' is not available. "
                         f"Available tools: {[t.name for t in registry.all_tools()]}",
            }),
            is_error=True,
        )

    # Validate input
    validation = tool.validate_input(tool_input)
    if not validation.valid:
        logger.warning(
            "Input validation failed for %s: %s", tool_name, validation.message,
        )
        return ToolResult(
            data=json.dumps({
                "error": f"Invalid input for '{tool_name}': {validation.message}",
            }),
            is_error=True,
        )

    # Execute
    start_time = time.monotonic()
    try:
        result = await tool.call(tool_input, context)
    except Exception as exc:
        logger.exception("Tool '%s' raised an exception", tool_name)
        return ToolResult(
            data=json.dumps({"error": f"Tool '{tool_name}' failed: {exc}"}),
            is_error=True,
        )
    elapsed = time.monotonic() - start_time

    # Truncate oversized results
    if len(result.data) > tool.max_result_size_chars:
        original_len = len(result.data)
        result.data = result.data[: tool.max_result_size_chars]
        result.truncated = True
        logger.info(
            "Truncated result for %s: %d -> %d chars",
            tool_name, original_len, tool.max_result_size_chars,
        )

    logger.debug(
        "Tool %s completed in %.2fs (error=%s, truncated=%s)",
        tool_name, elapsed, result.is_error, result.truncated,
    )
    return result


# ---------------------------------------------------------------------------
# Final Answer (forced)
# ---------------------------------------------------------------------------

async def _final_answer(params: QueryParams, state: LoopState) -> str:
    """Force a text-only final answer when max turns is exceeded.

    Calls the LLM without tools, with a directive to summarize findings.
    """
    force_prompt = (
        "You have reached the maximum number of tool-use turns. "
        "You MUST now provide your final answer based on the work done so far. "
        "Summarize all findings and produce your output. Do NOT call any tools."
    )

    state.messages.append(Message(role="user", content=force_prompt))

    api_messages = [msg.to_api_dict() for msg in state.messages]

    text_parts: list[str] = []
    async for event in params.llm_client.stream(
        messages=api_messages,
        system_prompt=params.system_prompt,
        tools=None,  # No tools — force text
        max_tokens=params.max_tokens,
        temperature=params.temperature,
    ):
        if event.type == EventType.TEXT_DELTA:
            text_parts.append(event.data.get("text", ""))

    return "".join(text_parts)


# ---------------------------------------------------------------------------
# Stop Hooks
# ---------------------------------------------------------------------------

async def _run_stop_hooks(
    params: QueryParams,
    state: LoopState,
) -> str | None:
    """Run stop hooks (quality gates) and return feedback if the agent
    should continue, or None if the loop should end.

    The hook engine is an optional object with an ``on_stop`` method:
        async def on_stop(state: LoopState) -> str | None

    Returns None to allow the loop to end, or a feedback string to inject
    as a user message for another iteration.
    """
    if params.hook_engine is None:
        return None

    hook = getattr(params.hook_engine, "run_stop_hooks", None)
    if hook is None:
        return None

    try:
        result = await hook(state)
        if hasattr(result, 'should_continue') and result.should_continue:
            return result.feedback
        if isinstance(result, str):
            return result
        return None
    except Exception as exc:
        logger.warning("Stop hook failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------

def _condense_for_history(messages: list[Message]) -> list[Message]:
    """Strip tool-call and tool-result messages to reduce context size.

    Keeps:
    - The first user message (may contain images)
    - All assistant text-only messages
    - The last 4 tool exchanges (to preserve recent context)

    Replaces removed tool exchanges with a summary message.
    """
    if len(messages) <= 6:
        return messages  # Nothing to compact

    # Always keep the first message (may have images)
    first = messages[0]

    # Find all tool-exchange boundaries: assistant msg with tool_calls + tool result msgs
    # We'll keep the last N exchanges
    keep_last_n = 4

    # Separate into "text-only" and "tool-exchange" segments
    condensed: list[Message] = [first]
    tool_exchange_count = 0
    tool_exchanges: list[list[Message]] = []
    current_exchange: list[Message] = []
    text_messages: list[Message] = []

    for msg in messages[1:]:
        if msg.role == "assistant" and isinstance(msg.content, list):
            # This is an assistant message with tool calls
            if current_exchange:
                tool_exchanges.append(current_exchange)
            current_exchange = [msg]
            tool_exchange_count += 1
        elif msg.role == "tool":
            current_exchange.append(msg)
        else:
            if current_exchange:
                tool_exchanges.append(current_exchange)
                current_exchange = []
            text_messages.append(msg)

    if current_exchange:
        tool_exchanges.append(current_exchange)

    # Keep text messages
    for msg in text_messages:
        condensed.append(msg)

    # Summarize removed tool exchanges
    removed_count = max(0, len(tool_exchanges) - keep_last_n)
    if removed_count > 0:
        condensed.append(
            Message(
                role="user",
                content=(
                    f"[System: {removed_count} earlier tool-use exchanges have been "
                    f"compacted to save context. The results of those tool calls are "
                    f"reflected in the assistant's text responses above.]"
                ),
            )
        )

    # Keep the last N tool exchanges
    for exchange in tool_exchanges[-keep_last_n:]:
        condensed.extend(exchange)

    return condensed


def _estimate_tokens(messages: list[Message]) -> int:
    """Rough token count estimate: ~4 characters per token."""
    total_chars = 0
    for msg in messages:
        if isinstance(msg.content, str):
            total_chars += len(msg.content)
        else:
            for block in msg.content:
                if block.text:
                    total_chars += len(block.text)
                if block.content:
                    total_chars += len(block.content)
                if block.tool_input:
                    total_chars += len(json.dumps(block.tool_input))
                # Don't count image data in token estimate
    return total_chars // 4


import re

_UNPARSED_TOOL_PATTERNS = [
    re.compile(r"<tool_call>", re.IGNORECASE),
    re.compile(r"<function=\w+>", re.IGNORECASE),
    re.compile(r"```tool_code", re.IGNORECASE),
]


def _contains_unparsed_tool_call(text: str) -> bool:
    """Detect raw/unparsed tool call patterns that sglang failed to parse."""
    return any(pat.search(text) for pat in _UNPARSED_TOOL_PATTERNS)
