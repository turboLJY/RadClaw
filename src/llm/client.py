"""
LLM client for RadClaw using vLLM's OpenAI-compatible API.
Replaces SearchClaw's litellm wrapper with direct openai SDK calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator

import openai

from src.core.types import EventType, StreamEvent

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen3.5-35B-A3B"
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    max_tokens: int = 8192
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    presence_penalty: float = 1.5
    enable_thinking: bool = True
    max_retries: int = 3
    retry_base_delay_ms: int = 500

    @classmethod
    def from_settings(cls, settings_path: str | Path = "config/settings.yaml") -> ModelConfig:
        path = Path(settings_path)
        if not path.exists():
            return cls()
        try:
            import yaml
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            m = data.get("model", {})
            return cls(
                model_name=m.get("path", cls.model_name),
                base_url=m.get("base_url", cls.base_url),
                api_key=m.get("api_key", cls.api_key),
                max_tokens=int(m.get("max_tokens", cls.max_tokens)),
                temperature=float(m.get("temperature", cls.temperature)),
                top_p=float(m.get("top_p", cls.top_p)),
                top_k=int(m.get("top_k", cls.top_k)),
                presence_penalty=float(m.get("presence_penalty", cls.presence_penalty)),
                enable_thinking=m.get("enable_thinking", cls.enable_thinking),
                max_retries=int(m.get("max_retries", cls.max_retries)),
                retry_base_delay_ms=int(m.get("retry_base_delay_ms", cls.retry_base_delay_ms)),
            )
        except Exception as e:
            logger.warning(f"Failed to load settings: {e}, using defaults")
            return cls()


def _is_retryable(error: Exception) -> bool:
    error_str = str(error).lower()
    for pattern in ("datainspectionfailed", "ratelimit", "429", "overloaded", "529", "500", "timeout", "408", "connection"):
        if pattern in error_str or pattern in type(error).__name__.lower():
            return True
    return False


class LLMClient:
    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.client = openai.AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
        )

    async def stream(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        target_max_tokens = max_tokens or self.config.max_tokens
        target_temp = temperature if temperature is not None else self.config.temperature

        api_messages = [{"role": "system", "content": system_prompt}] + messages

        completion_kwargs: dict = {
            "model": self.config.model_name,
            "messages": api_messages,
            "max_completion_tokens": target_max_tokens,
            "temperature": target_temp,
            "top_p": self.config.top_p,
            "stream": True,
        }
        if tools:
            completion_kwargs["tools"] = tools

        logger.debug(
            "LLM request: %d messages, %d tools, max_tokens=%d",
            len(api_messages),
            len(tools) if tools else 0,
            target_max_tokens,
        )
        for i, m in enumerate(api_messages):
            role = m.get("role", "?")
            content = m.get("content", "")
            if isinstance(content, list):
                types = [c.get("type", "?") for c in content]
                logger.debug("  msg[%d] role=%s content_blocks=%s", i, role, types)
            else:
                logger.debug("  msg[%d] role=%s content_len=%d", i, role, len(str(content)))

        last_error: Exception | None = None

        for attempt in range(1, self.config.max_retries + 2):
            try:
                response = await self.client.chat.completions.create(**completion_kwargs)

                current_tool_call: dict | None = None
                tool_call_args_buffer = ""
                accumulated_text = ""
                tool_calls_collected = []

                async for chunk in response:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta is None:
                        continue

                    if delta.content:
                        accumulated_text += delta.content
                        yield StreamEvent(
                            type=EventType.TEXT_DELTA,
                            data={"text": delta.content},
                        )

                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            if tc.function and tc.function.name:
                                if current_tool_call and tool_call_args_buffer:
                                    try:
                                        parsed_args = json.loads(tool_call_args_buffer)
                                    except json.JSONDecodeError:
                                        parsed_args = {"raw": tool_call_args_buffer}
                                    yield StreamEvent(
                                        type=EventType.TOOL_USE,
                                        data={
                                            "tool_use_id": current_tool_call["id"],
                                            "tool_name": current_tool_call["name"],
                                            "tool_input": parsed_args,
                                        },
                                    )
                                    tool_calls_collected.append(current_tool_call["name"])
                                current_tool_call = {
                                    "id": tc.id or f"call_{id(tc)}",
                                    "name": tc.function.name,
                                }
                                tool_call_args_buffer = tc.function.arguments or ""
                            elif tc.function and tc.function.arguments:
                                tool_call_args_buffer += tc.function.arguments

                if current_tool_call:
                    try:
                        parsed_args = json.loads(tool_call_args_buffer) if tool_call_args_buffer else {}
                    except json.JSONDecodeError:
                        parsed_args = {"raw": tool_call_args_buffer}
                    tool_calls_collected.append(current_tool_call["name"])
                    yield StreamEvent(
                        type=EventType.TOOL_USE,
                        data={
                            "tool_use_id": current_tool_call["id"],
                            "tool_name": current_tool_call["name"],
                            "tool_input": parsed_args,
                        },
                    )

                logger.info(
                    "LLM response: text_len=%d, tool_calls=%d, text_preview=%.200s",
                    len(accumulated_text),
                    len(tool_calls_collected),
                    accumulated_text[:200] if accumulated_text else "(none)",
                )
                return

            except Exception as e:
                last_error = e
                logger.error(f"LLM call failed (attempt {attempt}): {e}")

                if not _is_retryable(e) or attempt > self.config.max_retries:
                    break

                delay = (self.config.retry_base_delay_ms / 1000) * (2 ** (attempt - 1))
                yield StreamEvent(
                    type=EventType.STATUS,
                    data={"message": f"API error, retrying in {delay:.0f}s..."},
                )
                await asyncio.sleep(delay)

        if last_error:
            yield StreamEvent(
                type=EventType.ERROR,
                data={"message": f"LLM call failed: {str(last_error)}"},
            )

    async def side_query(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 2048,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=0.7,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"Side query failed: {e}")
            return ""


_shared_client: LLMClient | None = None


def set_shared_client(client: LLMClient) -> None:
    global _shared_client
    _shared_client = client


def get_shared_client() -> LLMClient:
    global _shared_client
    if _shared_client is None:
        _shared_client = LLMClient()
    return _shared_client
