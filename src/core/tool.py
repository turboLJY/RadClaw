"""
Tool base class and registry for RadClaw.
Adapted from SearchClaw with medical imaging extensions.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.types import ToolResult, ValidationResult

logger = logging.getLogger(__name__)

CACHE_DIR = Path("./cache")


@dataclass
class ToolUseContext:
    session_id: str = ""
    turn_count: int = 0
    cache_dir: Path = field(default_factory=lambda: CACHE_DIR)
    rate_limiter: Any = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)


class Tool(ABC):
    name: str = ""
    description: str = ""
    input_schema: dict

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "input_schema" not in cls.__dict__:
            cls.input_schema = {}

    is_concurrency_safe: bool = False
    is_read_only: bool = True
    max_result_size_chars: int = 30000

    @abstractmethod
    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        ...

    def prompt(self) -> str:
        return ""

    def validate_input(self, args: dict) -> ValidationResult:
        return ValidationResult(valid=True)

    def to_api_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            logger.warning(f"Tool '{tool.name}' already registered, overwriting")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def get_api_schemas(self) -> list[dict]:
        return [tool.to_api_schema() for tool in self._tools.values()]

    def get_concurrent_safe(self) -> set[str]:
        return {name for name, tool in self._tools.items() if tool.is_concurrency_safe}