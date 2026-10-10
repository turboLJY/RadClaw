"""
Hook execution engine for RadClaw.
Quality gates that can force additional agent iterations.
Adapted from SearchClaw.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.core.types import LoopState

logger = logging.getLogger(__name__)


@dataclass
class HookEvaluation:
    passed: bool
    feedback: str = ""


@dataclass
class HookResult:
    should_continue: bool = False
    feedback: str | None = None


class Hook(ABC):
    name: str = ""
    description: str = ""

    @abstractmethod
    async def evaluate(self, state: LoopState, **kwargs) -> HookEvaluation:
        ...


class HookEngine:
    def __init__(self):
        self.stop_hooks: list[Hook] = []

    def register_stop_hook(self, hook: Hook) -> None:
        self.stop_hooks.append(hook)
        logger.info(f"Registered stop hook: {hook.name}")

    async def run_stop_hooks(self, state: LoopState) -> HookResult:
        result = HookResult()
        for hook in self.stop_hooks:
            try:
                evaluation = await hook.evaluate(state)
                if not evaluation.passed:
                    logger.info(f"Stop hook '{hook.name}' failed: {evaluation.feedback}")
                    result.should_continue = True
                    result.feedback = evaluation.feedback
                    return result
            except Exception as e:
                logger.warning(f"Stop hook '{hook.name}' error: {e}")
        return result