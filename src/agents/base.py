"""
Agent configuration and factory for RadClaw.

An AgentConfig is a frozen configuration bundle that pairs a role-specific
system prompt with the tool registry and generation parameters needed to
drive the agentic loop (``src.core.loop``).  Agents themselves contain no
execution logic — the loop handles all LLM interaction and tool dispatch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.core.context import ContextBuilder
from src.core.types import AgentRole

if TYPE_CHECKING:
    from src.core.tool import ToolRegistry
    from src.core.types import PipelineState
    from src.hooks.engine import HookEngine

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Everything the agentic loop needs to run one agent."""

    role: AgentRole
    tool_registry: ToolRegistry
    system_prompt: str
    max_turns: int = 15
    hook_engine: HookEngine | None = None
    temperature: float = 1.0
    max_tokens: int = 8192


def build_agent(
    role: AgentRole,
    settings: dict,
    pipeline_state: PipelineState,
) -> AgentConfig:
    """Factory that creates the appropriate AgentConfig for *role*.

    Delegates to the role-specific builder in
    ``src.agents.{role_name}`` which knows how to assemble the tool
    registry and agent-specific overrides.

    Parameters
    ----------
    role:
        Which agent to build.
    settings:
        Parsed settings dict (typically from ``config/settings.yaml``).
        Agent-specific settings live under ``settings["agents"][role.value]``.
    pipeline_state:
        Shared pipeline state — used to inject upstream context into the
        system prompt.

    Returns
    -------
    AgentConfig
        Ready to be passed to ``query_loop``.
    """
    from src.agents.visual_analysis import build_visual_analysis_agent
    from src.agents.evidence_retrieval import build_evidence_retrieval_agent
    from src.agents.report_writing import build_report_writing_agent
    from src.agents.quality_review import build_quality_review_agent

    builders = {
        AgentRole.VISUAL_ANALYSIS: build_visual_analysis_agent,
        AgentRole.EVIDENCE_RETRIEVAL: build_evidence_retrieval_agent,
        AgentRole.REPORT_WRITING: build_report_writing_agent,
        AgentRole.QUALITY_REVIEW: build_quality_review_agent,
    }

    builder_fn = builders.get(role)
    if builder_fn is None:
        raise ValueError(f"No builder registered for role: {role}")

    config = builder_fn(settings, pipeline_state)
    logger.info(
        "Built agent %s: %d tools, max_turns=%d, temp=%.2f",
        role.value,
        len(config.tool_registry.all_tools()),
        config.max_turns,
        config.temperature,
    )
    return config


def _get_agent_settings(settings: dict, role: AgentRole) -> dict:
    """Extract agent-specific settings from the master settings dict.

    Falls back to sensible defaults if the settings section is missing.
    """
    agents_block = settings.get("agents", {})
    return agents_block.get(role.value, {})


def _build_system_prompt(
    role: AgentRole,
    tools: list,
    pipeline_state: PipelineState,
) -> str:
    """Convenience wrapper around ContextBuilder."""
    builder = ContextBuilder()
    return builder.build_agent_prompt(role, tools, pipeline_state)