"""
Report Writing agent builder.

Imports all 5 report tools from src/tools/report/ and builds
the agent configuration for the report writing stage.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.agents.base import AgentConfig, _get_agent_settings, _build_system_prompt
from src.core.types import AgentRole

if TYPE_CHECKING:
    from src.core.types import PipelineState

logger = logging.getLogger(__name__)


def build_report_writing_agent(
    settings: dict,
    pipeline_state: PipelineState,
) -> AgentConfig:
    """Create the report writing agent configuration.

    Tools registered (5):
        1. build_structured_finding     — Build a structured finding
        2. build_structured_impression  — Build a structured impression
        3. link_support_evidence        — Link support evidence to findings
        4. assemble_report_object       — Assemble the final report
        5. render_natural_language      — Render report as natural language

    Parameters
    ----------
    settings : dict
        Master settings dict.
    pipeline_state : PipelineState
        Shared pipeline state.

    Returns
    -------
    AgentConfig
    """
    from src.tools.report import build_report_registry

    registry = build_report_registry()

    system_prompt = _build_system_prompt(
        AgentRole.REPORT_WRITING,
        registry.all_tools(),
        pipeline_state,
    )

    agent_settings = _get_agent_settings(settings, AgentRole.REPORT_WRITING)

    return AgentConfig(
        role=AgentRole.REPORT_WRITING,
        tool_registry=registry,
        system_prompt=system_prompt,
        max_turns=int(agent_settings.get("max_turns", 10)),
        temperature=float(agent_settings.get("temperature", 0.7)),
        max_tokens=int(agent_settings.get("max_tokens", 8192)),
    )