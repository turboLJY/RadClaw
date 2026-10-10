"""
Quality Review agent builder.

Imports all 6 review tools from src/tools/review/ and builds
the agent configuration for the quality review stage.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.agents.base import AgentConfig, _get_agent_settings, _build_system_prompt
from src.core.types import AgentRole

if TYPE_CHECKING:
    from src.core.types import PipelineState

logger = logging.getLogger(__name__)


def build_quality_review_agent(
    settings: dict,
    pipeline_state: PipelineState,
) -> AgentConfig:
    """Create the quality review agent configuration.

    Tools registered (6):
        1. check_visual_consistency   — Verify findings match visual evidence
        2. check_evidence_consistency — Verify evidence supports findings
        3. check_internal_consistency — Check report self-consistency
        4. check_medical_standard     — Check medical reporting standards
        5. attribute_responsibility   — Attribute issues to pipeline stages
        6. inject_human_feedback      — Accept human reviewer feedback

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
    from src.tools.review import build_review_registry

    registry = build_review_registry()

    system_prompt = _build_system_prompt(
        AgentRole.QUALITY_REVIEW,
        registry.all_tools(),
        pipeline_state,
    )

    agent_settings = _get_agent_settings(settings, AgentRole.QUALITY_REVIEW)

    return AgentConfig(
        role=AgentRole.QUALITY_REVIEW,
        tool_registry=registry,
        system_prompt=system_prompt,
        max_turns=int(agent_settings.get("max_turns", 10)),
        temperature=float(agent_settings.get("temperature", 0.7)),
        max_tokens=int(agent_settings.get("max_tokens", 4096)),
    )