"""
Evidence Retrieval agent builder.

Imports all 7 evidence tools from src/tools/evidence/ and builds
the agent configuration for the evidence retrieval stage.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.agents.base import AgentConfig, _get_agent_settings, _build_system_prompt
from src.core.types import AgentRole

if TYPE_CHECKING:
    from src.core.types import PipelineState

logger = logging.getLogger(__name__)


def build_evidence_retrieval_agent(
    settings: dict,
    pipeline_state: PipelineState,
) -> AgentConfig:
    """Create the evidence retrieval agent configuration.

    Tools registered (7):
        1. standardize_observation  — Normalize findings to RadLex/SNOMED
        2. concept_lookup           — Search medical concept KB
        3. literature_search        — Search medical literature KB
        4. case_retrieval           — Search case database
        5. evidence_score_and_rank  — Score and rank evidence
        6. conflict_detection       — Cross-check for contradictions
        7. evidence_summary         — Produce consolidated summary

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
    from src.tools.evidence import build_evidence_registry

    knowledge_dir = settings.get("data", {}).get(
        "knowledge_dir", "./data/knowledge"
    )
    indexes_dir = settings.get("data", {}).get(
        "indexes_dir", "./data/indexes"
    )
    registry = build_evidence_registry(
        knowledge_dir=knowledge_dir,
        indexes_dir=indexes_dir,
        settings=settings,
    )

    system_prompt = _build_system_prompt(
        AgentRole.EVIDENCE_RETRIEVAL,
        registry.all_tools(),
        pipeline_state,
    )

    agent_settings = _get_agent_settings(settings, AgentRole.EVIDENCE_RETRIEVAL)

    return AgentConfig(
        role=AgentRole.EVIDENCE_RETRIEVAL,
        tool_registry=registry,
        system_prompt=system_prompt,
        max_turns=int(agent_settings.get("max_turns", 12)),
        temperature=float(agent_settings.get("temperature", 0.7)),
        max_tokens=int(agent_settings.get("max_tokens", 4096)),
    )