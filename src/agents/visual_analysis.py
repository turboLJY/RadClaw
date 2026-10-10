"""
Visual Analysis agent builder.

Registers all 8 visual tools and builds the agent configuration for
the visual analysis stage of the RadClaw pipeline.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.agents.base import AgentConfig, _get_agent_settings, _build_system_prompt
from src.core.types import AgentRole
from src.hooks.engine import HookEngine
from src.hooks.radiology_hooks import build_visual_hooks
from src.tools.visual import build_visual_registry

if TYPE_CHECKING:
    from src.core.types import PipelineState

logger = logging.getLogger(__name__)


def build_visual_analysis_agent(
    settings: dict,
    pipeline_state: PipelineState,
) -> AgentConfig:
    """Create the visual analysis agent configuration.

    Tools registered (8):
        1. select_slice_or_view  — Confirm imaging view type
        2. crop_roi              — Crop a region of interest
        3. zoom_and_enhance      — Enhance a cropped region
        4. multi_window_display  — Apply different window/level settings
        5. compose_panel         — Compose multiple images into a panel
        6. overlay_annotation    — Annotate images with markers
        7. generate_visual_probe — Generate analytical probe images
        8. link_visual_evidence  — Link findings to image evidence

    Parameters
    ----------
    settings : dict
        Master settings dict from config/settings.yaml.
    pipeline_state : PipelineState
        Shared pipeline state for prompt context injection.

    Returns
    -------
    AgentConfig
    """
    # Build the tool registry with all 8 visual tools
    registry = build_visual_registry()

    # Build the system prompt with tool instructions and pipeline context
    system_prompt = _build_system_prompt(
        AgentRole.VISUAL_ANALYSIS,
        registry.all_tools(),
        pipeline_state,
    )

    # Extract agent-specific settings
    agent_settings = _get_agent_settings(settings, AgentRole.VISUAL_ANALYSIS)

    # Build hook engine with quality gates
    hook_engine = HookEngine()
    hook_config = settings.get("hooks", {})
    for hook in build_visual_hooks(hook_config):
        hook_engine.register_stop_hook(hook)

    return AgentConfig(
        role=AgentRole.VISUAL_ANALYSIS,
        tool_registry=registry,
        system_prompt=system_prompt,
        max_turns=int(agent_settings.get("max_turns", 15)),
        hook_engine=hook_engine,
        temperature=float(agent_settings.get("temperature", 1.0)),
        max_tokens=int(agent_settings.get("max_tokens", 8192)),
    )