"""
Visual tool registry builder for RadClaw.

Imports all eight visual analysis tools and registers them in a
ToolRegistry for use by the visual analysis agent.
"""

from __future__ import annotations

from src.core.tool import ToolRegistry

from src.tools.visual.select_slice import SelectSliceTool
from src.tools.visual.crop_roi import CropROITool
from src.tools.visual.zoom_enhance import ZoomEnhanceTool
from src.tools.visual.multi_window import MultiWindowTool
from src.tools.visual.compose_panel import ComposePanelTool
from src.tools.visual.overlay_annotation import OverlayAnnotationTool
from src.tools.visual.generate_probe import GenerateProbeTool
from src.tools.visual.link_evidence import LinkEvidenceTool


def build_visual_registry() -> ToolRegistry:
    """Create a ToolRegistry populated with all visual analysis tools."""
    registry = ToolRegistry()

    registry.register(SelectSliceTool())
    registry.register(CropROITool())
    registry.register(ZoomEnhanceTool())
    registry.register(MultiWindowTool())
    registry.register(ComposePanelTool())
    registry.register(OverlayAnnotationTool())
    registry.register(GenerateProbeTool())
    registry.register(LinkEvidenceTool())

    return registry