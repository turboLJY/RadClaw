"""
SelectSliceTool — Select and confirm the imaging view type for analysis.

This is typically the first tool invoked in a visual analysis pipeline.
It validates the view type (PA, AP, lateral) and records the exam
information in the PipelineState so downstream tools know what kind
of image they are working with.
"""

from __future__ import annotations

import json
import logging

from src.core.tool import Tool, ToolUseContext
from src.core.types import PipelineState, ToolResult

logger = logging.getLogger(__name__)

_VALID_VIEW_TYPES = {
    "frontal_PA",
    "frontal_AP",
    "lateral",
    "lateral_decubitus",
    "lordotic",
    "oblique",
}


class SelectSliceTool(Tool):
    name = "select_slice_or_view"
    description = (
        "Select and confirm the imaging view type for analysis. "
        "This should be called first to establish the projection geometry "
        "(e.g. frontal_PA, frontal_AP, lateral) before any region-based analysis."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "view_type": {
                "type": "string",
                "description": (
                    "The imaging view/projection type. Common values: "
                    "'frontal_PA', 'frontal_AP', 'lateral', 'lateral_decubitus', "
                    "'lordotic', 'oblique'."
                ),
                "enum": list(_VALID_VIEW_TYPES),
            },
        },
        "required": ["view_type"],
    }
    is_concurrency_safe = True
    is_read_only = False  # modifies PipelineState

    def prompt(self) -> str:
        return (
            "Use select_slice_or_view as the FIRST step of any visual analysis. "
            "Identify the imaging view or projection type and call this tool "
            "to lock in the geometry so that subsequent spatial reasoning is correct."
        )

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        view_type: str = args.get("view_type", "").strip()

        if view_type not in _VALID_VIEW_TYPES:
            return ToolResult(
                data=json.dumps({
                    "error": f"Invalid view_type '{view_type}'. Must be one of: {sorted(_VALID_VIEW_TYPES)}",
                }),
                is_error=True,
            )

        pipeline_state: PipelineState = context.extra.get("pipeline_state")
        if pipeline_state is None:
            return ToolResult(
                data=json.dumps({"error": "pipeline_state not found in context"}),
                is_error=True,
            )

        # Record view info
        pipeline_state.exam_info["view"] = view_type
        pipeline_state.exam_info["study_type"] = "Chest X-Ray"

        # Determine image dimensions for reference
        original = pipeline_state.original_image
        img_size = original.size if original is not None else (0, 0)

        result = {
            "status": "confirmed",
            "view_type": view_type,
            "image_width": img_size[0],
            "image_height": img_size[1],
            "message": (
                f"View confirmed as {view_type}. "
                f"Image dimensions: {img_size[0]}x{img_size[1]} pixels. "
                "You may now proceed with region-based analysis."
            ),
        }

        logger.info("View selected: %s (%dx%d)", view_type, img_size[0], img_size[1])
        return ToolResult(
            data=json.dumps(result),
            metadata={"view_type": view_type},
        )