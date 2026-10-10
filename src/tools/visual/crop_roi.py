"""
CropROITool — Crop a region of interest from the original image.

Extracts a rectangular sub-image from the source image for
focused analysis.  Coordinates are in pixels with bounds checking.
The cropped image is saved to the SessionImageStore for downstream
reference.
"""

from __future__ import annotations

import json
import logging

from src.core.tool import Tool, ToolUseContext
from src.core.types import PipelineState, ToolResult
from src.utils.image_processing import crop_region

logger = logging.getLogger(__name__)


class CropROITool(Tool):
    name = "crop_roi"
    description = (
        "Crop a region of interest from the original image for detailed analysis. "
        "Provide pixel coordinates and dimensions. The cropped image is saved to "
        "the image store and can be referenced by its returned image_id."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "x": {
                "type": "integer",
                "description": "X coordinate of the top-left corner (pixels).",
            },
            "y": {
                "type": "integer",
                "description": "Y coordinate of the top-left corner (pixels).",
            },
            "width": {
                "type": "integer",
                "description": "Width of the crop region (pixels).",
                "minimum": 1,
            },
            "height": {
                "type": "integer",
                "description": "Height of the crop region (pixels).",
                "minimum": 1,
            },
            "description": {
                "type": "string",
                "description": (
                    "A brief description of what this ROI contains "
                    "(e.g. 'upper left region', 'central structure')."
                ),
            },
        },
        "required": ["x", "y", "width", "height", "description"],
    }
    is_concurrency_safe = True
    is_read_only = False  # writes to image store

    def prompt(self) -> str:
        return (
            "Use crop_roi when you want to isolate a specific anatomical region for "
            "closer inspection. Provide pixel coordinates for the bounding box and a "
            "short description of the region. The tool returns an image_id that you can "
            "pass to zoom_and_enhance, overlay_annotation, or generate_visual_probe."
        )

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        # Validate args
        try:
            x = int(args["x"])
            y = int(args["y"])
            w = int(args["width"])
            h = int(args["height"])
        except (KeyError, TypeError, ValueError) as exc:
            return ToolResult(
                data=json.dumps({"error": f"Invalid coordinates: {exc}"}),
                is_error=True,
            )

        desc: str = args.get("description", "ROI crop")

        if w <= 0 or h <= 0:
            return ToolResult(
                data=json.dumps({"error": "width and height must be positive integers"}),
                is_error=True,
            )

        pipeline_state: PipelineState = context.extra.get("pipeline_state")
        image_store = context.extra.get("image_store")

        if pipeline_state is None or image_store is None:
            return ToolResult(
                data=json.dumps({"error": "pipeline_state or image_store not found in context"}),
                is_error=True,
            )

        original = pipeline_state.original_image
        if original is None:
            return ToolResult(
                data=json.dumps({"error": "No original image available in pipeline state"}),
                is_error=True,
            )

        # Perform the crop
        try:
            cropped = crop_region(original, x, y, w, h)
        except Exception as exc:
            logger.exception("crop_region failed")
            return ToolResult(
                data=json.dumps({"error": f"Crop failed: {exc}"}),
                is_error=True,
            )

        # Save to store
        image_id = image_store.save(
            cropped, desc, tool_name=self.name,
            metadata={"crop_box": {"x": x, "y": y, "width": w, "height": h}},
        )

        result = {
            "image_id": image_id,
            "description": desc,
            "crop_box": {"x": x, "y": y, "width": w, "height": h},
            "original_size": {"width": original.width, "height": original.height},
            "cropped_size": {"width": cropped.width, "height": cropped.height},
            "message": (
                f"Cropped region ({x},{y},{w},{h}) saved as {image_id}. "
                f"Result size: {cropped.width}x{cropped.height}."
            ),
        }

        logger.info("Cropped ROI '%s' at (%d,%d,%d,%d) -> %s", desc, x, y, w, h, image_id)
        return ToolResult(
            data=json.dumps(result),
            metadata={"image_id": image_id},
        )