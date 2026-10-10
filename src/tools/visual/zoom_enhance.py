"""
ZoomEnhanceTool — Zoom into a region and apply image enhancement.

Retrieves a previously stored image from the SessionImageStore, applies
a zoom + enhancement pipeline, and saves the result back to the store.
"""

from __future__ import annotations

import json
import logging

from src.core.tool import Tool, ToolUseContext
from src.core.types import ToolResult
from src.utils.image_processing import enhance_image, zoom_image

logger = logging.getLogger(__name__)

_VALID_ENHANCE_TYPES = {"sharpen", "edge", "contrast"}


class ZoomEnhanceTool(Tool):
    name = "zoom_and_enhance"
    description = (
        "Zoom into a specific region and apply enhancement for detailed observation. "
        "Takes an existing image_id, applies a zoom factor, then enhances with one of: "
        "sharpen, edge detection, or contrast boost."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "image_id": {
                "type": "string",
                "description": "ID of the source image in the image store.",
            },
            "zoom_factor": {
                "type": "number",
                "description": "Magnification factor (default 2.0). Range [1.0, 8.0].",
                "default": 2.0,
                "minimum": 1.0,
                "maximum": 8.0,
            },
            "enhance_type": {
                "type": "string",
                "description": "Enhancement to apply: 'sharpen', 'edge', or 'contrast'.",
                "enum": list(_VALID_ENHANCE_TYPES),
                "default": "sharpen",
            },
        },
        "required": ["image_id"],
    }
    is_concurrency_safe = True
    is_read_only = False

    def prompt(self) -> str:
        return (
            "Use zoom_and_enhance after crop_roi to magnify a small area and apply "
            "image processing that makes subtle findings easier to see. Choose "
            "'sharpen' for fine detail, 'edge' for boundary detection, or 'contrast' "
            "for density differences."
        )

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        image_id: str = args.get("image_id", "")
        zoom_factor: float = float(args.get("zoom_factor", 2.0))
        enhance_type: str = args.get("enhance_type", "sharpen").strip().lower()

        image_store = context.extra.get("image_store")
        if image_store is None:
            return ToolResult(
                data=json.dumps({"error": "image_store not found in context"}),
                is_error=True,
            )

        entry = image_store.get(image_id)
        if entry is None:
            return ToolResult(
                data=json.dumps({"error": f"Image '{image_id}' not found in store"}),
                is_error=True,
            )

        src_img, src_desc = entry

        # Clamp zoom factor
        zoom_factor = max(1.0, min(zoom_factor, 8.0))

        if enhance_type not in _VALID_ENHANCE_TYPES:
            return ToolResult(
                data=json.dumps({
                    "error": (
                        f"Invalid enhance_type '{enhance_type}'. "
                        f"Must be one of: {sorted(_VALID_ENHANCE_TYPES)}"
                    ),
                }),
                is_error=True,
            )

        try:
            # Zoom the entire source image (full region)
            zoomed = zoom_image(src_img, 0, 0, src_img.width, src_img.height, zoom_factor)
            enhanced = enhance_image(zoomed, enhance_type)
        except Exception as exc:
            logger.exception("zoom_and_enhance processing failed")
            return ToolResult(
                data=json.dumps({"error": f"Processing failed: {exc}"}),
                is_error=True,
            )

        desc = f"{src_desc} [zoom={zoom_factor}x, {enhance_type}]"
        new_id = image_store.save(enhanced, desc, tool_name=self.name)
        image_store.add_tool_to_chain(new_id, self.name)

        result = {
            "image_id": new_id,
            "source_image_id": image_id,
            "zoom_factor": zoom_factor,
            "enhance_type": enhance_type,
            "output_size": {"width": enhanced.width, "height": enhanced.height},
            "description": desc,
            "message": (
                f"Applied {zoom_factor}x zoom + {enhance_type} enhancement. "
                f"Result saved as {new_id} ({enhanced.width}x{enhanced.height})."
            ),
        }

        logger.info(
            "Zoom+enhance: %s -> %s (zoom=%.1f, enhance=%s)",
            image_id, new_id, zoom_factor, enhance_type,
        )
        return ToolResult(
            data=json.dumps(result),
            metadata={"image_id": new_id},
        )