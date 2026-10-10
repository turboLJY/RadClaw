"""
ComposePanelTool — Compose multiple images into a comparison panel.

Retrieves images from the SessionImageStore and stitches them into a
single side-by-side (horizontal) or grid layout with optional labels.
"""

from __future__ import annotations

import json
import logging

from src.core.tool import Tool, ToolUseContext
from src.core.types import ToolResult
from src.utils.image_processing import compose_grid

logger = logging.getLogger(__name__)


class ComposePanelTool(Tool):
    name = "compose_comparison_panel"
    description = (
        "Compose multiple images into a side-by-side comparison panel. "
        "Provide a list of image_ids and optional labels; the tool stitches "
        "them into a single composite image for visual comparison."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "image_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of image IDs from the image store to compose.",
                "minItems": 1,
                "maxItems": 16,
            },
            "layout": {
                "type": "string",
                "description": "'horizontal' for a single row, 'grid' for a square-ish arrangement.",
                "enum": ["horizontal", "grid"],
                "default": "horizontal",
            },
            "labels": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Labels to display above each image. Must match the number of image_ids.",
            },
        },
        "required": ["image_ids"],
    }
    is_concurrency_safe = True
    is_read_only = False

    def prompt(self) -> str:
        return (
            "Use compose_comparison_panel to put multiple images side by side for "
            "direct visual comparison. This is useful after multi_window_render to "
            "compare different windowing, or to compare a cropped ROI with an "
            "annotated version."
        )

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        image_ids: list[str] = args.get("image_ids", [])
        layout: str = args.get("layout", "horizontal").strip().lower()
        labels: list[str] = args.get("labels", [])

        if not image_ids:
            return ToolResult(
                data=json.dumps({"error": "image_ids must be a non-empty list"}),
                is_error=True,
            )

        if layout not in ("horizontal", "grid"):
            layout = "horizontal"

        image_store = context.extra.get("image_store")
        if image_store is None:
            return ToolResult(
                data=json.dumps({"error": "image_store not found in context"}),
                is_error=True,
            )

        # Retrieve all images
        images = []
        descriptions = []
        missing = []

        for img_id in image_ids:
            entry = image_store.get(img_id)
            if entry is None:
                missing.append(img_id)
            else:
                images.append(entry[0])
                descriptions.append(entry[1])

        if missing:
            return ToolResult(
                data=json.dumps({
                    "error": f"Image(s) not found in store: {missing}",
                }),
                is_error=True,
            )

        # Use provided labels or fall back to image_ids
        if not labels or len(labels) != len(images):
            labels = image_ids

        try:
            composite = compose_grid(images, labels=labels, layout=layout)
        except Exception as exc:
            logger.exception("compose_grid failed")
            return ToolResult(
                data=json.dumps({"error": f"Composition failed: {exc}"}),
                is_error=True,
            )

        desc = f"Comparison panel ({layout}): {', '.join(image_ids)}"
        new_id = image_store.save(composite, desc, tool_name=self.name)

        result = {
            "image_id": new_id,
            "layout": layout,
            "source_image_ids": image_ids,
            "labels": labels,
            "output_size": {"width": composite.width, "height": composite.height},
            "description": desc,
            "message": (
                f"Composed {len(images)} images into {layout} panel. "
                f"Result saved as {new_id} ({composite.width}x{composite.height})."
            ),
        }

        logger.info("Composed panel %s from %s", new_id, image_ids)
        return ToolResult(
            data=json.dumps(result),
            metadata={"image_id": new_id},
        )