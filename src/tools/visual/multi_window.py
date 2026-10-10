"""
MultiWindowTool — Render the image with different display windows.

Applies radiological windowing presets (bone, lung, soft_tissue,
mediastinal) to highlight different tissue types.  Each windowed
rendering is saved independently to the image store.
"""

from __future__ import annotations

import json
import logging

from src.core.tool import Tool, ToolUseContext
from src.core.types import PipelineState, ToolResult
from src.utils.dicom_utils import WINDOW_FUNCTIONS, apply_window_by_name

logger = logging.getLogger(__name__)

_VALID_WINDOW_TYPES = list(WINDOW_FUNCTIONS.keys())


class MultiWindowTool(Tool):
    name = "multi_window_render"
    description = (
        "Render the image with different display windows to highlight different "
        "tissue types. Available windows: bone, lung, soft_tissue, mediastinal. "
        "Each rendering is saved separately to the image store."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "window_types": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": _VALID_WINDOW_TYPES,
                },
                "description": (
                    "List of window presets to apply. "
                    "Valid values: 'bone', 'lung', 'soft_tissue', 'mediastinal'."
                ),
                "minItems": 1,
                "maxItems": 4,
            },
        },
        "required": ["window_types"],
    }
    is_concurrency_safe = True
    is_read_only = False

    def prompt(self) -> str:
        return (
            "Use multi_window_render when you need to examine different tissue densities. "
            "'bone' highlights skeletal structures and calcifications; 'lung' brings out "
            "pulmonary parenchyma detail; 'soft_tissue' improves soft tissue boundaries; "
            "'mediastinal' reveals mediastinal structures. Request multiple windows at "
            "once for comparison."
        )

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        window_types: list[str] = args.get("window_types", [])

        if not window_types:
            return ToolResult(
                data=json.dumps({"error": "window_types must be a non-empty list"}),
                is_error=True,
            )

        # Validate
        invalid = [w for w in window_types if w.strip().lower() not in WINDOW_FUNCTIONS]
        if invalid:
            return ToolResult(
                data=json.dumps({
                    "error": (
                        f"Invalid window types: {invalid}. "
                        f"Valid options: {_VALID_WINDOW_TYPES}"
                    ),
                }),
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

        renderings: list[dict] = []

        for wt in window_types:
            wt = wt.strip().lower()
            try:
                windowed = apply_window_by_name(original, wt)
            except Exception as exc:
                logger.exception("Window '%s' failed", wt)
                renderings.append({
                    "window_type": wt,
                    "error": str(exc),
                })
                continue

            desc = f"Original image — {wt} window"
            img_id = image_store.save(windowed, desc, tool_name=self.name)
            renderings.append({
                "window_type": wt,
                "image_id": img_id,
                "description": desc,
                "size": {"width": windowed.width, "height": windowed.height},
            })
            logger.info("Rendered %s window -> %s", wt, img_id)

        result = {
            "renderings": renderings,
            "count": len([r for r in renderings if "image_id" in r]),
            "message": (
                f"Generated {len(renderings)} window rendering(s): "
                + ", ".join(
                    f"{r['window_type']}={r.get('image_id', 'FAILED')}"
                    for r in renderings
                )
            ),
        }

        return ToolResult(
            data=json.dumps(result),
            metadata={"image_ids": [r["image_id"] for r in renderings if "image_id" in r]},
        )