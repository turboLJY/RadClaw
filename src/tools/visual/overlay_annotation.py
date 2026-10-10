"""
OverlayAnnotationTool — Draw annotations on an image.

Supports bounding boxes, arrows, circles and text labels.  The annotated
image is saved as a new entry in the image store (the source is not
modified).
"""

from __future__ import annotations

import json
import logging

from src.core.tool import Tool, ToolUseContext
from src.core.types import ToolResult
from src.utils.image_processing import draw_annotation

logger = logging.getLogger(__name__)

_VALID_ANNOTATION_TYPES = {"bbox", "arrow", "circle", "text"}


class OverlayAnnotationTool(Tool):
    name = "overlay_annotation"
    description = (
        "Overlay annotations (bounding boxes, arrows, text labels, circles) on an "
        "image. Each annotation specifies a type, coordinates, optional label and colour. "
        "The annotated image is saved as a new entry — the source is not modified."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "image_id": {
                "type": "string",
                "description": "ID of the source image in the image store.",
            },
            "annotations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": list(_VALID_ANNOTATION_TYPES),
                            "description": (
                                "Annotation type: 'bbox' [x,y,w,h], 'arrow' [x1,y1,x2,y2], "
                                "'circle' [cx,cy,r], 'text' [x,y]."
                            ),
                        },
                        "coords": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Coordinate values (semantics depend on type).",
                        },
                        "label": {
                            "type": "string",
                            "description": "Text label for the annotation.",
                            "default": "",
                        },
                        "color": {
                            "type": "string",
                            "description": "Colour name (red, green, blue, yellow, cyan, magenta, white, orange) or hex (#RRGGBB).",
                            "default": "red",
                        },
                    },
                    "required": ["type", "coords"],
                },
                "description": "List of annotations to overlay.",
                "minItems": 1,
            },
        },
        "required": ["image_id", "annotations"],
    }
    is_concurrency_safe = True
    is_read_only = False

    def prompt(self) -> str:
        return (
            "Use overlay_annotation to mark findings on an image. Draw bounding boxes "
            "around abnormalities, arrows pointing to subtle findings, circles around "
            "nodules, or text labels for anatomical landmarks. This creates clear visual "
            "evidence that can be linked to findings via link_visual_evidence."
        )

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        image_id: str = args.get("image_id", "")
        annotations: list[dict] = args.get("annotations", [])

        if not annotations:
            return ToolResult(
                data=json.dumps({"error": "annotations must be a non-empty list"}),
                is_error=True,
            )

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

        # Validate annotation types
        for ann in annotations:
            ann_type = ann.get("type", "")
            if ann_type not in _VALID_ANNOTATION_TYPES:
                return ToolResult(
                    data=json.dumps({
                        "error": (
                            f"Invalid annotation type '{ann_type}'. "
                            f"Must be one of: {sorted(_VALID_ANNOTATION_TYPES)}"
                        ),
                    }),
                    is_error=True,
                )
            coords = ann.get("coords", [])
            if not coords:
                return ToolResult(
                    data=json.dumps({"error": f"Annotation of type '{ann_type}' has empty coords"}),
                    is_error=True,
                )

        try:
            annotated = draw_annotation(src_img, annotations)
        except Exception as exc:
            logger.exception("draw_annotation failed")
            return ToolResult(
                data=json.dumps({"error": f"Annotation failed: {exc}"}),
                is_error=True,
            )

        # Build description from annotations
        ann_summary = ", ".join(
            f"{a.get('type','?')}:{a.get('label','')}" for a in annotations[:5]
        )
        if len(annotations) > 5:
            ann_summary += f" (+{len(annotations)-5} more)"
        desc = f"{src_desc} [annotated: {ann_summary}]"

        new_id = image_store.save(annotated, desc, tool_name=self.name)
        image_store.add_tool_to_chain(new_id, self.name)

        result = {
            "image_id": new_id,
            "source_image_id": image_id,
            "annotation_count": len(annotations),
            "annotations_applied": [
                {"type": a.get("type"), "label": a.get("label", "")}
                for a in annotations
            ],
            "output_size": {"width": annotated.width, "height": annotated.height},
            "description": desc,
            "message": (
                f"Applied {len(annotations)} annotation(s) to {image_id}. "
                f"Result saved as {new_id}."
            ),
        }

        logger.info("Annotated %s -> %s (%d annotations)", image_id, new_id, len(annotations))
        return ToolResult(
            data=json.dumps(result),
            metadata={"image_id": new_id},
        )
