"""
GenerateProbeTool — Generate analytical probe images for deeper reasoning.

Creates diagnostic visualizations (histograms, edge maps, region masks,
intensity profiles) from stored images.  These probes help the VLM
quantify and reason about image features that are hard to assess by
eye alone.
"""

from __future__ import annotations

import json
import logging

import numpy as np
from PIL import Image

from src.core.tool import Tool, ToolUseContext
from src.core.types import ToolResult
from src.utils.image_processing import (
    compute_edge_map,
    compute_histogram,
    compute_intensity_profile,
)

logger = logging.getLogger(__name__)

_VALID_PROBE_TYPES = {"histogram", "edge_map", "region_mask", "intensity_profile"}


class GenerateProbeTool(Tool):
    name = "generate_visual_probe"
    description = (
        "Generate an analytical probe image for deeper reasoning about an image. "
        "Probe types: histogram (pixel intensity distribution), edge_map (Sobel edges), "
        "region_mask (high-intensity region segmentation), intensity_profile (row/column "
        "mean intensity chart)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "image_id": {
                "type": "string",
                "description": "ID of the source image in the image store.",
            },
            "probe_type": {
                "type": "string",
                "enum": list(_VALID_PROBE_TYPES),
                "description": (
                    "Type of probe to generate: 'histogram', 'edge_map', "
                    "'region_mask', 'intensity_profile'."
                ),
            },
            "axis": {
                "type": "string",
                "enum": ["horizontal", "vertical"],
                "description": "Axis for intensity_profile probe (default 'horizontal').",
                "default": "horizontal",
            },
        },
        "required": ["image_id", "probe_type"],
    }
    is_concurrency_safe = True
    is_read_only = False

    def prompt(self) -> str:
        return (
            "Use generate_visual_probe to produce quantitative visualizations. "
            "'histogram' shows the pixel intensity distribution — useful for assessing "
            "exposure and dynamic range. 'edge_map' highlights structural boundaries — "
            "good for detecting subtle consolidation borders or rib fractures. "
            "'region_mask' segments high-intensity regions — helpful for identifying "
            "opacities. 'intensity_profile' shows mean intensity along an axis — "
            "useful for detecting asymmetry between left and right lung fields."
        )

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        image_id: str = args.get("image_id", "")
        probe_type: str = args.get("probe_type", "").strip().lower()
        axis: str = args.get("axis", "horizontal").strip().lower()

        if probe_type not in _VALID_PROBE_TYPES:
            return ToolResult(
                data=json.dumps({
                    "error": (
                        f"Invalid probe_type '{probe_type}'. "
                        f"Must be one of: {sorted(_VALID_PROBE_TYPES)}"
                    ),
                }),
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
        analysis_data: dict = {}

        try:
            if probe_type == "histogram":
                probe_img = compute_histogram(src_img)
                # Compute summary stats
                arr = np.asarray(src_img.convert("L"), dtype=np.float64)
                analysis_data = {
                    "mean_intensity": float(np.mean(arr)),
                    "std_intensity": float(np.std(arr)),
                    "min_intensity": int(np.min(arr)),
                    "max_intensity": int(np.max(arr)),
                    "median_intensity": float(np.median(arr)),
                }

            elif probe_type == "edge_map":
                probe_img = compute_edge_map(src_img)
                # Compute edge density
                edge_arr = np.asarray(probe_img.convert("L"), dtype=np.float64)
                threshold = 30.0
                edge_pixels = int(np.sum(edge_arr > threshold))
                total_pixels = edge_arr.size
                analysis_data = {
                    "edge_density": round(edge_pixels / total_pixels, 4),
                    "edge_pixels": edge_pixels,
                    "total_pixels": total_pixels,
                    "mean_edge_magnitude": float(np.mean(edge_arr)),
                }

            elif probe_type == "region_mask":
                probe_img = self._generate_region_mask(src_img)
                mask_arr = np.asarray(probe_img.convert("L"), dtype=np.float64)
                bright_pixels = int(np.sum(mask_arr > 127))
                total = mask_arr.size
                analysis_data = {
                    "bright_region_fraction": round(bright_pixels / total, 4),
                    "bright_pixels": bright_pixels,
                    "total_pixels": total,
                }

            elif probe_type == "intensity_profile":
                probe_img = compute_intensity_profile(src_img, axis=axis)
                gray = np.asarray(src_img.convert("L"), dtype=np.float64)
                if axis in ("horizontal", "row", "x"):
                    profile = gray.mean(axis=0)
                else:
                    profile = gray.mean(axis=1)
                analysis_data = {
                    "axis": axis,
                    "profile_length": len(profile),
                    "profile_mean": float(np.mean(profile)),
                    "profile_std": float(np.std(profile)),
                    "profile_min": float(np.min(profile)),
                    "profile_max": float(np.max(profile)),
                }

            else:
                # Should not reach here due to validation above
                return ToolResult(
                    data=json.dumps({"error": f"Unhandled probe type: {probe_type}"}),
                    is_error=True,
                )

        except Exception as exc:
            logger.exception("Probe generation failed for type '%s'", probe_type)
            return ToolResult(
                data=json.dumps({"error": f"Probe generation failed: {exc}"}),
                is_error=True,
            )

        desc = f"{src_desc} [{probe_type} probe]"
        new_id = image_store.save(probe_img, desc, tool_name=self.name)
        image_store.add_tool_to_chain(new_id, self.name)

        result = {
            "image_id": new_id,
            "source_image_id": image_id,
            "probe_type": probe_type,
            "analysis_data": analysis_data,
            "output_size": {"width": probe_img.width, "height": probe_img.height},
            "description": desc,
            "message": (
                f"Generated {probe_type} probe from {image_id}. "
                f"Result saved as {new_id}."
            ),
        }

        logger.info("Generated %s probe %s -> %s", probe_type, image_id, new_id)
        return ToolResult(
            data=json.dumps(result),
            metadata={"image_id": new_id, "analysis_data": analysis_data},
        )

    @staticmethod
    def _generate_region_mask(img: Image.Image) -> Image.Image:
        """Create a binary mask highlighting high-intensity regions.

        Uses Otsu-style thresholding: pixels above (mean + 0.5*std) are
        marked bright, the rest are dark.  The result is a false-colour
        overlay where bright regions appear in red over a darkened
        version of the original.
        """
        gray = np.asarray(img.convert("L"), dtype=np.float64)
        threshold = float(np.mean(gray) + 0.5 * np.std(gray))
        mask = (gray > threshold).astype(np.uint8) * 255

        # Create overlay: dim original + red mask
        base = np.asarray(img.convert("RGB"), dtype=np.float64) * 0.4
        overlay = base.copy()
        overlay[:, :, 0] = np.where(mask > 0, 255, base[:, :, 0])  # Red channel
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        return Image.fromarray(overlay, mode="RGB")
