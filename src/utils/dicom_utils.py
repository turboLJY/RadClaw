"""
X-ray windowing utilities for RadClaw.

Simulates different CT window/level settings on chest X-ray images by
adjusting brightness, contrast, and gamma.  These are not true DICOM
windowing operations (no HU values) but provide useful perceptual
enhancements for different tissue types when applied to standard X-ray
images.
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)


def _apply_gamma(arr: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction to a float [0,1] array."""
    return np.power(np.clip(arr, 0.0, 1.0), 1.0 / gamma)


def _apply_window(
    img: Image.Image,
    *,
    gamma: float,
    brightness: float,
    contrast: float,
) -> Image.Image:
    """Core windowing helper: gamma -> brightness -> contrast."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0

    # Gamma
    arr = _apply_gamma(arr, gamma)

    # Brightness
    arr = arr * brightness

    # Contrast (centred at 0.5)
    arr = 0.5 + (arr - 0.5) * contrast

    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# Public API — one function per window preset
# ---------------------------------------------------------------------------

def apply_bone_window(img: Image.Image) -> Image.Image:
    """Enhance bone structures.

    Uses low gamma (brightens darks) and high contrast to emphasise
    cortical bone, fractures and calcifications.
    """
    logger.debug("Applying bone window to %dx%d image", *img.size)
    return _apply_window(img, gamma=0.5, brightness=1.3, contrast=2.2)


def apply_lung_window(img: Image.Image) -> Image.Image:
    """Enhance lung parenchyma.

    Uses high gamma (darkens mids/highlights) and moderate contrast to
    bring out pulmonary vasculature and airspace detail.
    """
    logger.debug("Applying lung window to %dx%d image", *img.size)
    return _apply_window(img, gamma=1.8, brightness=0.7, contrast=1.5)


def apply_soft_tissue_window(img: Image.Image) -> Image.Image:
    """Enhance soft tissue.

    Preserves mid-range brightness while boosting contrast for muscle,
    fat and organ boundaries.
    """
    logger.debug("Applying soft-tissue window to %dx%d image", *img.size)
    return _apply_window(img, gamma=1.0, brightness=1.0, contrast=1.8)


def apply_mediastinal_window(img: Image.Image) -> Image.Image:
    """Enhance mediastinal structures.

    Slightly brightens and raises contrast to differentiate vascular and
    lymph node structures in the mediastinum.
    """
    logger.debug("Applying mediastinal window to %dx%d image", *img.size)
    return _apply_window(img, gamma=0.8, brightness=1.1, contrast=2.0)


# ---------------------------------------------------------------------------
# Convenience map
# ---------------------------------------------------------------------------

WINDOW_FUNCTIONS: dict[str, callable] = {
    "bone": apply_bone_window,
    "lung": apply_lung_window,
    "soft_tissue": apply_soft_tissue_window,
    "mediastinal": apply_mediastinal_window,
}


def apply_window_by_name(img: Image.Image, window_type: str) -> Image.Image:
    """Look up and apply a window preset by name.

    Raises ``ValueError`` if *window_type* is not one of the known presets.
    """
    fn = WINDOW_FUNCTIONS.get(window_type.strip().lower())
    if fn is None:
        raise ValueError(
            f"Unknown window type '{window_type}'. "
            f"Must be one of: {', '.join(WINDOW_FUNCTIONS)}"
        )
    return fn(img)