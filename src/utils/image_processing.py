"""
PIL/numpy image processing utilities for RadClaw visual analysis tools.

Provides low-level image manipulation functions used by the visual tool suite:
cropping, zooming, enhancement, windowing, grid composition, annotation,
histogram analysis, edge detection, and intensity profiling.
"""

from __future__ import annotations

import base64
import io
import logging
import math
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FONT_SIZE = 14
_ANNOTATION_COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "white": (255, 255, 255),
    "orange": (255, 165, 0),
}

_WINDOW_PRESETS: dict[str, dict[str, float]] = {
    "bone": {"gamma": 0.5, "brightness": 1.3, "contrast": 2.2},
    "lung": {"gamma": 1.8, "brightness": 0.7, "contrast": 1.5},
    "soft_tissue": {"gamma": 1.0, "brightness": 1.0, "contrast": 1.8},
    "mediastinal": {"gamma": 0.8, "brightness": 1.1, "contrast": 2.0},
}


def _get_font(size: int = _FONT_SIZE) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Attempt to load a TrueType font, fall back to default."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", size)
        except (OSError, IOError):
            return ImageFont.load_default()


def _resolve_color(name: str) -> tuple[int, int, int]:
    """Resolve a colour name or hex string to an RGB tuple."""
    name = name.strip().lower()
    if name in _ANNOTATION_COLORS:
        return _ANNOTATION_COLORS[name]
    if name.startswith("#") and len(name) == 7:
        try:
            return (int(name[1:3], 16), int(name[3:5], 16), int(name[5:7], 16))
        except ValueError:
            pass
    return _ANNOTATION_COLORS["red"]


# ---------------------------------------------------------------------------
# Core Utilities
# ---------------------------------------------------------------------------

def crop_region(img: Image.Image, x: int, y: int, w: int, h: int) -> Image.Image:
    """Crop a rectangular region with bounds checking.

    Parameters
    ----------
    img : PIL.Image.Image
        Source image.
    x, y : int
        Top-left corner of the crop box (pixels).
    w, h : int
        Width and height of the crop box (pixels).

    Returns
    -------
    PIL.Image.Image
        Cropped image.
    """
    img_w, img_h = img.size
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    x2 = min(x + w, img_w)
    y2 = min(y + h, img_h)

    if x2 <= x or y2 <= y:
        logger.warning(
            "Crop region degenerate after clamping: (%d,%d,%d,%d) on %dx%d image",
            x, y, w, h, img_w, img_h,
        )
        # Return a 1x1 crop at the clamped origin rather than raising
        x2 = max(x + 1, x2)
        y2 = max(y + 1, y2)

    cropped = img.crop((x, y, x2, y2))
    logger.debug("Cropped region (%d,%d)-(%d,%d) from %dx%d image", x, y, x2, y2, img_w, img_h)
    return cropped


def zoom_image(
    img: Image.Image,
    x: int,
    y: int,
    w: int,
    h: int,
    zoom_factor: float = 2.0,
) -> Image.Image:
    """Crop a region then resize it up by *zoom_factor*.

    The output image dimensions are ``(w * zoom_factor, h * zoom_factor)``
    (clamped to the crop that fits within the source).
    """
    cropped = crop_region(img, x, y, w, h)
    new_w = max(1, int(cropped.width * zoom_factor))
    new_h = max(1, int(cropped.height * zoom_factor))
    zoomed = cropped.resize((new_w, new_h), Image.LANCZOS)
    logger.debug("Zoomed %dx%d -> %dx%d (factor=%.1f)", cropped.width, cropped.height, new_w, new_h, zoom_factor)
    return zoomed


def enhance_image(img: Image.Image, enhance_type: str) -> Image.Image:
    """Apply an enhancement filter to the image.

    Parameters
    ----------
    enhance_type : str
        One of ``"sharpen"``, ``"edge"``, ``"contrast"``.
    """
    enhance_type = enhance_type.strip().lower()

    if enhance_type == "sharpen":
        enhanced = img.filter(ImageFilter.SHARPEN)
        enhanced = ImageEnhance.Sharpness(enhanced).enhance(2.0)
    elif enhance_type == "edge":
        gray = img.convert("L")
        enhanced = gray.filter(ImageFilter.FIND_EDGES)
        enhanced = enhanced.convert("RGB")
    elif enhance_type == "contrast":
        enhanced = ImageEnhance.Contrast(img).enhance(2.0)
        enhanced = ImageEnhance.Brightness(enhanced).enhance(1.1)
    else:
        logger.warning("Unknown enhance_type '%s', returning original", enhance_type)
        enhanced = img.copy()

    return enhanced


def adjust_window(img: Image.Image, window_type: str) -> Image.Image:
    """Simulate CT-style windowing on an X-ray image.

    Adjusts gamma, brightness and contrast to mimic the radiological
    window/level settings used for *bone*, *lung*, *soft_tissue* and
    *mediastinal* views.
    """
    window_type = window_type.strip().lower()
    preset = _WINDOW_PRESETS.get(window_type)
    if preset is None:
        logger.warning("Unknown window_type '%s', returning original", window_type)
        return img.copy()

    arr = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0

    # Gamma correction
    gamma = preset["gamma"]
    arr = np.power(np.clip(arr, 0, 1), 1.0 / gamma)

    # Brightness scaling
    arr = arr * preset["brightness"]

    # Contrast adjustment (centre around 0.5)
    contrast = preset["contrast"]
    arr = 0.5 + (arr - 0.5) * contrast

    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# Composition / Annotation
# ---------------------------------------------------------------------------

def compose_grid(
    images: list[Image.Image],
    labels: list[str] | None = None,
    layout: str = "horizontal",
) -> Image.Image:
    """Stitch a list of images into a labelled grid.

    Parameters
    ----------
    layout : str
        ``"horizontal"`` for a single row, ``"grid"`` for a roughly square
        arrangement.
    """
    if not images:
        raise ValueError("compose_grid requires at least one image")

    labels = labels or [""] * len(images)
    if len(labels) < len(images):
        labels.extend([""] * (len(images) - len(labels)))

    font = _get_font()
    label_h = _FONT_SIZE + 8 if any(labels) else 0
    padding = 4

    if layout == "grid":
        cols = math.ceil(math.sqrt(len(images)))
        rows = math.ceil(len(images) / cols)
    else:
        cols = len(images)
        rows = 1

    # Uniform cell size = max across all images
    cell_w = max(im.width for im in images) + 2 * padding
    cell_h = max(im.height for im in images) + label_h + 2 * padding

    canvas_w = cols * cell_w
    canvas_h = rows * cell_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    for idx, (im, label) in enumerate(zip(images, labels)):
        r = idx // cols
        c = idx % cols
        ox = c * cell_w + padding
        oy = r * cell_h + padding

        # Centre the image within its cell
        ix = ox + (cell_w - 2 * padding - im.width) // 2
        iy = oy + label_h + (cell_h - label_h - 2 * padding - im.height) // 2
        canvas.paste(im.convert("RGB"), (ix, iy))

        if label:
            draw.text((ox + 4, oy + 2), label, fill=(255, 255, 255), font=font)

    return canvas


def draw_annotation(img: Image.Image, annotations: list[dict]) -> Image.Image:
    """Draw bounding boxes, arrows, circles and text labels on an image.

    Each dict in *annotations* should contain:
    - ``type``: ``"bbox"`` | ``"arrow"`` | ``"circle"`` | ``"text"``
    - ``coords``: list of ints (semantics depend on type, see below)
    - ``label``: optional text label
    - ``color``: optional colour name or hex string (default ``"red"``)

    Coordinate semantics:
    - ``bbox``:  [x, y, w, h]
    - ``arrow``: [x1, y1, x2, y2]
    - ``circle``: [cx, cy, r]
    - ``text``:  [x, y]
    """
    canvas = img.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = _get_font()

    for ann in annotations:
        ann_type = ann.get("type", "bbox")
        coords = ann.get("coords", [])
        label = ann.get("label", "")
        color = _resolve_color(ann.get("color", "red"))
        line_width = 2

        try:
            if ann_type == "bbox" and len(coords) >= 4:
                x, y, w, h = coords[:4]
                draw.rectangle([x, y, x + w, y + h], outline=color, width=line_width)
                if label:
                    draw.text((x, max(0, y - _FONT_SIZE - 2)), label, fill=color, font=font)

            elif ann_type == "arrow" and len(coords) >= 4:
                x1, y1, x2, y2 = coords[:4]
                draw.line([(x1, y1), (x2, y2)], fill=color, width=line_width)
                # Simple arrowhead
                angle = math.atan2(y2 - y1, x2 - x1)
                head_len = 12
                for da in (0.5, -0.5):
                    hx = x2 - head_len * math.cos(angle + da)
                    hy = y2 - head_len * math.sin(angle + da)
                    draw.line([(x2, y2), (int(hx), int(hy))], fill=color, width=line_width)
                if label:
                    mx = (x1 + x2) // 2
                    my = (y1 + y2) // 2
                    draw.text((mx, my), label, fill=color, font=font)

            elif ann_type == "circle" and len(coords) >= 3:
                cx, cy, r = coords[:3]
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=line_width)
                if label:
                    draw.text((cx + r + 4, cy - _FONT_SIZE // 2), label, fill=color, font=font)

            elif ann_type == "text" and len(coords) >= 2:
                x, y = coords[:2]
                text = label or ""
                if text:
                    draw.text((x, y), text, fill=color, font=font)

            else:
                logger.warning("Skipping annotation with type '%s' and %d coords", ann_type, len(coords))

        except Exception:
            logger.exception("Failed to draw annotation %s", ann)

    return canvas


# ---------------------------------------------------------------------------
# Analytical Visualizations
# ---------------------------------------------------------------------------

def compute_histogram(img: Image.Image) -> Image.Image:
    """Generate a histogram visualisation for the image.

    Returns a 512x300 RGB image plotting per-channel intensity histograms.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.asarray(img.convert("RGB"))
    fig, ax = plt.subplots(figsize=(5.12, 3.0), dpi=100)

    for ch, color in enumerate(("red", "green", "blue")):
        hist, bin_edges = np.histogram(arr[:, :, ch].ravel(), bins=256, range=(0, 256))
        ax.plot(bin_edges[:-1], hist, color=color, alpha=0.7, linewidth=0.8)

    ax.set_xlim(0, 255)
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Count")
    ax.set_title("Intensity Histogram")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="PNG")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def compute_edge_map(img: Image.Image) -> Image.Image:
    """Compute a Sobel edge-detection map.

    Returns a greyscale-to-RGB edge magnitude image.
    """
    gray = np.asarray(img.convert("L"), dtype=np.float64)

    # Sobel kernels
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

    from scipy.signal import convolve2d

    gx = convolve2d(gray, kx, mode="same", boundary="symm")
    gy = convolve2d(gray, ky, mode="same", boundary="symm")
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8) if magnitude.max() > 0 else magnitude.astype(np.uint8)

    edge_img = Image.fromarray(magnitude, mode="L").convert("RGB")
    return edge_img


def compute_intensity_profile(img: Image.Image, axis: str = "horizontal") -> Image.Image:
    """Compute mean-intensity profile along an axis.

    Parameters
    ----------
    axis : str
        ``"horizontal"`` — average across rows (profile along x).
        ``"vertical"`` — average across columns (profile along y).

    Returns a matplotlib chart as a PIL image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gray = np.asarray(img.convert("L"), dtype=np.float64)

    if axis.strip().lower() in ("horizontal", "row", "x"):
        profile = gray.mean(axis=0)
        xlabel = "Column (x)"
    else:
        profile = gray.mean(axis=1)
        xlabel = "Row (y)"

    fig, ax = plt.subplots(figsize=(5.12, 3.0), dpi=100)
    ax.plot(profile, color="steelblue", linewidth=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean Intensity")
    ax.set_title(f"Intensity Profile ({axis})")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="PNG")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image to a base64 string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")
