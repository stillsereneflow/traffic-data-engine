"""Renders Frame detections onto images for quick review (PIL only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .rle import decode
from .schema import Frame

# Colorblind-friendly palette; labels are assigned deterministically.
_PALETTE = [
    (230, 97, 0),
    (12, 123, 220),
    (64, 176, 166),
    (211, 95, 183),
    (255, 194, 10),
    (26, 133, 255),
    (212, 17, 89),
    (0, 158, 115),
]


def _color_for(label: str) -> tuple[int, int, int]:
    return _PALETTE[hash(label) % len(_PALETTE)]


def _overlay_masks(image: Image.Image, frame: Frame, min_score: float) -> Image.Image:
    """Blends every detection mask over the image as a semi-transparent fill."""
    overlay = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    for det in frame.detections:
        if det.mask_rle is None or det.score < min_score:
            continue
        mask = decode(det.mask_rle).astype(bool)
        overlay[mask] = (*_color_for(det.label), 110)
    if not overlay[..., 3].any():
        return image
    return Image.alpha_composite(
        image.convert("RGBA"), Image.fromarray(overlay, "RGBA")
    ).convert("RGB")


def draw_frame(frame: Frame, out_path: str | Path, min_score: float = 0.0) -> Path:
    """Draws masks (when present), boxes and label:score captions."""
    image = Image.open(frame.image_path).convert("RGB")
    image = _overlay_masks(image, frame, min_score)
    draw = ImageDraw.Draw(image)
    line_width = max(2, image.width // 400)
    try:
        font = ImageFont.truetype("arial.ttf", size=max(14, image.width // 70))
    except OSError:
        font = ImageFont.load_default()

    for det in sorted(frame.detections, key=lambda d: d.score):
        if det.score < min_score:
            continue
        color = _color_for(det.label)
        x1, y1, x2, y2 = det.box_xyxy
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        caption = f"{det.label} {det.score:.2f}"
        text_box = draw.textbbox((x1, y1), caption, font=font)
        text_height = text_box[3] - text_box[1]
        top = max(0.0, y1 - text_height - 2 * line_width)
        text_width = text_box[2] - text_box[0]
        draw.rectangle(
            (x1, top, x1 + text_width + 2 * line_width, top + text_height + 2 * line_width),
            fill=color,
        )
        draw.text((x1 + line_width, top + line_width), caption, fill="white", font=font)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return out_path
