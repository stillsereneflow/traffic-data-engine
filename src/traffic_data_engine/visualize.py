"""Renders Frame detections onto images for quick review (PIL only)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

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


def draw_frame(frame: Frame, out_path: str | Path, min_score: float = 0.0) -> Path:
    """Draws boxes and label:score captions; returns the output path."""
    image = Image.open(frame.image_path).convert("RGB")
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
