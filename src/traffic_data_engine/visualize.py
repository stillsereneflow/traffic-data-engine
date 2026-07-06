"""Renders Frame detections onto images for quick review (PIL only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .rle import decode
from .schema import Frame
from .taxonomy import Taxonomy

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


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """A font that can render Chinese, with graceful fallbacks."""
    for name in ("msyh.ttc", "simhei.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_ranking(
    query: str, results: list[tuple[str, float]], out_path: str | Path
) -> Path:
    """Ranked retrieval results side by side, scores overlaid, query on top."""
    thumb_height = 300
    gap = 8
    title_height = 52
    thumbs: list[tuple[Image.Image, float]] = []
    for path, score in results:
        image = Image.open(path).convert("RGB")
        width = max(1, int(image.width * thumb_height / image.height))
        thumbs.append((image.resize((width, thumb_height)), score))

    total_width = sum(t.width for t, _ in thumbs) + gap * max(0, len(thumbs) - 1)
    canvas = Image.new("RGB", (max(total_width, 480), title_height + thumb_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), f'query: "{query}"', fill=(30, 30, 30), font=_font(26))
    x = 0
    for thumb, score in thumbs:
        canvas.paste(thumb, (x, title_height))
        draw.rectangle((x, title_height, x + 108, title_height + 34), fill=(12, 123, 220))
        draw.text((x + 10, title_height + 4), f"{score:.3f}", fill="white", font=_font(22))
        x += thumb.width + gap

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def draw_scene_card(
    frame: Frame, taxonomy: Taxonomy, out_path: str | Path, panel_width: int = 480
) -> Path:
    """Image on the left, the VLM's validated scene tags on the right.

    The panel lists every taxonomy group with its selected tags (bilingual),
    so a reviewer can read the whole scene assessment at a glance.
    """
    image = Image.open(frame.image_path).convert("RGB")
    target_height = 720
    scale = target_height / image.height
    image = image.resize((int(image.width * scale), target_height))

    card = Image.new("RGB", (image.width + panel_width, target_height), "white")
    card.paste(image, (0, 0))
    draw = ImageDraw.Draw(card)

    header_font = _font(26)
    tag_font = _font(21)
    x = image.width + 24
    y = 20
    tags = frame.scene.tags if frame.scene else {}
    for group in taxonomy.groups:
        draw.text((x, y), f"{group.zh}  {group.name}", fill=(30, 30, 30), font=header_font)
        y += 36
        selected = tags.get(group.name, [])
        if not selected:
            draw.text((x + 16, y), "—", fill=(160, 160, 160), font=tag_font)
            y += 30
        for name in selected:
            option = next((o for o in group.options if o.name == name), None)
            zh = option.zh if option else ""
            draw.text((x + 16, y), f"• {zh}  {name}", fill=_color_for(group.name), font=tag_font)
            y += 30
        y += 12

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    card.save(out_path)
    return out_path


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
