"""Data model shared by every pipeline stage.

A :class:`Frame` holds everything the pipeline knows about one image:
detections (optionally with RLE masks), scene tags and provenance. Frames
serialize to plain JSON so each stage can be run, cached and audited
independently.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Detection:
    label: str
    score: float
    box_xyxy: tuple[float, float, float, float]  # pixels, (x1, y1, x2, y2)
    mask_rle: dict | None = None  # COCO-style uncompressed RLE, see rle.py
    source: str = ""  # model that produced it, e.g. "grounding-dino-tiny"


@dataclass
class SceneTags:
    tags: dict[str, list[str]] = field(default_factory=dict)  # group -> selected tags
    raw_response: str = ""  # unparsed VLM output, kept for auditing
    source: str = ""


@dataclass
class Frame:
    image_path: str
    width: int = 0
    height: int = 0
    detections: list[Detection] = field(default_factory=list)
    scene: SceneTags | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> Frame:
        detections = [
            Detection(
                label=d["label"],
                score=d["score"],
                box_xyxy=tuple(d["box_xyxy"]),
                mask_rle=d.get("mask_rle"),
                source=d.get("source", ""),
            )
            for d in data.get("detections", [])
        ]
        scene_data = data.get("scene")
        scene = (
            SceneTags(
                tags=scene_data.get("tags", {}),
                raw_response=scene_data.get("raw_response", ""),
                source=scene_data.get("source", ""),
            )
            if scene_data
            else None
        )
        return Frame(
            image_path=data["image_path"],
            width=data.get("width", 0),
            height=data.get("height", 0),
            detections=detections,
            scene=scene,
        )


def save_frames(frames: list[Frame], path: str | Path) -> None:
    payload = [frame.to_dict() for frame in frames]
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_frames(path: str | Path) -> list[Frame]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Frame.from_dict(item) for item in payload]
