"""COCO instances JSON -> Frame ground truth for street-class evaluation."""

from __future__ import annotations

from pathlib import Path

from .schema import Detection, Frame

# COCO categories that matter for driving scenes.
STREET_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
}


def select_street_images(
    coco: dict, keep_classes: set[str] | None = None, min_objects: int = 3
) -> list[dict]:
    """COCO image records with at least ``min_objects`` street-class boxes.

    Crowd annotations (``iscrowd=1``) are group boxes and are excluded here
    and in the conversion, matching how they are ignored in this repo's
    simplified evaluation.
    """
    keep_classes = keep_classes or STREET_CLASSES
    categories = {c["id"]: c["name"] for c in coco["categories"]}
    counts: dict[int, int] = {}
    for annotation in coco["annotations"]:
        if annotation.get("iscrowd"):
            continue
        if categories[annotation["category_id"]] in keep_classes:
            counts[annotation["image_id"]] = counts.get(annotation["image_id"], 0) + 1
    selected = [img for img in coco["images"] if counts.get(img["id"], 0) >= min_objects]
    return sorted(selected, key=lambda img: img["id"])


def coco_to_frames(
    coco: dict,
    images_dir: str | Path,
    image_ids: set[int] | None = None,
    keep_classes: set[str] | None = None,
) -> list[Frame]:
    """Converts COCO ground truth into Frames (score 1.0, xywh -> xyxy)."""
    keep_classes = keep_classes or STREET_CLASSES
    categories = {c["id"]: c["name"] for c in coco["categories"]}
    images = {img["id"]: img for img in coco["images"]}
    if image_ids is not None:
        images = {i: img for i, img in images.items() if i in image_ids}

    frames = {
        image_id: Frame(
            image_path=str(Path(images_dir) / img["file_name"]),
            width=img["width"],
            height=img["height"],
        )
        for image_id, img in images.items()
    }
    for annotation in coco["annotations"]:
        if annotation.get("iscrowd") or annotation["image_id"] not in frames:
            continue
        label = categories[annotation["category_id"]]
        if label not in keep_classes:
            continue
        x, y, w, h = annotation["bbox"]
        frames[annotation["image_id"]].detections.append(
            Detection(label=label, score=1.0, box_xyxy=(x, y, x + w, y + h), source="coco-gt")
        )
    return [frames[i] for i in sorted(frames)]
