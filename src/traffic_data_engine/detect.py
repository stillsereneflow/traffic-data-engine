"""Open-vocabulary detection stage built on Grounding DINO.

Torch and transformers are imported lazily inside the detector constructor,
so the core package installs and tests without the ``[models]`` extra. The
class list is free text: adding a rare class ("wheelchair", "fallen tree")
costs nothing.
"""

from __future__ import annotations

from pathlib import Path

from .schema import Detection, Frame

# A practical default for driving scenes; override with --classes.
DEFAULT_CLASSES = [
    "car",
    "truck",
    "bus",
    "person",
    "bicycle",
    "motorcycle",
    "traffic light",
    "traffic sign",
    "traffic cone",
]


def build_query(classes: list[str]) -> str:
    """Grounding DINO expects lowercase phrases separated by '. '."""
    phrases = [c.strip().lower() for c in classes if c.strip()]
    if not phrases:
        raise ValueError("classes must contain at least one non-empty phrase")
    return ". ".join(phrases) + "."

def normalize_label(text: str, classes: list[str]) -> str | None:
    """Maps a predicted phrase back to the requested class list.

    Grounding DINO occasionally returns merged phrases ("car. truck");
    the longest requested class contained in the phrase wins. Returns None
    for phrases that match no requested class — those detections are
    dropped rather than passed through with unvetted labels.
    """
    text = text.strip().lower()
    if not text:
        return None
    exact = [c for c in classes if c.strip().lower() == text]
    if exact:
        return exact[0]
    contained = [c for c in classes if c.strip().lower() and c.strip().lower() in text]
    if contained:
        return max(contained, key=lambda c: len(c.strip()))
    return None


class GroundingDinoDetector:
    """Folder-level pseudo-labeling with HuggingFace Grounding DINO."""

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: str | None = None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        self.model_id = model_id
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def detect(self, image_path: str | Path, classes: list[str] | None = None) -> Frame:
        from PIL import Image

        classes = classes or DEFAULT_CLASSES
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(
            images=image, text=build_query(classes), return_tensors="pt"
        ).to(self.device)
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        frame = Frame(image_path=str(image_path), width=image.width, height=image.height)
        boxes = results["boxes"]
        # transformers renamed the phrase field from "labels" to "text_labels".
        phrases = list(results.get("text_labels", results.get("labels")) or [])
        # Quirk: an image with zero detections still yields one empty phrase,
        # so align the phrase list with the boxes before zipping strictly.
        if len(phrases) != len(boxes):
            phrases = phrases[: len(boxes)] + [""] * (len(boxes) - len(phrases))
        for box, score, phrase in zip(boxes, results["scores"], phrases, strict=True):
            label = normalize_label(str(phrase), classes)
            if label is None:
                continue
            frame.detections.append(
                Detection(
                    label=label,
                    score=float(score),
                    box_xyxy=tuple(float(v) for v in box.tolist()),
                    source=self.model_id,
                )
            )
        return frame
