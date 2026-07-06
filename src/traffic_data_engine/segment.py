"""Segmentation stage: SAM masks prompted with detection boxes.

Consumes the Frame JSON written by the detect stage, adds a COCO-style RLE
mask to every detection, and writes the same format back. Torch imports are
lazy, matching the other model stages.
"""

from __future__ import annotations

import numpy as np

from .boxes import clip_boxes
from .rle import encode
from .schema import Frame


class SamSegmenter:
    """Box-prompted segmentation with HuggingFace SAM."""

    def __init__(self, model_id: str = "facebook/sam-vit-base", device: str | None = None):
        import torch
        from transformers import SamModel, SamProcessor

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        self.model_id = model_id

    def segment(self, frame: Frame, batch_size: int = 16) -> Frame:
        """Fills ``mask_rle`` for every detection in the frame, in place."""
        from PIL import Image

        if not frame.detections:
            return frame
        image = Image.open(frame.image_path).convert("RGB")
        boxes = clip_boxes(
            np.array([det.box_xyxy for det in frame.detections]), image.width, image.height
        )

        masks: list[np.ndarray] = []
        for start in range(0, len(boxes), batch_size):
            chunk = boxes[start : start + batch_size].tolist()
            inputs = self.processor(
                image, input_boxes=[chunk], return_tensors="pt"
            ).to(self.device)
            with self._torch.no_grad():
                outputs = self.model(**inputs, multimask_output=False)
            processed = self.processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )[0]  # (chunk, 1, H, W) bool tensor
            masks.extend(m[0].numpy().astype(np.uint8) for m in processed)

        for detection, mask in zip(frame.detections, masks, strict=True):
            detection.mask_rle = encode(mask)
        return frame


def segment_frames(
    frames: list[Frame], model_id: str = "facebook/sam-vit-base", device: str | None = None
) -> list[Frame]:
    segmenter = SamSegmenter(model_id=model_id, device=device)
    for frame in frames:
        segmenter.segment(frame)
    return frames


def frames_mask_coverage(frame: Frame) -> float:
    """Fraction of detections that carry a mask (quick sanity metric)."""
    if not frame.detections:
        return 1.0
    with_mask = sum(1 for d in frame.detections if d.mask_rle is not None)
    return with_mask / len(frame.detections)
