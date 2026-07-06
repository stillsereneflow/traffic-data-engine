"""Pseudo-label quality evaluation: COCO-style AP implemented from scratch.

Matching follows the standard detection protocol: within each image,
predictions are taken in descending score order and greedily matched to the
not-yet-matched ground-truth box with the highest IoU at or above the
threshold. The precision-recall curve over all predictions (sorted globally
by score) is summarized with 101-point interpolated average precision, as in
COCO. Implemented in plain numpy so the numbers are auditable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boxes import iou_matrix
from .schema import Frame


@dataclass
class ClassEval:
    label: str
    ap: float
    num_gt: int
    num_pred: int


@dataclass
class EvalResult:
    iou_threshold: float
    per_class: list[ClassEval]

    @property
    def mean_ap(self) -> float:
        if not self.per_class:
            return 0.0
        return float(np.mean([c.ap for c in self.per_class]))


def match_predictions(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """True-positive flags for one image's predictions of a single class.

    Returns a boolean array aligned with the input prediction order.
    """
    pred_boxes = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    tp = np.zeros(len(pred_boxes), dtype=bool)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return tp

    order = np.argsort(-np.asarray(pred_scores, dtype=np.float64))
    ious = iou_matrix(pred_boxes, gt_boxes)
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)
    for pred_index in order:
        candidates = ious[pred_index].copy()
        candidates[gt_matched] = -1.0
        best_gt = int(np.argmax(candidates))
        if candidates[best_gt] >= iou_threshold:
            gt_matched[best_gt] = True
            tp[pred_index] = True
    return tp


def average_precision(scores: np.ndarray, tp: np.ndarray, num_gt: int) -> float:
    """101-point interpolated AP from per-prediction scores and TP flags."""
    if num_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0

    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    tp_sorted = np.asarray(tp, dtype=bool)[order]
    cum_tp = np.cumsum(tp_sorted)
    cum_fp = np.cumsum(~tp_sorted)
    recall = cum_tp / num_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1)

    # Precision envelope evaluated at 101 evenly spaced recall points.
    ap = 0.0
    for recall_point in np.linspace(0.0, 1.0, 101):
        mask = recall >= recall_point
        ap += float(np.max(precision[mask])) if np.any(mask) else 0.0
    return ap / 101.0


def evaluate_frames(
    pred_frames: list[Frame],
    gt_frames: list[Frame],
    labels: list[str] | None = None,
    iou_threshold: float = 0.5,
) -> EvalResult:
    """Per-class AP and mAP of predicted frames against ground-truth frames.

    Frames are aligned by image *file name* (so Windows and POSIX paths to
    the same files compare equal); a prediction for an image that has no
    ground-truth frame is an error, not a silent skip.
    """
    from pathlib import Path

    gt_by_image = {Path(frame.image_path).name: frame for frame in gt_frames}
    for frame in pred_frames:
        if Path(frame.image_path).name not in gt_by_image:
            raise ValueError(f"no ground truth for image '{frame.image_path}'")

    if labels is None:
        labels = sorted(
            {d.label for f in gt_frames for d in f.detections}
            | {d.label for f in pred_frames for d in f.detections}
        )

    per_class: list[ClassEval] = []
    for label in labels:
        scores: list[float] = []
        flags: list[bool] = []
        num_gt = sum(
            sum(1 for d in frame.detections if d.label == label) for frame in gt_frames
        )
        for frame in pred_frames:
            preds = [d for d in frame.detections if d.label == label]
            gt_frame = gt_by_image[Path(frame.image_path).name]
            gts = [d for d in gt_frame.detections if d.label == label]
            if not preds:
                continue
            pred_boxes = np.array([d.box_xyxy for d in preds])
            pred_scores = np.array([d.score for d in preds])
            gt_boxes = (
                np.array([d.box_xyxy for d in gts]) if gts else np.zeros((0, 4))
            )
            tp = match_predictions(pred_boxes, pred_scores, gt_boxes, iou_threshold)
            scores.extend(pred_scores.tolist())
            flags.extend(tp.tolist())

        ap = average_precision(np.array(scores), np.array(flags), num_gt)
        per_class.append(
            ClassEval(label=label, ap=ap, num_gt=num_gt, num_pred=len(scores))
        )

    # Classes without ground truth have undefined AP and are excluded from mAP.
    valid = [c for c in per_class if not np.isnan(c.ap)]
    return EvalResult(iou_threshold=iou_threshold, per_class=valid)
