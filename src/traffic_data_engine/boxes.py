"""Bounding-box operations in pure numpy. Boxes are (x1, y1, x2, y2)."""

from __future__ import annotations

import numpy as np


def box_area(boxes: np.ndarray) -> np.ndarray:
    """Areas of an (N, 4) array of boxes; degenerate boxes get area 0."""
    widths = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None)
    heights = np.clip(boxes[:, 3] - boxes[:, 1], 0.0, None)
    return widths * heights


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N, 4) and (M, 4) boxes, shape (N, M)."""
    boxes_a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    boxes_b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)

    left = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    top = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    right = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    bottom = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    intersection = np.clip(right - left, 0.0, None) * np.clip(bottom - top, 0.0, None)
    union = box_area(boxes_a)[:, None] + box_area(boxes_b)[None, :] - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0.0, intersection / union, 0.0)
    return iou


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy non-maximum suppression; returns kept indices, highest score first."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores)
    keep: list[int] = []
    while order.size > 0:
        current = order[0]
        keep.append(int(current))
        if order.size == 1:
            break
        rest = order[1:]
        ious = iou_matrix(boxes[current : current + 1], boxes[rest])[0]
        order = rest[ious <= iou_threshold]
    return np.array(keep, dtype=np.int64)


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """COCO (x, y, w, h) -> (x1, y1, x2, y2)."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    out = boxes.copy()
    out[:, 2] = boxes[:, 0] + boxes[:, 2]
    out[:, 3] = boxes[:, 1] + boxes[:, 3]
    return out


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """(x1, y1, x2, y2) -> COCO (x, y, w, h)."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    out = boxes.copy()
    out[:, 2] = boxes[:, 2] - boxes[:, 0]
    out[:, 3] = boxes[:, 3] - boxes[:, 1]
    return out
