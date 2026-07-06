"""COCO-style uncompressed run-length encoding for binary masks.

The encoding matches pycocotools' uncompressed RLE: counts alternate between
runs of 0s and 1s in *column-major* (Fortran) order, always starting with the
count of 0s (which may be zero).
"""

from __future__ import annotations

import numpy as np


def encode(mask: np.ndarray) -> dict:
    """Encodes a binary (H, W) mask into uncompressed RLE."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    flat = np.asfortranarray(mask).ravel(order="F").astype(np.uint8)

    counts: list[int] = []
    current = 0  # RLE starts with a run of zeros
    run = 0
    for value in flat:
        if value == current:
            run += 1
        else:
            counts.append(run)
            current = 1 - current
            run = 1
    counts.append(run)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def decode(encoded: dict) -> np.ndarray:
    """Decodes uncompressed RLE back into a binary (H, W) uint8 mask."""
    height, width = encoded["size"]
    counts = encoded["counts"]
    total = height * width
    if sum(counts) != total:
        raise ValueError(f"RLE counts sum to {sum(counts)}, expected {total}")

    flat = np.zeros(total, dtype=np.uint8)
    position = 0
    value = 0
    for count in counts:
        if value == 1:
            flat[position : position + count] = 1
        position += count
        value = 1 - value
    return flat.reshape((height, width), order="F")


def mask_area(encoded: dict) -> int:
    """Number of foreground pixels, straight from the counts."""
    return int(sum(encoded["counts"][1::2]))
