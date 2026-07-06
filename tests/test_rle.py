import numpy as np
import pytest

from traffic_data_engine.rle import decode, encode, mask_area


def test_known_tiny_mask():
    mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    encoded = encode(mask)
    # Column-major flattening gives [1, 0, 0, 1]: zero 0s, one 1, two 0s, one 1.
    assert encoded["size"] == [2, 2]
    assert encoded["counts"] == [0, 1, 2, 1]
    assert mask_area(encoded) == 2


def test_roundtrip_random_masks():
    rng = np.random.default_rng(3)
    for _ in range(20):
        height = int(rng.integers(1, 40))
        width = int(rng.integers(1, 40))
        mask = (rng.random((height, width)) > 0.6).astype(np.uint8)
        restored = decode(encode(mask))
        assert np.array_equal(restored, mask)


def test_all_zero_and_all_one():
    zeros = np.zeros((5, 3), dtype=np.uint8)
    assert decode(encode(zeros)).sum() == 0
    ones = np.ones((5, 3), dtype=np.uint8)
    assert decode(encode(ones)).sum() == 15
    assert mask_area(encode(ones)) == 15


def test_decode_rejects_bad_counts():
    with pytest.raises(ValueError):
        decode({"size": [2, 2], "counts": [1, 1]})  # sums to 2, expected 4


def test_encode_rejects_non_2d():
    with pytest.raises(ValueError):
        encode(np.zeros((2, 2, 3), dtype=np.uint8))
