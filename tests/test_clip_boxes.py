import numpy as np

from traffic_data_engine.boxes import clip_boxes


def test_clip_boxes_clamps_to_extent():
    boxes = np.array(
        [
            [-5.0, -5.0, 10.0, 10.0],
            [90.0, 40.0, 120.0, 80.0],
            [10.0, 10.0, 20.0, 20.0],  # already inside
        ]
    )
    clipped = clip_boxes(boxes, width=100, height=60)
    assert np.allclose(clipped[0], [0.0, 0.0, 10.0, 10.0])
    assert np.allclose(clipped[1], [90.0, 40.0, 100.0, 60.0])
    assert np.allclose(clipped[2], boxes[2])


def test_clip_boxes_does_not_mutate_input():
    boxes = np.array([[-1.0, 0.0, 5.0, 5.0]])
    clip_boxes(boxes, 10, 10)
    assert boxes[0, 0] == -1.0
