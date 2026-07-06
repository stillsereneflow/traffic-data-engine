import numpy as np

from traffic_data_engine.boxes import box_area, iou_matrix, nms, xywh_to_xyxy, xyxy_to_xywh


def test_iou_identical_boxes():
    box = np.array([[0.0, 0.0, 2.0, 2.0]])
    assert iou_matrix(box, box)[0, 0] == 1.0


def test_iou_disjoint_boxes():
    a = np.array([[0.0, 0.0, 1.0, 1.0]])
    b = np.array([[2.0, 2.0, 3.0, 3.0]])
    assert iou_matrix(a, b)[0, 0] == 0.0


def test_iou_hand_computed_overlap():
    a = np.array([[0.0, 0.0, 2.0, 2.0]])  # area 4
    b = np.array([[1.0, 0.0, 3.0, 2.0]])  # area 4, intersection 2
    assert np.isclose(iou_matrix(a, b)[0, 0], 2.0 / 6.0)


def test_iou_degenerate_box_is_zero():
    a = np.array([[1.0, 1.0, 1.0, 1.0]])  # zero area
    b = np.array([[0.0, 0.0, 2.0, 2.0]])
    assert iou_matrix(a, b)[0, 0] == 0.0
    assert box_area(a)[0] == 0.0


def test_nms_suppresses_overlaps_keeps_distinct():
    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],  # heavy overlap with the first
            [50.0, 50.0, 60.0, 60.0],  # far away
        ]
    )
    scores = np.array([0.9, 0.8, 0.7])
    kept = nms(boxes, scores, iou_threshold=0.5)
    assert kept.tolist() == [0, 2]


def test_nms_keeps_all_below_threshold():
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [8.0, 8.0, 18.0, 18.0]])  # slight overlap
    scores = np.array([0.5, 0.9])
    kept = nms(boxes, scores, iou_threshold=0.5)
    assert sorted(kept.tolist()) == [0, 1]
    assert kept[0] == 1  # highest score first


def test_xywh_roundtrip():
    xywh = np.array([[10.0, 20.0, 30.0, 40.0]])
    assert np.allclose(xyxy_to_xywh(xywh_to_xyxy(xywh)), xywh)
    assert np.allclose(xywh_to_xyxy(xywh), [[10.0, 20.0, 40.0, 60.0]])
