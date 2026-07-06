import numpy as np
import pytest

from traffic_data_engine.evaluate import average_precision, evaluate_frames, match_predictions
from traffic_data_engine.schema import Detection, Frame


def det(label, score, box):
    return Detection(label=label, score=score, box_xyxy=tuple(box))


def test_match_perfect_prediction():
    tp = match_predictions(
        np.array([[0, 0, 10, 10]]), np.array([0.9]), np.array([[0, 0, 10, 10]]), 0.5
    )
    assert tp.tolist() == [True]


def test_match_duplicate_is_false_positive():
    # Two predictions on the same ground truth: only the higher score matches.
    tp = match_predictions(
        np.array([[0, 0, 10, 10], [1, 1, 11, 11]]),
        np.array([0.8, 0.9]),
        np.array([[0, 0, 10, 10]]),
        0.5,
    )
    assert tp.tolist() == [False, True]


def test_match_respects_iou_threshold():
    pred = np.array([[0, 0, 10, 10]])
    gt = np.array([[6, 0, 16, 10]])  # IoU = 4/16 = 0.25
    assert match_predictions(pred, np.array([0.9]), gt, 0.5).tolist() == [False]
    assert match_predictions(pred, np.array([0.9]), gt, 0.2).tolist() == [True]


def test_average_precision_hand_cases():
    # One GT, one perfect prediction: AP = 1.
    assert average_precision(np.array([0.9]), np.array([True]), num_gt=1) == pytest.approx(1.0)

    # One GT; a higher-scored false positive before the true positive:
    # the precision envelope is 0.5 everywhere -> AP = 0.5.
    ap = average_precision(np.array([0.9, 0.8]), np.array([False, True]), num_gt=1)
    assert ap == pytest.approx(0.5)

    # Two GT, only one found: precision 1 up to recall 0.5, then 0.
    # 101-point interpolation counts the 51 recall points <= 0.5.
    ap = average_precision(np.array([0.9]), np.array([True]), num_gt=2)
    assert ap == pytest.approx(51 / 101)

    # No predictions at all.
    assert average_precision(np.array([]), np.array([]), num_gt=3) == 0.0

    # Undefined without ground truth.
    assert np.isnan(average_precision(np.array([0.9]), np.array([False]), num_gt=0))


def test_evaluate_frames_two_images_two_classes():
    gt = [
        Frame(
            "a.jpg",
            detections=[det("car", 1.0, [0, 0, 10, 10]), det("person", 1.0, [20, 20, 30, 30])],
        ),
        Frame("b.jpg", detections=[det("car", 1.0, [5, 5, 15, 15])]),
    ]
    pred = [
        Frame(
            "a.jpg",
            detections=[det("car", 0.9, [0, 0, 10, 10]), det("person", 0.8, [20, 20, 30, 30])],
        ),
        Frame("b.jpg", detections=[det("car", 0.7, [5, 5, 15, 15])]),
    ]
    result = evaluate_frames(pred, gt, iou_threshold=0.5)
    assert result.mean_ap == pytest.approx(1.0)
    by_label = {c.label: c for c in result.per_class}
    assert by_label["car"].num_gt == 2
    assert by_label["car"].num_pred == 2
    assert by_label["person"].ap == pytest.approx(1.0)


def test_evaluate_frames_rejects_unknown_image():
    gt = [Frame("a.jpg")]
    pred = [Frame("mystery.jpg")]
    with pytest.raises(ValueError, match="mystery.jpg"):
        evaluate_frames(pred, gt)


def test_evaluate_excludes_classes_without_gt():
    gt = [Frame("a.jpg", detections=[det("car", 1.0, [0, 0, 10, 10])])]
    pred = [
        Frame(
            "a.jpg",
            detections=[det("car", 0.9, [0, 0, 10, 10]), det("unicorn", 0.9, [0, 0, 5, 5])],
        )
    ]
    result = evaluate_frames(pred, gt)
    labels = [c.label for c in result.per_class]
    assert labels == ["car"]  # 'unicorn' has no GT: AP undefined, excluded from mAP
