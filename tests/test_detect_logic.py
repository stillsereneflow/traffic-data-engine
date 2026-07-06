import pytest

from traffic_data_engine.detect import build_query, normalize_label


def test_build_query_joins_lowercase_phrases():
    assert build_query(["Car", "Traffic Light"]) == "car. traffic light."


def test_build_query_rejects_empty():
    with pytest.raises(ValueError):
        build_query(["", "  "])


def test_normalize_exact_match():
    assert normalize_label("car", ["car", "truck"]) == "car"
    assert normalize_label(" Car ", ["car", "truck"]) == "car"


def test_normalize_merged_phrase_takes_longest_contained():
    classes = ["car", "traffic light", "light"]
    assert normalize_label("traffic light", classes) == "traffic light"
    # Merged phrase: the longest requested class contained in the text wins.
    assert normalize_label("car. traffic light", classes) == "traffic light"


def test_normalize_unknown_phrase_dropped():
    assert normalize_label("zebra", ["car", "truck"]) is None
    assert normalize_label("", ["car"]) is None
