from traffic_data_engine.schema import Frame, SceneTags
from traffic_data_engine.tag import tag_statistics


def test_tag_statistics_counts_across_frames():
    frames = [
        Frame("a.jpg", scene=SceneTags(tags={"conditions": ["night", "rain"]})),
        Frame("b.jpg", scene=SceneTags(tags={"conditions": ["night"], "road_events": []})),
        Frame("c.jpg"),  # untagged frames are skipped
    ]
    stats = tag_statistics(frames)
    assert stats["conditions"]["night"] == 2
    assert stats["conditions"]["rain"] == 1
    assert stats["road_events"] == {}


def test_tag_statistics_empty():
    assert tag_statistics([]) == {}
