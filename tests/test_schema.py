from traffic_data_engine.schema import Detection, Frame, SceneTags, load_frames, save_frames


def test_frame_json_roundtrip(tmp_path):
    frames = [
        Frame(
            image_path="a.jpg",
            width=1280,
            height=720,
            detections=[
                Detection(
                    label="car",
                    score=0.91,
                    box_xyxy=(10.0, 20.0, 110.0, 220.0),
                    source="grounding-dino-tiny",
                ),
                Detection(
                    label="person",
                    score=0.55,
                    box_xyxy=(0.0, 0.0, 5.0, 5.0),
                    mask_rle={"size": [720, 1280], "counts": [921600]},
                ),
            ],
            scene=SceneTags(
                tags={"scene_type": ["urban_street"], "conditions": ["night", "rain"]},
                raw_response='{"scene_type": ["urban_street"]}',
                source="qwen2.5-vl-3b",
            ),
        ),
        Frame(image_path="b.jpg"),
    ]
    path = tmp_path / "frames.json"
    save_frames(frames, path)
    restored = load_frames(path)

    assert len(restored) == 2
    assert restored[0].detections[0].label == "car"
    assert restored[0].detections[0].box_xyxy == (10.0, 20.0, 110.0, 220.0)
    assert restored[0].detections[1].mask_rle == {"size": [720, 1280], "counts": [921600]}
    assert restored[0].scene.tags["conditions"] == ["night", "rain"]
    assert restored[1].scene is None
    assert restored[1].detections == []
