from traffic_data_engine.coco import coco_to_frames, select_street_images

FIXTURE = {
    "categories": [
        {"id": 1, "name": "person"},
        {"id": 3, "name": "car"},
        {"id": 90, "name": "toothbrush"},
    ],
    "images": [
        {"id": 10, "file_name": "10.jpg", "width": 640, "height": 480},
        {"id": 20, "file_name": "20.jpg", "width": 640, "height": 480},
    ],
    "annotations": [
        {"image_id": 10, "category_id": 3, "bbox": [10, 20, 30, 40], "iscrowd": 0},
        {"image_id": 10, "category_id": 1, "bbox": [0, 0, 5, 5], "iscrowd": 0},
        {"image_id": 10, "category_id": 3, "bbox": [0, 0, 9, 9], "iscrowd": 1},  # crowd: skip
        {"image_id": 20, "category_id": 90, "bbox": [1, 1, 2, 2], "iscrowd": 0},  # not street
    ],
}


def test_select_street_images_counts_only_street_classes():
    selected = select_street_images(FIXTURE, min_objects=2)
    assert [img["id"] for img in selected] == [10]
    # Image 20 only has a toothbrush; image 10's crowd box does not count.
    assert select_street_images(FIXTURE, min_objects=3) == []


def test_coco_to_frames_converts_boxes_and_skips_crowd():
    frames = coco_to_frames(FIXTURE, "imgs", image_ids={10})
    assert len(frames) == 1
    frame = frames[0]
    assert frame.image_path.endswith("10.jpg")
    assert frame.width == 640
    labels = sorted(d.label for d in frame.detections)
    assert labels == ["car", "person"]  # crowd annotation dropped
    car = next(d for d in frame.detections if d.label == "car")
    assert car.box_xyxy == (10, 20, 40, 60)  # xywh -> xyxy
    assert car.score == 1.0
