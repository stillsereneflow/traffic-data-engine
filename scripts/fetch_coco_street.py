#!/usr/bin/env python3
"""Builds a small COCO street-scene evaluation set.

Downloads the val2017 annotations once (~241 MB zip), selects images with at
least ``--min-objects`` street-class boxes, downloads those images
(~150 KB each) and writes ground-truth Frames to ``gt_frames.json``.

Usage:
    python scripts/fetch_coco_street.py --count 80
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from traffic_data_engine.coco import coco_to_frames, select_street_images  # noqa: E402
from traffic_data_engine.schema import save_frames  # noqa: E402

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANNOTATIONS_MEMBER = "annotations/instances_val2017.json"


def download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "traffic-data-engine/0.1"})
    with urllib.request.urlopen(request) as response:
        target.write_bytes(response.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=80, help="number of images to fetch")
    parser.add_argument("--min-objects", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=Path("data/coco"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    annotations_json = args.out_dir / "instances_val2017.json"
    if not annotations_json.exists():
        annotations_zip = args.out_dir / "annotations_trainval2017.zip"
        if not annotations_zip.exists():
            print(f"downloading {ANNOTATIONS_URL} (~241 MB) ...")
            download(ANNOTATIONS_URL, annotations_zip)
        print("extracting instances_val2017.json ...")
        with zipfile.ZipFile(annotations_zip) as archive:
            annotations_json.write_bytes(archive.read(ANNOTATIONS_MEMBER))

    coco = json.loads(annotations_json.read_text(encoding="utf-8"))
    selected = select_street_images(coco, min_objects=args.min_objects)[: args.count]
    print(f"selected {len(selected)} street-scene images")

    images_dir = args.out_dir / "images"
    images_dir.mkdir(exist_ok=True)
    for index, image in enumerate(selected):
        target = images_dir / image["file_name"]
        if target.exists():
            continue
        download(image["coco_url"], target)
        if index % 10 == 0:
            print(f"  {index + 1}/{len(selected)} images")
        time.sleep(0.15)  # be polite

    frames = coco_to_frames(coco, images_dir, image_ids={img["id"] for img in selected})
    save_frames(frames, args.out_dir / "gt_frames.json")
    total_boxes = sum(len(f.detections) for f in frames)
    print(f"wrote {len(frames)} ground-truth frames with {total_boxes} boxes")


if __name__ == "__main__":
    main()
