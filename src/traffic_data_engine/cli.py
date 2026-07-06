"""Command line interface.

Every stage reads and writes the shared Frame JSON, so stages compose:

    tde detect --images data/samples --out out/frames.json --render out/vis
"""

from __future__ import annotations

import argparse
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _iter_images(images_dir: Path) -> list[Path]:
    paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise SystemExit(f"no images found in {images_dir}")
    return paths


def _cmd_detect(args: argparse.Namespace) -> None:
    from .detect import DEFAULT_CLASSES, GroundingDinoDetector
    from .schema import save_frames

    classes = args.classes.split(",") if args.classes else DEFAULT_CLASSES
    detector = GroundingDinoDetector(
        model_id=args.model, box_threshold=args.box_threshold
    )
    frames = []
    for image_path in _iter_images(Path(args.images)):
        frame = detector.detect(image_path, classes)
        frames.append(frame)
        print(f"{image_path.name}: {len(frame.detections)} detections")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_frames(frames, out_path)
    print(f"frames written to {out_path}")

    if args.render:
        from .visualize import draw_frame

        for frame in frames:
            rendered = draw_frame(frame, Path(args.render) / Path(frame.image_path).name)
            print(f"rendered {rendered}")


def _cmd_segment(args: argparse.Namespace) -> None:
    from .schema import load_frames, save_frames
    from .segment import SamSegmenter, frames_mask_coverage

    frames = load_frames(args.frames)
    segmenter = SamSegmenter(model_id=args.model)
    for frame in frames:
        segmenter.segment(frame)
        coverage = frames_mask_coverage(frame)
        print(f"{Path(frame.image_path).name}: {coverage:.0%} of detections masked")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_frames(frames, out_path)
    print(f"frames written to {out_path}")

    if args.render:
        from .visualize import draw_frame

        for frame in frames:
            rendered = draw_frame(frame, Path(args.render) / Path(frame.image_path).name)
            print(f"rendered {rendered}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="tde", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect = subparsers.add_parser("detect", help="open-vocabulary detection on a folder")
    detect.add_argument("--images", required=True, help="folder of input images")
    detect.add_argument("--out", required=True, help="output Frame JSON path")
    detect.add_argument("--classes", default="", help="comma-separated class phrases")
    detect.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    detect.add_argument("--box-threshold", type=float, default=0.35)
    detect.add_argument("--render", default="", help="also render annotated images here")
    detect.set_defaults(func=_cmd_detect)

    segment = subparsers.add_parser("segment", help="add SAM masks to detected frames")
    segment.add_argument("--frames", required=True, help="Frame JSON from the detect stage")
    segment.add_argument("--out", required=True, help="output Frame JSON path")
    segment.add_argument("--model", default="facebook/sam-vit-base")
    segment.add_argument("--render", default="", help="also render annotated images here")
    segment.set_defaults(func=_cmd_segment)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
