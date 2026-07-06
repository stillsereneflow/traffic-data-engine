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


def _cmd_tag(args: argparse.Namespace) -> None:
    from .schema import load_frames, save_frames
    from .tag import QwenSceneTagger
    from .taxonomy import load_taxonomy

    taxonomy = load_taxonomy(args.taxonomy)
    frames = load_frames(args.frames)
    tagger = QwenSceneTagger(taxonomy, model_id=args.model)
    for frame in frames:
        tagger.tag(frame)
        picked = sum(len(v) for v in frame.scene.tags.values())
        print(f"{Path(frame.image_path).name}: {picked} tags")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_frames(frames, out_path)
    print(f"frames written to {out_path}")

    if args.render:
        from .visualize import draw_scene_card

        for frame in frames:
            rendered = draw_scene_card(
                frame, taxonomy, Path(args.render) / Path(frame.image_path).name
            )
            print(f"rendered {rendered}")


def _cmd_eval(args: argparse.Namespace) -> None:
    from .evaluate import evaluate_frames
    from .schema import load_frames

    result = evaluate_frames(
        load_frames(args.pred), load_frames(args.gt), iou_threshold=args.iou
    )
    print(f"{'class':<16}{'AP':>8}{'num_gt':>8}{'num_pred':>10}")
    for cls in sorted(result.per_class, key=lambda c: -c.ap):
        print(f"{cls.label:<16}{cls.ap:>8.3f}{cls.num_gt:>8}{cls.num_pred:>10}")
    print(f"\nmAP@{args.iou:g} = {result.mean_ap:.3f} over {len(result.per_class)} classes")


def _cmd_mine_index(args: argparse.Namespace) -> None:
    from .mine import ClipMiner, save_index

    paths = [str(p) for p in _iter_images(Path(args.images))]
    miner = ClipMiner(model_id=args.model)
    embeddings = miner.embed_images(paths)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_index(args.out, embeddings, paths)
    print(f"indexed {len(paths)} images -> {args.out}")


def _cmd_mine(args: argparse.Namespace) -> None:
    from .mine import ClipMiner, load_index, rank_by_similarity

    embeddings, paths = load_index(args.index)
    miner = ClipMiner(model_id=args.model)
    order, sims = rank_by_similarity(miner.embed_text(args.query), embeddings, args.top_k)
    results = [(paths[i], float(s)) for i, s in zip(order, sims, strict=True)]
    for path, score in results:
        print(f"{score:.3f}  {path}")

    if args.render:
        from .visualize import draw_ranking

        rendered = draw_ranking(args.query, results, args.render)
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

    tag = subparsers.add_parser("tag", help="VLM scene tagging with the traffic taxonomy")
    tag.add_argument("--frames", required=True, help="Frame JSON from an earlier stage")
    tag.add_argument("--out", required=True, help="output Frame JSON path")
    tag.add_argument("--taxonomy", default="configs/taxonomy.yaml")
    tag.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    tag.add_argument("--render", default="", help="also render scene cards here")
    tag.set_defaults(func=_cmd_tag)

    evaluate = subparsers.add_parser("eval", help="pseudo-label AP/mAP against ground truth")
    evaluate.add_argument("--pred", required=True, help="predicted Frame JSON")
    evaluate.add_argument("--gt", required=True, help="ground-truth Frame JSON")
    evaluate.add_argument("--iou", type=float, default=0.5)
    evaluate.set_defaults(func=_cmd_eval)

    mine_index = subparsers.add_parser("mine-index", help="build a CLIP index over a folder")
    mine_index.add_argument("--images", required=True)
    mine_index.add_argument("--out", required=True, help="output .npz index path")
    mine_index.add_argument("--model", default="openai/clip-vit-base-patch32")
    mine_index.set_defaults(func=_cmd_mine_index)

    mine = subparsers.add_parser("mine", help="retrieve scenes by text query")
    mine.add_argument("--index", required=True, help=".npz index from mine-index")
    mine.add_argument("--query", required=True, help='e.g. "a construction zone at night"')
    mine.add_argument("--top-k", type=int, default=3)
    mine.add_argument("--model", default="openai/clip-vit-base-patch32")
    mine.add_argument("--render", default="", help="render the ranked results to this file")
    mine.set_defaults(func=_cmd_mine)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
