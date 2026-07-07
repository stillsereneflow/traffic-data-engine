# Design: traffic-data-engine

An auto-labeling and scene-understanding pipeline for driving data, built to
run on a single consumer GPU (6 GB VRAM). The pipeline turns raw driving
images into reviewable pseudo-labels and traffic-domain scene tags, measures
its own label quality against ground truth, and mines long-tail scenes.

## Goals

- **Data engine, not a model zoo**: every stage reads and writes the same
  plain-JSON `Frame` format, so stages can be run, cached, inspected and
  swapped independently.
- **Traffic-domain taxonomy as a first-class artifact**: the scene/intent/
  violation tag set lives in a reviewed YAML file informed by traffic
  engineering practice (signal control types, vulnerable road users,
  rule-violation categories) — not an ad-hoc prompt string.
- **Trustworthy metrics**: average precision is implemented from scratch in
  numpy and unit-tested on hand-computed cases, so pseudo-label quality
  numbers are transparent.
- **6 GB VRAM budget**: stages load one model at a time; the VLM runs
  quantized. Nothing in the pipeline assumes a data-center GPU.

## Non-goals

- No training or fine-tuning (inference-only pipeline).
- No video tracking in the first milestones (frames are independent).
- No labeling UI — outputs are JSON/COCO for existing review tools.

## Pipeline

```
images ──> detect ──> segment ──> tag ──────> Frame JSON ──> eval (vs GT)
           (Grounding  (SAM,      (Qwen2.5-VL             └─> mine (CLIP
            DINO,       boxes ->   + taxonomy                  embedding
            open-vocab) masks)     prompts)                    search)
```

- **detect** — open-vocabulary detection with Grounding DINO (HuggingFace
  `IDEA-Research/grounding-dino-tiny`); the class list is free text, so rare
  classes ("wheelchair", "fallen tree") cost nothing to add.
- **segment** — SAM (`facebook/sam-vit-base`) prompted with detected boxes;
  masks are stored as COCO-style RLE inside the Frame.
- **tag** — Qwen2.5-VL (3B, 8-bit) answers a structured prompt generated
  from `configs/taxonomy.yaml` and must return strict JSON; responses are
  validated against the taxonomy and invalid tags are dropped.
- **eval** — matches pseudo-labels against ground-truth boxes (greedy by
  score at a configurable IoU threshold) and reports per-class AP and mAP,
  implemented from scratch.
- **mine** — CLIP image embeddings + text queries to retrieve long-tail
  scenes ("a construction zone at night") from unlabeled pools.

## Module boundaries

| Module | Depends on models? | Tested in CI |
| --- | --- | --- |
| `schema`, `boxes`, `rle`, `taxonomy`, `prompts`, `evaluate` | no (numpy/PIL only) | yes |
| `detect`, `segment`, `tag`, `mine` | yes (lazy imports) | interface only |
| `cli` | dispatches to both | argument parsing |

GPU-dependent modules import torch/transformers inside the class
constructor, so the core library installs and tests cleanly without the
`[models]` extra.

## Error handling

- Taxonomy validation fails loudly at load time (duplicate or empty tags).
- VLM responses that are not valid JSON or contain unknown tags are recorded
  with `raw_response` preserved and the invalid parts dropped, never
  silently accepted — auto-labels must be auditable.
- Evaluation refuses to compare frames whose image paths do not align.

## Testing

- Pure-logic modules have exhaustive unit tests, including hand-computed
  AP/PR cases and RLE round-trips.
- CI (GitHub Actions, CPU-only): ruff lint + pytest.
- GPU stages are validated by running the shipped demo on real images and
  committing the rendered results to the README.

## Roadmap

- [x] M0: scaffolding — package layout, CI, design doc
- [x] M1: core data model + from-scratch AP evaluation + taxonomy/prompt
      builder, fully unit-tested
- [x] M2: detection stage on real images (Grounding DINO) + demo outputs
- [x] M3: SAM segmentation stage + RLE mask storage
- [x] M4: VLM scene tagging with the traffic taxonomy (Qwen2.5-VL, 8-bit;
      4-bit bitsandbytes corrupted generations on Windows)
- [x] M5: CLIP long-tail mining + COCO street-class evaluation report
- [ ] M6: BDD100K adapter and end-to-end quality report in the README
