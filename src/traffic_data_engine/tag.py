"""Scene tagging stage: Qwen2.5-VL answers the taxonomy prompt per frame.

The model must return strict JSON; the response is validated against the
taxonomy by ``prompts.parse_scene_response`` and rejected items are surfaced,
never silently kept. The model loads in 8-bit when bitsandbytes is
available and falls back to fp16 with CPU offload otherwise — either way it
runs next to a 6 GB GPU.
"""

from __future__ import annotations

from collections import Counter

from .prompts import build_scene_prompt, parse_scene_response
from .schema import Frame, SceneTags
from .taxonomy import Taxonomy


class QwenSceneTagger:
    def __init__(
        self,
        taxonomy: Taxonomy,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        max_new_tokens: int = 256,
        quantize: bool = True,
    ):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self._torch = torch
        self.taxonomy = taxonomy
        self.prompt = build_scene_prompt(taxonomy)
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

        kwargs: dict = {"device_map": "auto"}
        if quantize:
            try:
                import bitsandbytes  # noqa: F401  (probe availability)
                from transformers import BitsAndBytesConfig

                # 8-bit: fits a 3B VLM into ~4 GB and is numerically much
                # sturdier than 4-bit on Windows builds of bitsandbytes,
                # which we observed corrupting generated tokens.
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            except ImportError:
                quantize = False
        if not quantize:
            # fp16 with automatic CPU offload when VRAM runs short.
            kwargs["torch_dtype"] = torch.float16

        # Bound the vision token budget so 4K dashcam frames stay within 6 GB.
        self.processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
        self.model.eval()

    def tag(self, frame: Frame) -> Frame:
        """Fills ``frame.scene`` in place; keeps the raw response for audits."""
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": frame.image_path},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt"
        ).to(self.model.device)
        with self._torch.no_grad():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        trimmed = generated[:, inputs.input_ids.shape[1] :]
        response = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]

        try:
            tags, rejected = parse_scene_response(response, self.taxonomy)
        except ValueError:
            tags = {group.name: [] for group in self.taxonomy.groups}
            rejected = ["<unparseable response>"]
        if rejected:
            print(f"  rejected: {rejected}")
        frame.scene = SceneTags(tags=tags, raw_response=response, source=self.model_id)
        return frame


def tag_statistics(frames: list[Frame]) -> dict[str, Counter]:
    """Tag frequency per group across frames — the long-tail overview."""
    stats: dict[str, Counter] = {}
    for frame in frames:
        if frame.scene is None:
            continue
        for group, values in frame.scene.tags.items():
            stats.setdefault(group, Counter()).update(values)
    return stats
