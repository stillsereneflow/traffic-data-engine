"""Long-tail mining: CLIP embeddings + free-text queries over image pools.

Build an index once, then retrieve scenes by description ("a construction
zone at night") to prioritize rare cases for labeling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def rank_by_similarity(
    query: np.ndarray, index: np.ndarray, top_k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Row indices and cosine similarities of the ``top_k`` best matches.

    Both the query vector and the index rows must be L2-normalized.
    """
    sims = index @ query
    top_k = max(0, min(top_k, len(sims)))
    order = np.argsort(-sims)[:top_k]
    return order, sims[order]


def save_index(path: str | Path, embeddings: np.ndarray, image_paths: list[str]) -> None:
    np.savez(path, embeddings=embeddings, paths=np.array(image_paths))


def load_index(path: str | Path) -> tuple[np.ndarray, list[str]]:
    data = np.load(path, allow_pickle=False)
    return data["embeddings"], [str(p) for p in data["paths"]]


class ClipMiner:
    """CLIP ViT-B/32 image/text embeddings (lazy torch imports)."""

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: str | None = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        self.model_id = model_id

    def embed_images(self, image_paths: list[str | Path], batch_size: int = 16) -> np.ndarray:
        from PIL import Image

        chunks: list[np.ndarray] = []
        for start in range(0, len(image_paths), batch_size):
            images = [
                Image.open(p).convert("RGB") for p in image_paths[start : start + batch_size]
            ]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            with self._torch.no_grad():
                features = self.model.get_image_features(**inputs)
            chunks.append(features.cpu().numpy())
        embeddings = np.concatenate(chunks, axis=0)
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def embed_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with self._torch.no_grad():
            features = self.model.get_text_features(**inputs)
        vector = features[0].cpu().numpy()
        return vector / np.linalg.norm(vector)
