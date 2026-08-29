"""Детерминированный энкодер на хеш-признаках символьных n-грамм.

Нужен только для того, чтобы прогнать весь пайплайн без сети и без torch
(CI, отладка анализа и визуализации). В экспериментах не используется.
"""
from __future__ import annotations

import numpy as np

from ..config import EncoderSpec
from .base import BaseEncoder, EncodeResult


class DummyEncoder(BaseEncoder):
    def __init__(self, spec: EncoderSpec, dim: int = 128, n_buckets: int = 2**14, seed: int = 0):
        self.spec = spec
        self.key = spec.key
        self.dim = dim
        self.n_buckets = n_buckets
        rng = np.random.default_rng(seed)
        self.projection = rng.normal(size=(n_buckets, dim)).astype(np.float32) / np.sqrt(dim)

    def _features(self, text: str) -> np.ndarray:
        text = text.lower()
        grams = [text[i : i + 4] for i in range(max(len(text) - 3, 1))]
        vec = np.zeros(self.dim, dtype=np.float32)
        for g in grams:
            vec += self.projection[hash(g) % self.n_buckets]
        return vec / max(len(grams), 1)

    def encode(self, texts: list[str], batch_size: int = 32, show_progress: bool = False) -> EncodeResult:
        emb = np.stack([self._features(t) for t in texts]).astype(np.float32)
        return EncodeResult(embeddings=emb, scores=None, meta={"model": "dummy", "dim": self.dim})
