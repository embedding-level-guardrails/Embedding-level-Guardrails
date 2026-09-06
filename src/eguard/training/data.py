"""Датасеты и коллаторы для contrastive- и classification-веток RQ3."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils import get_logger

logger = get_logger(__name__)


class PairDataset(Dataset):
    """Триплеты из data/processed/pairs/{name}/pairs_{split}.jsonl.

    `pair_types` — это ручка для абляции RQ2: обучение на подмножестве типов пар
    при прочих равных.
    """

    def __init__(self, pairs: list[dict], pair_types: list[str] | None = None):
        if pair_types:
            allowed = set(pair_types)
            pairs = [p for p in pairs if p["pair_type"] in allowed]
        self.pairs = pairs
        if not pairs:
            raise ValueError("Пустой набор пар после фильтра pair_types")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> dict[str, Any]:
        p = self.pairs[i]
        return {
            "anchor": p["anchor"],
            "positive": p["positive"],
            "negative": p["negative"],
            "anchor_label": int(p["anchor_label"]),
            "negative_label": int(p["negative_label"]),
            "pair_type": p["pair_type"],
        }


class TextDataset(Dataset):
    """Плоские тексты с бинарной меткой — вход classification-ветки."""

    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict[str, Any]:
        r = self.records[i]
        return {"text": r["text"], "label": int(r["label"])}


def collate_pairs(batch: list[dict]) -> dict[str, Any]:
    return {
        "anchor": [b["anchor"] for b in batch],
        "positive": [b["positive"] for b in batch],
        "negative": [b["negative"] for b in batch],
        "anchor_label": torch.tensor([b["anchor_label"] for b in batch], dtype=torch.long),
        "negative_label": torch.tensor([b["negative_label"] for b in batch], dtype=torch.long),
        "pair_type": [b["pair_type"] for b in batch],
    }


def collate_texts(batch: list[dict]) -> dict[str, Any]:
    return {
        "text": [b["text"] for b in batch],
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }


def texts_and_labels(records: list[dict]) -> tuple[list[str], np.ndarray]:
    return [r["text"] for r in records], np.array([int(r["label"]) for r in records])
