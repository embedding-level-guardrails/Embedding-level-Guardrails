"""Датасеты: загрузка с HF Hub, нормализация разметки, чтение готовых сплитов.

Раскладка на диске:
    data/processed/{dataset}/{split}.jsonl   # по записи на строку
    data/processed/{dataset}/summary.json    # статистика сплитов
Порядок строк в jsonl — это порядок строк в кешах эмбеддингов, менять его нельзя
без пересчёта artifacts/embeddings (см. проверку длин в eguard.embeddings.load_xy).
"""
from __future__ import annotations

from pathlib import Path
from types import ModuleType

from ..utils import read_jsonl
from . import aegis, toxicchat

LOADERS: dict[str, ModuleType] = {"aegis": aegis, "toxicchat": toxicchat}


def get_loader(name: str) -> ModuleType:
    """Загрузчик датасета по имени из конфига (`dataset.name`)."""
    try:
        return LOADERS[name]
    except KeyError:
        known = ", ".join(sorted(LOADERS))
        raise KeyError(f"Unknown dataset '{name}'. Available: {known}") from None


def split_path(root: str | Path, dataset: str, split: str) -> Path:
    return Path(root) / dataset / f"{split}.jsonl"


def load_split(root: str | Path, dataset: str, split: str) -> list[dict]:
    path = split_path(root, dataset, split)
    if not path.exists():
        raise FileNotFoundError(f"No {path}. Run scripts/00_prepare_data.py")
    return read_jsonl(path)


__all__ = ["aegis", "get_loader", "load_split", "split_path", "toxicchat"]
