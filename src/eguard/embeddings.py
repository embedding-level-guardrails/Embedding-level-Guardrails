"""Вычисление и кеширование эмбеддингов.

Раскладка на диске:
    artifacts/embeddings/{dataset}/{encoder_key}/{split}.npy          # [n, dim] float32
    artifacts/embeddings/{dataset}/{encoder_key}/{split}.scores.npy   # [n] (если у модели есть голова)
    artifacts/embeddings/{dataset}/{encoder_key}/{split}.meta.json
Порядок строк совпадает с порядком записей в data/processed/{dataset}/{split}.jsonl.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .encoders import BaseEncoder
from .utils import ensure_dir, get_logger, save_json

logger = get_logger(__name__)


def emb_dir(root: str | Path, dataset: str, encoder_key: str) -> Path:
    return Path(root) / dataset / encoder_key


def emb_paths(root: str | Path, dataset: str, encoder_key: str, split: str) -> dict[str, Path]:
    d = emb_dir(root, dataset, encoder_key)
    return {
        "emb": d / f"{split}.npy",
        "scores": d / f"{split}.scores.npy",
        "meta": d / f"{split}.meta.json",
    }


def compute_and_cache(
    encoder: BaseEncoder,
    texts: list[str],
    root: str | Path,
    dataset: str,
    split: str,
    batch_size: int = 32,
    overwrite: bool = False,
) -> Path:
    paths = emb_paths(root, dataset, encoder.key, split)
    if paths["emb"].exists() and not overwrite:
        logger.info("%s/%s: embeddings already exist, skipping (--overwrite for recalculating)", encoder.key, split)
        return paths["emb"]

    ensure_dir(paths["emb"].parent)
    result = encoder.encode(texts, batch_size=batch_size)

    assert result.embeddings.shape[0] == len(texts), "number of embeddings != number of texts"
    np.save(paths["emb"], result.embeddings.astype(np.float32))
    if result.scores is not None:
        np.save(paths["scores"], result.scores.astype(np.float32))
    save_json(paths["meta"], {"split": split, "dataset": dataset, **(result.meta or {})})
    logger.info("Saved: %s %s", paths["emb"], result.embeddings.shape)
    return paths["emb"]


def load_embeddings(
    root: str | Path, dataset: str, encoder_key: str, split: str
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    paths = emb_paths(root, dataset, encoder_key, split)
    if not paths["emb"].exists():
        raise FileNotFoundError(f"No {paths['emb']}. Run scripts/01_embed.py")
    emb = np.load(paths["emb"])
    scores = np.load(paths["scores"]) if paths["scores"].exists() else None
    meta = json.loads(paths["meta"].read_text(encoding="utf-8")) if paths["meta"].exists() else {}
    return emb, scores, meta


def load_xy(cfg, encoder_key: str, split: str):
    """(X, y, records, scores) for singular split. Order of lines is corresponding."""
    from .data import load_split

    records = load_split(cfg.paths.processed, cfg.dataset.name, split)
    emb, scores, _ = load_embeddings(cfg.paths.embeddings, cfg.dataset.name, encoder_key, split)
    if len(records) != len(emb):
        raise ValueError(
            f"{encoder_key}/{split}: {len(emb)} embeddings against {len(records)} records — "
            "data overwritten after embedding, rerun 01_embed.py --overwrite"
        )
    y = np.array([r["label"] for r in records], dtype=int)
    return emb, y, records, scores


def available_encoders(root: str | Path, dataset: str) -> list[str]:
    base = Path(root) / dataset
    return sorted(p.name for p in base.glob("*") if p.is_dir()) if base.exists() else []
