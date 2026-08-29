"""Шаг 0: AEGIS с HF Hub -> нормализованные train/val/test в data/processed/.

    python scripts/00_prepare_data.py --config configs/rq1.yaml
    python scripts/00_prepare_data.py --synthetic 400   # офлайн-прогон без сети
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eguard.config import load_config  # noqa: E402
from eguard.data import get_loader, split_path  # noqa: E402
from eguard.utils import get_logger, save_json, set_seed, write_jsonl  # noqa: E402

logger = get_logger("prepare_data")


def stratified_split(records: list[dict], fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Отрезает от records долю fraction со стратификацией по метке."""
    rng = np.random.default_rng(seed)
    rest, taken = [], []
    for label in (0, 1):
        idx = [i for i, r in enumerate(records) if r["label"] == label]
        rng.shuffle(idx)
        cut = int(round(len(idx) * fraction))
        taken_set = set(idx[:cut])
        for i in idx:
            (taken if i in taken_set else rest).append(records[i])
    rng.shuffle(rest)
    rng.shuffle(taken)
    return rest, taken


def summarize(name: str, records: list[dict]) -> dict:
    labels = Counter(r["label_name"] for r in records)
    cats = Counter(r["category"] for r in records if r["label"] == 1)
    lens = [r["n_chars"] for r in records] or [0]
    return {
        "split": name,
        "n": len(records),
        "labels": dict(labels),
        "harm_rate": round(labels["harm"] / max(len(records), 1), 3),
        "median_chars": int(np.median(lens)),
        "p95_chars": int(np.percentile(lens, 95)),
        "top_harm_categories": dict(cats.most_common(10)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rq1.yaml")
    ap.add_argument("--synthetic", type=int, default=0,
                    help="generate N synthetic lines intead of downloading from HF")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    loader = get_loader(cfg.dataset.name)

    if args.synthetic:
        logger.warning("SYNTHETIC data (%d line) — only for pipeline check-up", args.synthetic)
        raw = loader.make_synthetic(args.synthetic, seed=cfg.seed)
    else:
        logger.info("Loading %s from HF Hub", cfg.dataset.hf_id)
        raw = loader.load_raw(cfg.dataset)

    splits: dict[str, list[dict]] = {}
    for split, rows in raw.items():
        records = loader.normalize_rows(rows, cfg.dataset)
        logger.info("%s: %d -> %d records after filters", split, len(rows), len(records))
        splits[split] = records

    # Валидацию отрезаем от train: официальный test трогать нельзя.
    if "train" in splits and cfg.dataset.val_fraction > 0:
        splits["train"], splits["val"] = stratified_split(
            splits["train"], cfg.dataset.val_fraction, cfg.seed
        )

    summary = {"dataset": cfg.dataset.name, "config": cfg.raw["dataset"], "splits": []}
    for split, records in splits.items():
        for i, r in enumerate(records):
            r["split"] = split
            r["row_index"] = i
        path = split_path(cfg.paths.processed, cfg.dataset.name, split)
        n = write_jsonl(path, records)
        stats = summarize(split, records)
        summary["splits"].append(stats)
        logger.info("Written %s (%d lines): harm_rate=%.3f", path, n, stats["harm_rate"])

    save_json(Path(cfg.paths.processed) / cfg.dataset.name / "summary.json", summary)
    logger.info("Done. Summary: %s", Path(cfg.paths.processed) / cfg.dataset.name / "summary.json")


if __name__ == "__main__":
    main()
