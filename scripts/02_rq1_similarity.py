"""Шаг 2 (RQ1): косинусная близость внутри/между классами + геометрия пространства.

    python scripts/02_rq1_similarity.py --config configs/rq1.yaml

На выходе:
    results/rq1/similarity.csv   — intra/inter косинусы, сырые и центрированные
    results/rq1/geometry.csv     — silhouette, ARI, анизотропия, harm-направление
    results/figures/cos_hist_{model}.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eguard.analysis.geometry import geometry_report  # noqa: E402
from eguard.analysis.similarity import cosine_analysis, flatten_cosine_row, similarity_samples  # noqa: E402
from eguard.config import load_config  # noqa: E402
from eguard.embeddings import available_encoders, load_xy  # noqa: E402
from eguard.utils import ensure_dir, get_logger, save_json, set_seed  # noqa: E402
from eguard.viz import plot_similarity_hist  # noqa: E402

logger = get_logger("rq1_similarity")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rq1.yaml")
    ap.add_argument("--encoders", nargs="*", default=None)
    ap.add_argument("--split", default="test", help="split for analysis")
    ap.add_argument("--train-split", default="train", help="where centroids are taken from for harm-direction")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    keys = args.encoders or available_encoders(cfg.paths.embeddings, cfg.dataset.name)
    if not keys:
        raise SystemExit("No calculated embeddings — first scripts/01_embed.py")
    logger.info("Модели: %s", keys)

    sim_cfg = cfg.analysis.get("similarity", {})
    geo_cfg = cfg.analysis.get("geometry", {})
    out_dir = ensure_dir(Path(cfg.paths.results) / "rq1")

    sim_rows, geo_rows, raw_dump = [], [], {}

    for key in keys:
        x, y, _, _ = load_xy(cfg, key, args.split)
        x_tr, y_tr, _, _ = load_xy(cfg, key, args.train_split)
        logger.info("%s: %s, harm_rate=%.3f", key, x.shape, y.mean())

        stats = cosine_analysis(
            x, y, sample_per_class=sim_cfg.get("sample_per_class", 2000), seed=cfg.seed, center=True
        )
        sim_rows.extend(flatten_cosine_row(key, args.split, stats))
        raw_dump[key] = stats

        geo = geometry_report(x_tr, y_tr, x, y, k_values=geo_cfg.get("kmeans_k", [2, 8]), seed=cfg.seed)
        geo_rows.append({"model": key, "split": args.split, **geo})

        samples = similarity_samples(x, y, sample_per_class=500, seed=cfg.seed)
        plot_similarity_hist(
            samples, f"{key} — cosine similarity ({cfg.dataset.name}/{args.split})",
            Path(cfg.paths.figures) / f"cos_hist_{key}.png",
        )

    sim_df = pd.DataFrame(sim_rows).round(4)
    geo_df = pd.DataFrame(geo_rows).round(4)
    sim_df.to_csv(out_dir / "similarity.csv", index=False)
    geo_df.to_csv(out_dir / "geometry.csv", index=False)
    save_json(out_dir / "similarity_full.json", raw_dump)

    print("\n=== Cosine ===")
    print(sim_df.to_string(index=False))
    print("\n=== Geometry ===")
    print(geo_df.to_string(index=False))
    logger.info("Resulst: %s", out_dir)


if __name__ == "__main__":
    main()
