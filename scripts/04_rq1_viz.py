"""Шаг 4 (RQ1): UMAP / t-SNE / PCA проекции embedding space.

    python scripts/04_rq1_viz.py --config configs/rq1.yaml
    python scripts/04_rq1_viz.py --methods pca tsne --n-points 2000

Каждая картинка — два подграфика: раскраска по safe/harm и по harm-категории
(второе нужно, чтобы увидеть, распадается ли harm на подкластеры по темам).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eguard.config import load_config  # noqa: E402
from eguard.embeddings import available_encoders, load_xy  # noqa: E402
from eguard.utils import get_logger, set_seed, subsample_indices  # noqa: E402
from eguard.viz import plot_projection, project  # noqa: E402

logger = get_logger("rq1_viz")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rq1.yaml")
    ap.add_argument("--encoders", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--n-points", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    methods = args.methods or cfg.viz.get("methods", ["pca", "tsne"])
    split = args.split or cfg.viz.get("split", "test")
    n_points = args.n_points or cfg.viz.get("n_points", 3000)

    keys = args.encoders or available_encoders(cfg.paths.embeddings, cfg.dataset.name)
    if not keys:
        raise SystemExit("No calculated embeddings — first scripts/01_embed.py")

    rng = np.random.default_rng(cfg.seed)
    for key in keys:
        x, y, records, _ = load_xy(cfg, key, split)
        idx = subsample_indices(len(x), n_points, rng)
        xs, ys = x[idx], y[idx]
        cats = [records[i]["category"] if records[i]["label"] == 1 else "safe" for i in idx]

        for method in methods:
            coords = project(xs, method, seed=cfg.seed)
            if coords is None:
                continue
            path = plot_projection(
                coords, ys, f"{key} · {method.upper()} · {cfg.dataset.name}/{split}",
                Path(cfg.paths.figures) / f"{method}_{key}_{split}.png", categories=cats,
            )
            logger.info("Saved: %s", path)


if __name__ == "__main__":
    main()
