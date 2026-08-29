"""Шаг 3 (RQ1): линейный проб + kNN + centroid-distance поверх frozen-эмбеддингов.

    python scripts/03_rq1_probe.py --config configs/rq1.yaml

На выходе results/rq1/probe.csv со столбцами auroc / auprc / f1 / fpr / fnr /
tpr_at_fpr_0.01 для каждой модели и каждого типа проба. Для Prompt Guard
дополнительно считается строка native_head — качество его собственной головы
на тех же данных.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eguard.analysis.baselines import length_probe, tfidf_probe  # noqa: E402
from eguard.analysis.probe import (  # noqa: E402
    binary_metrics,
    bootstrap_ci,
    centroid_distance_probe,
    knn_probe,
    logistic_probe,
)
from eguard.config import load_config  # noqa: E402
from eguard.data import load_split  # noqa: E402
from eguard.embeddings import available_encoders, load_xy  # noqa: E402
from eguard.utils import ensure_dir, get_logger, set_seed  # noqa: E402

logger = get_logger("rq1_probe")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rq1.yaml")
    ap.add_argument("--encoders", nargs="*", default=None)
    ap.add_argument("--skip-knn", action="store_true", help="kNN is slow on big train")
    ap.add_argument("--skip-controls", action="store_true", help="do not calculate TF-IDF and length baseline")
    ap.add_argument("--n-boot", type=int, default=1000, help="iterations of bootstrap; 0 — turn off CI")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    probe_cfg = cfg.analysis.get("probe", {})
    normalize = cfg.analysis.get("normalize", True)
    target_fpr = probe_cfg.get("fpr_target", 0.01)

    keys = args.encoders or available_encoders(cfg.paths.embeddings, cfg.dataset.name)
    if not keys:
        raise SystemExit("No calculated embeddings — first scripts/01_embed.py")

    out_dir = ensure_dir(Path(cfg.paths.results) / "rq1")
    rows = []

    for key in keys:
        x_tr, y_tr, rec_tr, _ = load_xy(cfg, key, "train")
        x_va, y_va, rec_va, _ = load_xy(cfg, key, "val")
        x_te, y_te, rec_te, scores_te = load_xy(cfg, key, "test")
        logger.info("%s: train=%d val=%d test=%d dim=%d", key, len(y_tr), len(y_va), len(y_te), x_tr.shape[1])

        res = logistic_probe(
            x_tr, y_tr, x_va, y_va, x_te, y_te,
            c_grid=probe_cfg.get("c_grid"), class_weight=probe_cfg.get("class_weight", "balanced"),
            normalize=normalize, target_fpr=target_fpr, seed=cfg.seed,
        )
        ci = bootstrap_ci(y_te, res["test_scores"], target_fpr, args.n_boot, seed=cfg.seed) if args.n_boot else {}
        rows.append({"model": key, "probe": "logreg", **res["metrics"], **ci})
        np.save(out_dir / f"test_scores_{key}.npy", res["test_scores"])

        rows.append({
            "model": key, "probe": "centroid_distance",
            **centroid_distance_probe(x_tr, y_tr, x_te, y_te, normalize=normalize, target_fpr=target_fpr),
        })

        if not args.skip_knn:
            rows.append({
                "model": key, "probe": f"knn_k{probe_cfg.get('knn_k', 15)}",
                **knn_probe(x_tr, y_tr, x_te, y_te, k=probe_cfg.get("knn_k", 15),
                            normalize=normalize, target_fpr=target_fpr),
            })

        if scores_te is not None:
            native = binary_metrics(y_te, scores_te, target_fpr=target_fpr)
            native_ci = bootstrap_ci(y_te, scores_te, target_fpr, args.n_boot, seed=cfg.seed) if args.n_boot else {}
            rows.append({"model": key, "probe": "native_head", **native, **native_ci})

    # Контроли считаются один раз: они не зависят от энкодера, но задают планку,
    # ниже которой любой проб на эмбеддингах ничего интересного не показывает.
    if not args.skip_controls:
        rec_tr = load_split(cfg.paths.processed, cfg.dataset.name, "train")
        rec_te = load_split(cfg.paths.processed, cfg.dataset.name, "test")
        y_tr = np.array([r["label"] for r in rec_tr])
        y_te = np.array([r["label"] for r in rec_te])
        txt_tr = [r["text"] for r in rec_tr]
        txt_te = [r["text"] for r in rec_te]
        rows.append({"model": "—", "probe": "control_tfidf",
                     **tfidf_probe(txt_tr, y_tr, txt_te, y_te, target_fpr=target_fpr, seed=cfg.seed)})
        rows.append({"model": "—", "probe": "control_length",
                     **length_probe(txt_te, y_te, target_fpr=target_fpr)})

    df = pd.DataFrame(rows)
    front = ["model", "probe", "auroc", "auroc_lo", "auroc_hi", "auprc", "f1",
             "fpr", "fnr", f"tpr_at_fpr_{target_fpr:g}"]
    front = [c for c in front if c in df.columns]
    cols = front + [c for c in df.columns if c not in front]
    df = df[cols].round(4)
    df.to_csv(out_dir / "probe.csv", index=False)

    print("\n=== Probes on frozen embeddings (test) ===")
    print(df[front].to_string(index=False))
    logger.info("Results: %s", out_dir / "probe.csv")


if __name__ == "__main__":
    main()
