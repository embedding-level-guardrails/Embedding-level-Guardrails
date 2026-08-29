"""Шаг 6 (RQ1): error analysis — где именно frozen-пространство ошибается.

    python scripts/06_error_analysis.py --config configs/rq1.yaml

Две вещи, обе адресуют конкретные пробелы из литобзора:

1. `hard_pairs_{model}.csv` — пары safe/harm с максимальным косинусом между собой.
   Safe-Embed фиксирует, что готовые энкодеры систематически путают лексически
   похожие safe/harm пары, но не показывает, какие именно. Эта таблица — сырьё
   для hard negatives в RQ2.
2. `errors_{model}.csv` — топ ложных срабатываний и пропусков логистического проба
   при пороге, выставленном на FPR=1%. Именно эта рабочая точка интересна для
   продакшена, и именно её ошибки надо смотреть глазами.

Файлы содержат фрагменты исходных текстов AEGIS (в т.ч. вредоносных) —
это рабочие артефакты анализа, не публиковать вместе с репозиторием.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eguard.analysis.probe import tpr_at_fpr  # noqa: E402
from eguard.config import load_config  # noqa: E402
from eguard.embeddings import available_encoders, load_xy  # noqa: E402
from eguard.utils import ensure_dir, get_logger, l2_normalize, set_seed  # noqa: E402

logger = get_logger("error_analysis")

SNIPPET = 220


def snippet(text: str, n: int = SNIPPET) -> str:
    text = " ".join(text.split())
    return text[:n] + ("…" if len(text) > n else "")


def hard_pairs(x: np.ndarray, y: np.ndarray, records: list[dict], top_k: int = 50) -> pd.DataFrame:
    """Пары (safe, harm) с наибольшим косинусом: то, что энкодер считает почти одинаковым."""
    v = l2_normalize(x.astype(np.float64))
    idx_safe = np.where(y == 0)[0]
    idx_harm = np.where(y == 1)[0]
    sim = v[idx_safe] @ v[idx_harm].T          # [n_safe, n_harm]

    flat = np.argsort(sim, axis=None)[::-1][:top_k]
    rows = []
    for f in flat:
        i, j = np.unravel_index(f, sim.shape)
        s_rec, h_rec = records[idx_safe[i]], records[idx_harm[j]]
        rows.append({
            "cosine": round(float(sim[i, j]), 4),
            "harm_category": h_rec["category"],
            "safe_text": snippet(s_rec["text"]),
            "harm_text": snippet(h_rec["text"]),
            "safe_id": s_rec["id"],
            "harm_id": h_rec["id"],
        })
    return pd.DataFrame(rows)


def probe_errors(scores: np.ndarray, y: np.ndarray, records: list[dict],
                 target_fpr: float, top_k: int = 30) -> pd.DataFrame:
    """Ложные срабатывания и пропуски при пороге, выставленном на FPR=target."""
    _, threshold = tpr_at_fpr(y, scores, target_fpr)
    if not np.isfinite(threshold):
        threshold = float(np.quantile(scores, 1 - target_fpr))

    pred = (scores >= threshold).astype(int)
    rows = []
    for kind, mask, order in (
        ("false_positive", (pred == 1) & (y == 0), -1),   # safe, названный harm — цена для юзера
        ("false_negative", (pred == 0) & (y == 1), +1),   # пропущенный harm — цена для сервиса
    ):
        idx = np.where(mask)[0]
        idx = idx[np.argsort(order * scores[idx])][:top_k]
        for i in idx:
            rows.append({
                "kind": kind,
                "score": round(float(scores[i]), 4),
                "threshold": round(float(threshold), 4),
                "true_label": records[i]["label_name"],
                "category": records[i]["category"],
                "agreement": records[i]["agreement"],
                "text": snippet(records[i]["text"]),
                "id": records[i]["id"],
            })
    columns = ["kind", "score", "threshold", "true_label", "category", "agreement", "text", "id"]
    return pd.DataFrame(rows, columns=columns)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rq1.yaml")
    ap.add_argument("--encoders", nargs="*", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    target_fpr = cfg.analysis.get("probe", {}).get("fpr_target", 0.01)

    keys = args.encoders or available_encoders(cfg.paths.embeddings, cfg.dataset.name)
    if not keys:
        raise SystemExit("Нет посчитанных эмбеддингов — сначала scripts/01_embed.py")

    out_dir = ensure_dir(Path(cfg.paths.results) / "rq1" / "errors")
    summary = []

    for key in keys:
        x, y, records, _ = load_xy(cfg, key, args.split)

        pairs = hard_pairs(x, y, records, top_k=args.top_k)
        pairs.to_csv(out_dir / f"hard_pairs_{key}.csv", index=False)
        logger.info("%s: hard-пары, максимальный cos(safe, harm) = %.4f", key, pairs["cosine"].max())

        score_path = Path(cfg.paths.results) / "rq1" / f"test_scores_{key}.npy"
        n_fp = n_fn = None
        if score_path.exists() and args.split == "test":
            errors = probe_errors(np.load(score_path), y, records, target_fpr)
            errors.to_csv(out_dir / f"errors_{key}.csv", index=False)
            n_fp = int((errors["kind"] == "false_positive").sum())
            n_fn = int((errors["kind"] == "false_negative").sum())
        else:
            logger.info("%s: нет test_scores — сначала scripts/03_rq1_probe.py", key)

        summary.append({
            "model": key,
            "max_cross_class_cosine": round(float(pairs["cosine"].max()), 4),
            "mean_top_pair_cosine": round(float(pairs["cosine"].mean()), 4),
            "top_confused_category": pairs["harm_category"].mode().iat[0] if len(pairs) else None,
            "n_false_positives_listed": n_fp,
            "n_false_negatives_listed": n_fn,
        })

    df = pd.DataFrame(summary)
    df.to_csv(Path(cfg.paths.results) / "rq1" / "error_summary.csv", index=False)
    print("\n=== Error analysis ===")
    print(df.to_string(index=False))
    logger.info("Tables: %s", out_dir)


if __name__ == "__main__":
    main()
