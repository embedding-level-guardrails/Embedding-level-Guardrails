"""Low-FPR evaluation. Torch-free.

Convention everywhere: positive = harmful, flagged iff ``score >= threshold``.

A. Transferred operating threshold. For each (model, readout, seed) the
   threshold is chosen once on source ``calibration``: the highest TPR among
   thresholds with empirical FPR <= target (ties in TPR -> the higher
   threshold, i.e. the lower FPR). It is then applied unchanged to every test
   set (FPR, FNR, TPR, TP/FP/TN/FN). OOD FPR may exceed the target.
B. Test ROC. On each test set separately: the highest TPR among achievable
   thresholds with empirical FPR <= target. Uses test labels; its threshold is
   not an operating threshold. Plus ROC-AUC and average precision (step-wise
   AP from sklearn, not trapezoidal PR-AUC).

Tied scores: candidate thresholds are the distinct score values, so tied rows
are always flagged together; an operating point that would need to split a tie
is not achievable and is never reported.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

SOURCE_SPLITS = ("train", "validation", "calibration")
GROUP = ["protocol", "variant", "model", "readout", "seed"]


def _check(y_true: ArrayLike, y_score: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("y_true and y_score must be 1-D arrays of equal length")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Both classes must be present")
    if not np.all(np.isfinite(s)):
        raise ValueError("Scores must be finite")
    return y, s


def tpr_at_fpr(y_true: ArrayLike, y_score: ArrayLike, max_fpr: float = 0.01) -> dict[str, float]:
    """Best achievable ROC point with FPR <= max_fpr on this data."""
    y, s = _check(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y, s, drop_intermediate=False)
    ok = np.flatnonzero(fpr <= max_fpr)  # never empty: the first point is (0, 0) at +inf
    best = ok[np.argmax(tpr[ok])]  # argmax takes the first = highest threshold on ties
    return {"threshold": float(thresholds[best]), "tpr": float(tpr[best]), "fpr": float(fpr[best])}


def confusion_at_threshold(y_true: ArrayLike, y_score: ArrayLike, threshold: float) -> dict[str, float]:
    y, s = _check(y_true, y_score)
    flagged = s >= threshold
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    tn = int((~flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "fpr": fp / (fp + tn), "fnr": fn / (fn + tp), "tpr": tp / (tp + fn)}


def select_threshold(calibration: pd.DataFrame, target_fpr: float) -> float:
    """Operating threshold from source calibration predictions only."""
    split = set(calibration["split"])
    if split != {"calibration"}:
        raise ValueError(f"Threshold must be selected on calibration only, got splits {sorted(split)}")
    return tpr_at_fpr(calibration["label"], calibration["score"], target_fpr)["threshold"]


def split_metrics(y_true: ArrayLike, y_score: ArrayLike, threshold: float, target_fpr: float) -> dict[str, Any]:
    y, s = _check(y_true, y_score)
    return {
        "n": len(y), "harmful_rate": float(y.mean()), "threshold": float(threshold),
        **confusion_at_threshold(y, s, threshold),
        "tpr_at_fpr": tpr_at_fpr(y, s, target_fpr)["tpr"],
        "roc_auc": float(roc_auc_score(y, s)),
        "avg_precision": float(average_precision_score(y, s)),
    }


def _subsets(df: pd.DataFrame):
    yield "all", df
    adv = df["adversarial"]
    if adv.notna().any():
        for value, name in ((True, "adversarial"), (False, "vanilla")):
            sub = df[adv.map(lambda v, value=value: v is not None and not pd.isna(v) and bool(v) == value)]
            if sub["label"].nunique() == 2:
                yield name, sub


def evaluate_predictions(preds: pd.DataFrame, target_fpr: float) -> tuple[pd.DataFrame, dict[tuple, float]]:
    """One row per (group, test set, subset). Returns the table and the thresholds."""
    rows, thresholds = [], {}
    for key, g in preds.groupby(GROUP, sort=True):
        threshold = select_threshold(g[g["split"] == "calibration"], target_fpr)
        thresholds[key] = threshold
        for test_set, t in g[~g["split"].isin(SOURCE_SPLITS)].groupby("split", sort=True):
            for subset, sub in _subsets(t):
                rows.append({**dict(zip(GROUP, key)), "test_set": test_set, "subset": subset,
                             **split_metrics(sub["label"], sub["score"], threshold, target_fpr)})
    return pd.DataFrame(rows), thresholds


BOOT_METRICS = ("fpr", "fnr", "tpr_at_fpr", "roc_auc", "avg_precision")


def _boot_metrics(y: np.ndarray, s: np.ndarray, threshold: float, target_fpr: float) -> dict[str, float]:
    if len(np.unique(y)) < 2:
        return {m: np.nan for m in BOOT_METRICS}
    m = split_metrics(y, s, threshold, target_fpr)
    return {k: m[k] for k in BOOT_METRICS}


def paired_bootstrap(
    preds: pd.DataFrame, thresholds: dict[tuple, float], target_fpr: float,
    n_boot: int, seed: int, readout: str = "linear_probe", reference: str = "base",
) -> pd.DataFrame:
    """Per (seed, test set): percentile 95% intervals of each model's metrics and
    of paired differences (model - reference, and between fine-tuned modes),
    resampling test examples with the same indices for every model. Thresholds
    stay fixed (calibration uncertainty is not included). Test sets hold one
    example per family, so example and family resampling coincide.

    This is test-set sampling uncertainty for one training run; spread across
    training seeds is reported separately in the seed summary."""
    rng = np.random.default_rng(seed)
    out = []
    sub = preds[(preds["readout"] == readout) & ~preds["split"].isin(SOURCE_SPLITS)]
    for (protocol, variant, run_seed, test_set), g in sub.groupby(["protocol", "variant", "seed", "split"], sort=True):
        wide = g.pivot_table(index="sample_id", columns="model", values="score")
        labels = g.drop_duplicates("sample_id").set_index("sample_id")["label"].loc[wide.index].to_numpy()
        models = list(wide.columns)
        thr = {m: thresholds[(protocol, variant, m, readout, run_seed)] for m in models}
        idx = rng.integers(0, len(wide), size=(n_boot, len(wide)))
        boots = {m: pd.DataFrame([_boot_metrics(labels[i], wide[m].to_numpy()[i], thr[m], target_fpr) for i in idx])
                 for m in models}
        comparisons = [(m, None) for m in models]
        comparisons += [(m, reference) for m in models if m != reference and reference in models]
        finetuned = [m for m in models if m != reference]
        comparisons += [(a, b) for k, a in enumerate(finetuned) for b in finetuned[k + 1:]]
        for a, b in comparisons:
            for metric in BOOT_METRICS:
                values = boots[a][metric] - (boots[b][metric] if b else 0.0)
                lo, hi = np.nanpercentile(values, [2.5, 97.5])
                out.append({"protocol": protocol, "variant": variant, "readout": readout, "seed": run_seed,
                            "test_set": test_set, "model": a, "minus": b or "", "metric": metric,
                            "ci_low": float(lo), "ci_high": float(hi), "n_boot": n_boot})
    return pd.DataFrame(out)
