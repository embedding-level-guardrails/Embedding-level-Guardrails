import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def _validate_predictions(
    y_true: ArrayLike, y_score: ArrayLike,
) -> tuple[NDArray, NDArray]:
    labels = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=float)
    if scores.ndim == 2 and scores.shape[1] == 2:
        scores = scores[:, 1]
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("Expected labels (N,) and scores (N,) or (N, 2).")
    if labels.size == 0 or not np.all(np.isin(labels, [0, 1])):
        raise ValueError("y_true must contain nonempty binary labels 0 and 1.")
    if np.unique(labels).size != 2:
        raise ValueError("Both classes must be present to calculate these metrics.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("y_score must contain only finite values.")
    return labels, scores


def pr_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    return pr_curve_stats(y_true, y_score)["auc"]


def fpr_fnr(
    y_true: ArrayLike,
    y_score: ArrayLike
) -> dict[str, NDArray]:
    labels, scores = _validate_predictions(y_true, y_score)
    # FPR = (FP / (FP + TN))
    # TPR = (TP / (TP + FN))
    # FNR = (FN / (TP + FN))
    fpr, tpr, grid = roc_curve(labels, scores, drop_intermediate=False)
    return {"thresholds": grid, "fpr": fpr, "fnr": 1.0 - tpr, "tpr": tpr}


def tpr_at_fpr(
    y_true: ArrayLike,
    y_score: ArrayLike,
    max_fpr: float = 0.01
) -> dict[str, float]:
    if not np.isfinite(max_fpr) or not 0.0 <= max_fpr <= 1.0:
        raise ValueError("max_fpr must be finite and between 0 and 1.")
    rates = fpr_fnr(y_true, y_score)
    candidates = np.flatnonzero(rates["fpr"] <= max_fpr)
    if candidates.size == 0:
        raise ValueError("No threshold satisfies the requested FPR constraint.")
    best = max(
        candidates,
        key=lambda i: (rates["tpr"][i], -rates["fpr"][i], rates["thresholds"][i]),
    )
    return {
        "threshold": float(rates["thresholds"][best]),
        "fpr": float(rates["fpr"][best]),
        "fnr": float(rates["fnr"][best]),
        "tpr": float(rates["tpr"][best]),
    }


def roc_curve_stats(y_true: ArrayLike, y_score: ArrayLike) -> dict[str, NDArray | float]:
    rates = fpr_fnr(y_true, y_score)
    return {**rates, "auc": float(auc(rates["fpr"], rates["tpr"]))}


def pr_curve_stats(y_true: ArrayLike, y_score: ArrayLike) -> dict[str, NDArray | float]:
    labels, scores = _validate_predictions(y_true, y_score)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    return {
        "precision": precision,
        "recall": recall,
        "thresholds": np.append(thresholds, np.nan),
        "auc": float(auc(recall, precision)),
        "prevalence": float(labels.mean()),
    }
