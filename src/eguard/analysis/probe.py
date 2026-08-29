"""Линейный проб поверх замороженных эмбеддингов + метрики.

Метрики намеренно шире, чем accuracy/F1: для guardrail важна рабочая точка с
низким FPR, поэтому основной репортируемый показатель — TPR@FPR=1%, плюс явные
FPR и FNR. Рядом стоит kNN как нелинейная верхняя граница: если kNN заметно
лучше линейного проба, значит safety-сигнал в пространстве есть, но он не
линейно отделим — это и есть аргумент в пользу contrastive-дообучения (RQ3).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import KNeighborsClassifier

from ..utils import l2_normalize


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 0.01) -> tuple[float, float]:
    """Максимальный TPR при FPR <= target. Возвращает (tpr, threshold)."""
    fpr, tpr, thr = roc_curve(y_true, scores)
    ok = np.where(fpr <= target_fpr)[0]
    if len(ok) == 0:
        return 0.0, float("inf")
    i = ok[np.argmax(tpr[ok])]
    return float(tpr[i]), float(thr[i])


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5,
                   target_fpr: float = 0.01) -> dict[str, float]:
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tpr_low_fpr, thr_low_fpr = tpr_at_fpr(y_true, scores, target_fpr)

    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
        "accuracy": float((tp + tn) / max(len(y_true), 1)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "fpr": float(fp / max(fp + tn, 1)),
        "fnr": float(fn / max(fn + tp, 1)),
        f"tpr_at_fpr_{target_fpr:g}": tpr_low_fpr,
        "threshold_at_target_fpr": thr_low_fpr,
        "n_eval": int(len(y_true)),
        "harm_rate": float(y_true.mean()),
    }


def bootstrap_ci(
    y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 0.01,
    n_boot: int = 1000, alpha: float = 0.05, seed: int = 42,
) -> dict[str, float]:
    """Перцентильные CI для AUC ROC и TPR@FPR стратифицированным бутстрэпом.

    Нужны потому, что после фильтров test-сплит AEGIS — несколько сотен примеров,
    и разница в третьем знаке между моделями там ничего не значит.
    """
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y_true == 1)[0]
    idx_neg = np.where(y_true == 0)[0]
    auc_rocs, tprs = [], []

    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(idx_pos, size=len(idx_pos), replace=True),
            rng.choice(idx_neg, size=len(idx_neg), replace=True),
        ])
        yb, sb = y_true[idx], scores[idx]
        if yb.min() == yb.max():
            continue
        auc_rocs.append(roc_auc_score(yb, sb))
        tprs.append(tpr_at_fpr(yb, sb, target_fpr)[0])

    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "auroc_lo": float(np.percentile(auc_rocs, lo)),
        "auroc_hi": float(np.percentile(auc_rocs, hi)),
        f"tpr_at_fpr_{target_fpr:g}_lo": float(np.percentile(tprs, lo)),
        f"tpr_at_fpr_{target_fpr:g}_hi": float(np.percentile(tprs, hi)),
        "n_boot": int(len(auc_rocs)),
    }


def _prep(x: np.ndarray, normalize: bool) -> np.ndarray:
    x = x.astype(np.float64)
    return l2_normalize(x) if normalize else x


def logistic_probe(
    x_train: np.ndarray, y_train: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray,
    c_grid: list[float] | None = None,
    class_weight: str | None = "balanced",
    normalize: bool = True,
    target_fpr: float = 0.01,
    seed: int = 42,
) -> dict[str, object]:
    """L2-логрегрессия на frozen-эмбеддингах; C выбирается по AUC ROC на val."""
    c_grid = c_grid or [0.01, 0.1, 1.0, 10.0]
    xtr, xva, xte = (_prep(a, normalize) for a in (x_train, x_val, x_test))

    best_c, best_auc = c_grid[0], -np.inf
    for c in c_grid:
        clf = LogisticRegression(C=c, max_iter=3000, class_weight=class_weight, random_state=seed)
        clf.fit(xtr, y_train)
        auc = roc_auc_score(y_val, clf.predict_proba(xva)[:, 1])
        if auc > best_auc:
            best_c, best_auc = c, auc

    clf = LogisticRegression(C=best_c, max_iter=3000, class_weight=class_weight, random_state=seed)
    clf.fit(np.vstack([xtr, xva]), np.concatenate([y_train, y_val]))
    scores = clf.predict_proba(xte)[:, 1]

    metrics = binary_metrics(y_test, scores, target_fpr=target_fpr)
    metrics.update({"best_C": best_c, "val_auroc": float(best_auc)})
    return {"metrics": metrics, "test_scores": scores, "model": clf}


def knn_probe(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
    k: int = 15, normalize: bool = True, target_fpr: float = 0.01,
) -> dict[str, float]:
    xtr, xte = (_prep(a, normalize) for a in (x_train, x_test))
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance")
    knn.fit(xtr, y_train)
    scores = knn.predict_proba(xte)[:, 1]
    return binary_metrics(y_test, scores, target_fpr=target_fpr)


def centroid_distance_probe(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
    normalize: bool = True, target_fpr: float = 0.01,
) -> dict[str, float]:
    """Скор = cos(x, mu_harm) - cos(x, mu_safe). Прообраз distance-based guardrail (RQ5)."""
    xtr, xte = (_prep(a, normalize) for a in (x_train, x_test))
    mu_safe = l2_normalize(xtr[y_train == 0].mean(0)[None])[0]
    mu_harm = l2_normalize(xtr[y_train == 1].mean(0)[None])[0]
    scores = xte @ mu_harm - xte @ mu_safe
    return binary_metrics(y_test, scores, threshold=0.0, target_fpr=target_fpr)
