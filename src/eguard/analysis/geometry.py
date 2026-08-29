"""Геометрия embedding space вокруг метки safe/harm.

Отвечает на две вещи, которых нет в разобранной литературе:
  * есть ли вообще кластеры и совпадают ли они с safety-меткой
    (silhouette + KMeans/ARI, а не только средний косинус);
  * насколько сигнал одномерен — то есть существует ли единое «harm-направление»
    (проекция на разницу центроидов против полноразмерного линейного пробa).
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    roc_auc_score,
    silhouette_score,
)

from ..utils import l2_normalize, subsample_indices


def anisotropy(x: np.ndarray, sample: int = 2000, seed: int = 42) -> dict[str, float]:
    """Средний косинус случайных пар + доля дисперсии в PC1."""
    rng = np.random.default_rng(seed)
    idx = subsample_indices(len(x), sample, rng)
    v = l2_normalize(x[idx].astype(np.float64))
    sim = v @ v.T
    iu = np.triu_indices(sim.shape[0], k=1)
    pca = PCA(n_components=min(10, v.shape[1], len(idx) - 1)).fit(x[idx])
    return {
        "mean_random_cosine": float(sim[iu].mean()),
        "pc1_explained_var": float(pca.explained_variance_ratio_[0]),
        "top10_explained_var": float(pca.explained_variance_ratio_.sum()),
    }


def direction_analysis(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, y_eval: np.ndarray) -> dict[str, float]:
    """Единое harm-направление d = mu_harm - mu_safe, оценённое на train.

    Проецируем eval на d и смотрим AUC ROC: сколько safety-сигнала лежит в одной оси.
    Сравнение этого числа с AUC ROC логистического проба показывает, линейно-одномерна
    ли разделимость или требуется полноразмерная граница.
    """
    mu_safe = x_train[y_train == 0].mean(0)
    mu_harm = x_train[y_train == 1].mean(0)
    d = mu_harm - mu_safe
    norm = np.linalg.norm(d)
    if norm == 0:
        return {"direction_auc_roc": float("nan"), "direction_cohens_d": float("nan"),
                "direction_norm": 0.0, "direction_vs_pc1_cos": float("nan")}

    d_unit = d / norm
    proj = x_eval @ d_unit
    a, b = proj[y_eval == 1], proj[y_eval == 0]
    pooled = np.sqrt(0.5 * (a.var() + b.var()))

    pc1 = PCA(n_components=1).fit(x_train).components_[0]
    return {
        "direction_auc_roc": float(roc_auc_score(y_eval, proj)),
        "direction_cohens_d": float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan"),
        "direction_norm": float(norm),
        "direction_vs_pc1_cos": float(abs(np.dot(d_unit, pc1))),
    }


def cluster_analysis(x: np.ndarray, y: np.ndarray, k_values: list[int], seed: int = 42,
                     sample: int = 5000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    idx = subsample_indices(len(x), sample, rng)
    xs, ys = x[idx], y[idx]

    out: dict[str, float] = {
        "silhouette_label_cosine": float(silhouette_score(xs, ys, metric="cosine")),
        "davies_bouldin_label": float(davies_bouldin_score(xs, ys)),
    }
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(xs)
        out[f"ari_k{k}"] = float(adjusted_rand_score(ys, km.labels_))
        out[f"nmi_k{k}"] = float(normalized_mutual_info_score(ys, km.labels_))
    return out


def geometry_report(
    x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, y_eval: np.ndarray,
    k_values: list[int] | None = None, seed: int = 42,
) -> dict[str, float]:
    k_values = k_values or [2, 8]
    return {
        **anisotropy(x_eval, seed=seed),
        **direction_analysis(x_train, y_train, x_eval, y_eval),
        **cluster_analysis(x_eval, y_eval, k_values, seed=seed),
    }
