"""Косинусная близость внутри классов vs между классами.

Важная оговорка, из-за которой сырой mean cosine сам по себе мало что значит:
у эмбеддингов трансформеров пространство анизотропно — все векторы смещены в
общий конус, поэтому любые два случайных текста уже имеют высокий косинус.
Поэтому здесь считается три вещи:
  1. сырые intra/inter косинусы (сопоставимо с тем, что репортят в статьях);
  2. те же величины после вычитания глобального среднего (anisotropy-correction);
  3. baseline "случайная пара" — чтобы было видно, от чего отсчитывать разрыв.
"""
from __future__ import annotations

import numpy as np

from ..utils import l2_normalize, subsample_indices


def _pair_values(sim: np.ndarray, same_set: bool) -> np.ndarray:
    """Значения косинусов без диагонали (для матрицы "класс сам с собой")."""
    if same_set:
        iu = np.triu_indices(sim.shape[0], k=1)
        return sim[iu]
    return sim.ravel()


def _describe(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "n_pairs": int(values.size),
    }


def cosine_analysis(
    x: np.ndarray,
    y: np.ndarray,
    sample_per_class: int = 2000,
    seed: int = 42,
    center: bool = True,
) -> dict[str, object]:
    """Возвращает статистики intra/inter косинусов (сырые и центрированные)."""
    rng = np.random.default_rng(seed)
    out: dict[str, object] = {}

    for variant in ("raw", "centered") if center else ("raw",):
        vecs = x - x.mean(axis=0, keepdims=True) if variant == "centered" else x
        vecs = l2_normalize(vecs.astype(np.float64))

        idx_safe = np.where(y == 0)[0]
        idx_harm = np.where(y == 1)[0]
        idx_safe = idx_safe[subsample_indices(len(idx_safe), sample_per_class, rng)]
        idx_harm = idx_harm[subsample_indices(len(idx_harm), sample_per_class, rng)]

        s, h = vecs[idx_safe], vecs[idx_harm]
        intra_safe = _pair_values(s @ s.T, same_set=True)
        intra_harm = _pair_values(h @ h.T, same_set=True)
        inter = _pair_values(s @ h.T, same_set=False)

        all_idx = np.concatenate([idx_safe, idx_harm])
        rnd = vecs[all_idx]
        baseline = _pair_values(rnd @ rnd.T, same_set=True)

        intra_mean = 0.5 * (intra_safe.mean() + intra_harm.mean())
        pooled_std = float(np.sqrt(0.5 * (intra_safe.var() + intra_harm.var())))
        gap = float(intra_mean - inter.mean())

        out[variant] = {
            "intra_safe": _describe(intra_safe),
            "intra_harm": _describe(intra_harm),
            "inter": _describe(inter),
            "random_pair_baseline_mean": float(baseline.mean()),
            "gap_intra_minus_inter": gap,
            # Разрыв в единицах разброса: сравнимо между моделями с разной анизотропией.
            "gap_standardized": float(gap / pooled_std) if pooled_std > 0 else float("nan"),
            "centroid_cosine_safe_harm": float(
                np.dot(l2_normalize(s.mean(0)[None])[0], l2_normalize(h.mean(0)[None])[0])
            ),
        }

    return out


def similarity_samples(
    x: np.ndarray, y: np.ndarray, sample_per_class: int = 500, seed: int = 42, center: bool = False
) -> dict[str, np.ndarray]:
    """Наборы значений косинусов для гистограмм."""
    rng = np.random.default_rng(seed)
    vecs = x - x.mean(axis=0, keepdims=True) if center else x
    vecs = l2_normalize(vecs.astype(np.float64))

    idx_safe = np.where(y == 0)[0]
    idx_harm = np.where(y == 1)[0]
    idx_safe = idx_safe[subsample_indices(len(idx_safe), sample_per_class, rng)]
    idx_harm = idx_harm[subsample_indices(len(idx_harm), sample_per_class, rng)]
    s, h = vecs[idx_safe], vecs[idx_harm]

    return {
        "safe–safe": _pair_values(s @ s.T, True),
        "harm–harm": _pair_values(h @ h.T, True),
        "safe–harm": _pair_values(s @ h.T, False),
    }


def flatten_cosine_row(model: str, split: str, stats: dict) -> list[dict]:
    """Разворачивает результат cosine_analysis в строки для CSV."""
    rows = []
    for variant, st in stats.items():
        rows.append(
            {
                "model": model,
                "split": split,
                "variant": variant,
                "cos_intra_safe": st["intra_safe"]["mean"],
                "cos_intra_harm": st["intra_harm"]["mean"],
                "cos_inter": st["inter"]["mean"],
                "cos_random_pair": st["random_pair_baseline_mean"],
                "gap": st["gap_intra_minus_inter"],
                "gap_std_units": st["gap_standardized"],
                "centroid_cos": st["centroid_cosine_safe_harm"],
            }
        )
    return rows
