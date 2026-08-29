"""Визуализация embedding space."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

from ..utils import ensure_dir, get_logger, l2_normalize  # noqa: E402

logger = get_logger(__name__)

SAFE_COLOR = "#2f6fdb"
HARM_COLOR = "#d1495b"


def project(x: np.ndarray, method: str, seed: int = 42) -> np.ndarray | None:
    """[n, d] -> [n, 2]. Возвращает None, если метод недоступен."""
    x = l2_normalize(x.astype(np.float64))
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(x)
    if method == "tsne":
        perplexity = min(30, max(5, (len(x) - 1) // 4))
        # PCA-препроцессинг: стандартная практика, ускоряет и стабилизирует t-SNE.
        x50 = PCA(n_components=min(50, x.shape[1], len(x) - 1), random_state=seed).fit_transform(x)
        return TSNE(n_components=2, perplexity=perplexity, init="pca",
                    random_state=seed, metric="cosine").fit_transform(x50)
    if method == "umap":
        try:
            import umap
        except ImportError:
            logger.warning("umap-learn не установлен, пропускаю UMAP (pip install umap-learn)")
            return None
        return umap.UMAP(n_components=2, metric="cosine", random_state=seed,
                         n_neighbors=15, min_dist=0.1).fit_transform(x)
    raise ValueError(f"Неизвестный метод проекции: {method}")


def plot_projection(
    coords: np.ndarray, labels: np.ndarray, title: str, out_path: str | Path,
    categories: list[str] | None = None,
) -> Path:
    """Два подграфика: раскраска по safe/harm и по harm-категории."""
    ncols = 2 if categories is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6), squeeze=False)

    ax = axes[0][0]
    for value, name, color in ((0, "safe", SAFE_COLOR), (1, "harm", HARM_COLOR)):
        m = labels == value
        ax.scatter(coords[m, 0], coords[m, 1], s=6, alpha=0.45, c=color, label=f"{name} (n={m.sum()})")
    ax.legend(markerscale=3, frameon=False)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])

    if categories is not None:
        ax2 = axes[0][1]
        cats = np.array(categories)
        top = [c for c, _ in sorted(
            ((c, int((cats == c).sum())) for c in set(cats)), key=lambda kv: -kv[1])][:8]
        cmap = plt.get_cmap("tab10")
        for i, cat in enumerate(top):
            m = cats == cat
            ax2.scatter(coords[m, 0], coords[m, 1], s=6, alpha=0.5, color=cmap(i % 10), label=f"{cat}")
        rest = ~np.isin(cats, top)
        if rest.any():
            ax2.scatter(coords[rest, 0], coords[rest, 1], s=4, alpha=0.2, color="#999999", label="other")
        ax2.legend(markerscale=3, frameon=False, fontsize=7, loc="best")
        ax2.set_title(f"{title} — по категориям")
        ax2.set_xticks([]); ax2.set_yticks([])

    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_similarity_hist(samples: dict[str, np.ndarray], title: str, out_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"safe–safe": SAFE_COLOR, "harm–harm": HARM_COLOR, "safe–harm": "#6c757d"}
    for name, values in samples.items():
        ax.hist(values, bins=80, alpha=0.5, density=True, label=f"{name} (μ={values.mean():.3f})",
                color=colors.get(name))
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(frameon=False)
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
