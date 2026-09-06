"""Тесты лоссов и трекинга RQ3 (нужен torch, mlflow не требуется).

    pytest -q tests/test_training.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")

from eguard.training.losses import (  # noqa: E402
    embedding_kl_regularizer,
    info_nce,
    supervised_contrastive,
    triplet_margin,
)
from eguard.training.tracking import NullRun, flatten_params, start_run  # noqa: E402


def _two_clusters(n=8, dim=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    half = n // 2
    base = torch.randn(2, dim, generator=g)
    emb = torch.cat([base[0].repeat(half, 1), base[1].repeat(half, 1)])
    emb = emb + 0.01 * torch.randn(n, dim, generator=g)
    labels = torch.tensor([0] * half + [1] * half)
    return emb, labels


def test_info_nce_rewards_alignment():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(8, 16, generator=g)
    n = torch.randn(8, 16, generator=g)
    aligned = info_nce(a, a.clone(), n)
    random = info_nce(a, torch.randn(8, 16, generator=g), n)
    assert aligned < random


def test_info_nce_gradients_are_finite():
    g = torch.Generator().manual_seed(1)
    a = torch.randn(8, 16, generator=g, requires_grad=True)
    p = torch.randn(8, 16, generator=g)
    n = torch.randn(8, 16, generator=g)
    info_nce(a, p, n).backward()
    assert torch.isfinite(a.grad).all()


def test_supcon_is_finite_and_lower_for_separated_classes():
    """Регресс-тест: -inf на диагонали умножался на нулевую маску и давал NaN."""
    emb, labels = _two_clusters()
    emb.requires_grad_(True)
    loss = supervised_contrastive(emb, labels)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(emb.grad).all()

    g = torch.Generator().manual_seed(3)
    mixed = supervised_contrastive(torch.randn(8, 16, generator=g), labels)
    assert loss < mixed


def test_supcon_handles_batch_without_positives():
    g = torch.Generator().manual_seed(0)
    loss = supervised_contrastive(torch.randn(4, 16, generator=g), torch.tensor([0, 1, 2, 3]))
    assert torch.isfinite(loss)


def test_triplet_margin_is_zero_when_ordering_is_correct():
    a = torch.tensor([[1.0, 0.0]])
    p = torch.tensor([[1.0, 0.0]])
    n = torch.tensor([[-1.0, 0.0]])
    assert triplet_margin(a, p, n, margin=0.2).item() == pytest.approx(0.0)
    assert triplet_margin(a, n, p, margin=0.2).item() > 0


def test_kl_regularizer_is_zero_against_itself():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(6, 12, generator=g)
    assert embedding_kl_regularizer(x, x).item() == pytest.approx(0.0, abs=1e-6)


def test_flatten_params_flattens_nested_config():
    flat = flatten_params({"training": {"loss": "info_nce", "lr": 2e-5}, "types": ["a", "b"]})
    assert flat == {"training.loss": "info_nce", "training.lr": 2e-5, "types": "a, b"}


def test_tracking_falls_back_to_noop_when_disabled():
    run = start_run({"enabled": False}, run_name="x")
    assert isinstance(run, NullRun) and run.enabled is False
    # интерфейс не должен падать: пайплайн работает без mlflow
    run.log_params({"a": 1}); run.log_metrics({"m": 1.0}, step=1); run.finish()


def test_evaluate_uses_centroid_score_without_head():
    """Без головы скор считается по центроидам train — тем же, что в RQ5."""
    from eguard.analysis.probe import binary_metrics
    from eguard.utils import l2_normalize

    rng = np.random.default_rng(0)
    x_train = np.vstack([rng.normal(0, 0.1, (20, 8)), rng.normal(3, 0.1, (20, 8))])
    y_train = np.array([0] * 20 + [1] * 20)
    x_eval = np.vstack([rng.normal(0, 0.1, (10, 8)), rng.normal(3, 0.1, (10, 8))])
    y_eval = np.array([0] * 10 + [1] * 10)

    mu_safe = l2_normalize(x_train[y_train == 0].mean(0)[None])[0]
    mu_harm = l2_normalize(x_train[y_train == 1].mean(0)[None])[0]
    scores = x_eval @ mu_harm - x_eval @ mu_safe
    assert binary_metrics(y_eval, scores, threshold=0.0)["auroc"] > 0.9


def test_next_cycled_restarts_exhausted_loader():
    """Регресс: в joint загрузчик текстов короче загрузчика пар и кончался посреди эпохи."""
    from eguard.training.loop import _next_cycled

    loader = [1, 2]                      # короткий "загрузчик"
    iterator = iter(loader)
    seen = []
    for _ in range(5):                   # шагов больше, чем батчей
        item, iterator = _next_cycled(iterator, loader)
        seen.append(item)
    assert seen == [1, 2, 1, 2, 1]


def test_joint_step_count_uses_pairs_and_cycles_texts():
    """Число шагов берётся по парам; текстов меньше — они должны циклиться, а не падать."""
    from torch.utils.data import DataLoader

    from eguard.training.data import TextDataset, collate_texts
    from eguard.training.loop import _next_cycled

    texts = [{"text": f"t{i}", "label": i % 2} for i in range(8)]
    text_loader = DataLoader(TextDataset(texts), batch_size=4, collate_fn=collate_texts)
    assert len(text_loader) == 2

    iterator = iter(text_loader)
    for _ in range(6):                   # имитируем 6 шагов по парам при 2 батчах текстов
        batch, iterator = _next_cycled(iterator, text_loader)
        assert len(batch["text"]) == 4
