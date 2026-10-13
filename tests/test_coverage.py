"""Тесты сборки триплетов в режиме coverage (src/eguard/pairs/coverage.py) и
объективов SupCon на текстах (src/eguard/training/loop.py). Без сети и без модели."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eguard.pairs.coverage import (  # noqa: E402
    CoverageConfig,
    build_split_coverage,
    out_of_fold_harm_probability,
)


def _toy_pool(n_per_group: int = 40, dim: int = 16, seed: int = 0):
    """Четыре группы: safe/harm × vanilla/adversarial.

    Векторы устроены как у исходного e5: есть ось «класс» и ось «форма» (adversarial),
    и форма в сумме весит больше — поэтому adversarial-тексты разных классов похожи.
    """
    rng = np.random.default_rng(seed)
    records, vectors = [], []
    for label in (0, 1):
        for adversarial in (False, True):
            for k in range(n_per_group):
                v = rng.normal(0, 0.3, dim)
                v[0] += 1.0 if label else -1.0          # класс
                v[1] += 3.0 if adversarial else 0.0     # форма
                vectors.append(v)
                records.append({"id": f"{label}{int(adversarial)}-{k}",
                                "text": f"text {label} {int(adversarial)} {k}",
                                "label": label, "adversarial": adversarial,
                                "category": f"cat{k % 2}" if label else "safe"})
    return records, np.array(vectors)


def _build(records, x, **cfg):
    params = dict(benign_twin_fraction=0.5, top_k=3, search_depth=30, max_negative_uses=3,
                  max_positive_uses=3, noise_threshold=0.95, noise_folds=3)
    params.update(cfg)
    return build_split_coverage("train", records, x, np.random.default_rng(1), CoverageConfig(**params))


def test_every_record_is_anchor_exactly_once():
    records, x = _toy_pool()
    pairs, report = _build(records, x)
    anchors = Counter(p["anchor_id"] for p in pairs)
    assert set(anchors) == {r["id"] for r in records}
    assert max(anchors.values()) == 1
    assert report.texts_cover_pool                         # тексты триплетов == весь пул


def test_types_and_roles():
    records, x = _toy_pool()
    by_id = {r["id"]: r for r in records}
    pairs, _ = _build(records, x)
    for p in pairs:
        a, pos, neg = by_id[p["anchor_id"]], by_id[p["positive_id"]], by_id[p["negative_id"]]
        assert pos["label"] == a["label"] and neg["label"] != a["label"]
        if p["pair_type"] == "adversarial_contrast" and p["variant"] != "fallback":
            assert a["adversarial"] and not pos["adversarial"] and neg["adversarial"]
        elif p["pair_type"] == "benign_twin":
            assert a["label"] == 1 and not a["adversarial"]
        else:
            assert p["pair_type"] == "safe_harm_contrast" and not a["adversarial"]


def test_adversarial_negative_shares_form():
    """Негатив adversarial-якоря — из ближайших adversarial другого класса: та же форма."""
    records, x = _toy_pool()
    pairs, _ = _build(records, x)
    adv = [p for p in pairs if p["pair_type"] == "adversarial_contrast" and p["variant"] != "fallback"]
    assert adv
    # форма весит больше класса, поэтому у исходных векторов негатив ближе позитива
    assert np.mean([p["frozen_cos_negative"] > p["frozen_cos_positive"] for p in adv]) > 0.9


def test_reuse_caps_respected():
    records, x = _toy_pool()
    pairs, report = _build(records, x, max_negative_uses=2, max_positive_uses=2)
    mined_neg = Counter(p["negative_id"] for p in pairs if p["variant"] in ("nearest", "nearest_negative"))
    mined_pos = Counter(p["positive_id"] for p in pairs if p["variant"] == "nearest")
    assert max(mined_neg.values()) <= 2 and max(mined_pos.values(), default=0) <= 2
    assert report.max_negative_uses <= 2 and report.max_positive_uses <= 2


def test_suspected_mislabels_are_never_positives_or_negatives():
    """Текст с перевёрнутой меткой остаётся якорем, но не попадает в позитивы и негативы."""
    records, x = _toy_pool()
    victim = next(i for i, r in enumerate(records) if r["label"] == 1 and not r["adversarial"])
    records[victim] = {**records[victim], "label": 0, "category": "safe"}   # «безопасный», лежит среди вредных
    pairs, report = _build(records, x, noise_threshold=0.8)
    vid = records[victim]["id"]
    assert report.n_suspect >= 1
    assert all(vid not in (p["positive_id"], p["negative_id"]) for p in pairs)
    assert sum(p["anchor_id"] == vid for p in pairs) == 1


def test_out_of_fold_probability_flags_flipped_label():
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(-2, 1, (60, 8)), rng.normal(2, 1, (60, 8))])
    y = np.array([0] * 60 + [1] * 60)
    y[5] = 1                                                # ошибка разметки внутри кластера safe
    p = out_of_fold_harm_probability(x, y, folds=5)
    p_other = np.where(y == 1, 1 - p, p)
    assert p_other[5] > 0.9 and np.median(p_other) < 0.1


def test_build_is_deterministic():
    records, x = _toy_pool()
    first = [p["pair_id"] for p in _build(records, x)[0]]
    second = [p["pair_id"] for p in _build(records, x)[0]]
    assert first == second


def test_coverage_config_rejects_unknown_keys():
    with pytest.raises(ValueError):
        CoverageConfig.from_dict({"top_k": 5, "no_such_key": 1})


# --- объективы на текстах ----------------------------------------------------------

torch = pytest.importorskip("torch")


def test_text_losses_per_objective():
    from eguard.training.loop import TrainConfig, needs_head, needs_pairs, text_losses

    torch.manual_seed(0)
    emb = torch.randn(8, 16)
    labels = torch.tensor([0, 1] * 4)
    head = torch.nn.Linear(16, 2)

    _, ce = text_losses(emb, labels, TrainConfig(objective="classification"), head)
    _, sup = text_losses(emb, labels, TrainConfig(objective="supcon"))
    _, both = text_losses(emb, labels, TrainConfig(objective="supcon_joint"), head)
    assert set(ce) == {"loss_ce"}
    assert {"loss_supcon", "cos_margin"} <= set(sup) and "loss_ce" not in sup
    assert {"loss_ce", "loss_supcon"} <= set(both)

    assert needs_pairs("contrastive") and needs_pairs("joint")
    assert not needs_pairs("supcon") and not needs_pairs("supcon_joint")
    assert needs_head("supcon_joint") and not needs_head("supcon")


def test_supcon_loss_drops_when_classes_separate():
    from eguard.training.loop import TrainConfig, text_losses

    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    mixed = torch.randn(8, 16, generator=torch.Generator().manual_seed(1))
    separated = torch.cat([torch.ones(4, 16), -torch.ones(4, 16)]) + 0.01 * mixed
    cfg = TrainConfig(objective="supcon")
    assert text_losses(separated, labels, cfg)[0] < text_losses(mixed, labels, cfg)[0]
