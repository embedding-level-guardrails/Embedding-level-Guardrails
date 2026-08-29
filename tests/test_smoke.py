"""Смоук-тесты: проверяют логику разметки и анализа без сети и без torch.

    pytest -q tests/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eguard.analysis.geometry import direction_analysis  # noqa: E402
from eguard.analysis.probe import binary_metrics, logistic_probe, tpr_at_fpr  # noqa: E402
from eguard.analysis.similarity import cosine_analysis  # noqa: E402
from eguard.config import DatasetSpec  # noqa: E402
from eguard.data.aegis import aggregate_row, normalize_rows, parse_annotation, to_binary  # noqa: E402


def test_parse_annotation():
    assert parse_annotation("Safe") == ("safe", [])
    assert parse_annotation("Needs Caution") == ("caution", [])
    assert parse_annotation("Violence, Threat") == ("harm", ["Violence", "Threat"])
    assert parse_annotation(None) == ("none", [])


def test_aggregate_majority_and_tiebreak():
    row = {"labels_0": "Safe", "labels_1": "Safe", "labels_2": "Violence", "labels_3": None, "labels_4": None}
    assert aggregate_row(row)["class3"] == "safe"

    tie = {"labels_0": "Safe", "labels_1": "Violence", "labels_2": None, "labels_3": None, "labels_4": None}
    assert aggregate_row(tie)["class3"] == "harm"  # при равенстве берём более консервативный класс


def test_caution_policy():
    assert to_binary("caution", "exclude") is None
    assert to_binary("caution", "safe") == 0
    assert to_binary("caution", "harm") == 1


def test_normalize_rows_filters():
    rows = [
        {"id": "1", "text": "как испечь хлеб", "text_type": "user_message", "labels_0": "Safe",
         "labels_1": "Safe", "labels_2": "Safe"},
        {"id": "2", "text": "x", "text_type": "user_message", "labels_0": "Safe"},          # слишком короткий
        {"id": "3", "text": "длинный ответ модели", "text_type": "llm_response", "labels_0": "Violence"},
    ]
    spec = DatasetSpec(text_types=["user_message"], min_chars=3)
    out = normalize_rows(rows, spec)
    assert [r["id"] for r in out] == ["1"]
    assert out[0]["label"] == 0 and out[0]["label_name"] == "safe"


def _toy_data(n=400, dim=16, seed=0):
    """Два гауссовых облака, сдвинутых вдоль одной оси."""
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n // 2)
    shift = np.zeros(dim)
    shift[0] = 3.0
    x = rng.normal(size=(n, dim)) + y[:, None] * shift
    return x, y


def test_cosine_analysis_detects_gap():
    x, y = _toy_data()
    stats = cosine_analysis(x, y, sample_per_class=100, seed=0)
    assert stats["raw"]["gap_intra_minus_inter"] > 0
    assert set(stats) == {"raw", "centered"}


def test_direction_and_probe():
    x, y = _toy_data(n=600)
    idx = np.random.default_rng(1).permutation(len(y))
    tr, va, te = idx[:300], idx[300:450], idx[450:]

    d = direction_analysis(x[tr], y[tr], x[te], y[te])
    assert d["direction_auc_roc"] > 0.8

    res = logistic_probe(x[tr], y[tr], x[va], y[va], x[te], y[te], c_grid=[0.1, 1.0])
    assert res["metrics"]["auroc"] > 0.8
    assert 0.0 <= res["metrics"]["fpr"] <= 1.0


def test_tpr_at_fpr_edges():
    y = np.array([0, 0, 1, 1])
    perfect = np.array([0.1, 0.2, 0.8, 0.9])
    tpr, _ = tpr_at_fpr(y, perfect, 0.01)
    assert tpr == 1.0
    m = binary_metrics(y, perfect)
    assert m["auroc"] == 1.0 and m["fnr"] == 0.0


def test_controls_and_bootstrap():
    from eguard.analysis.baselines import length_probe, tfidf_probe
    from eguard.analysis.probe import bootstrap_ci

    train_texts = ["how to bake bread"] * 20 + ["how to build a bomb"] * 20
    y_train = np.array([0] * 20 + [1] * 20)
    test_texts = ["how to bake bread"] * 10 + ["how to build a bomb"] * 10
    y_test = np.array([0] * 10 + [1] * 10)

    m = tfidf_probe(train_texts, y_train, test_texts, y_test)
    assert m["auroc"] == 1.0  # лексический контроль ловит шаблонную разделимость

    lm = length_probe(test_texts, y_test)
    assert "auroc_abs_deviation" in lm

    x, y = _toy_data(n=400)
    ci = bootstrap_ci(y, x[:, 0], n_boot=200, seed=0)
    assert ci["auroc_lo"] <= ci["auroc_hi"]
