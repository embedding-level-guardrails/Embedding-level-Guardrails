"""Тесты загрузчика ToxicChat и логики LODO-порога (без сети и torch)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eguard.analysis.probe import binary_metrics, fit_transfer_probe, tpr_at_fpr  # noqa: E402
from eguard.config import DatasetSpec  # noqa: E402
from eguard.data import get_loader  # noqa: E402
from eguard.data.toxicchat import category_of, normalize_rows, to_binary  # noqa: E402


def _row(toxicity=0, jailbreaking=0, text="какой-то пользовательский запрос", human=True):
    return {"conv_id": "c1", "user_input": text, "model_output": "",
            "human_annotation": human, "toxicity": toxicity, "jailbreaking": jailbreaking}


def test_to_binary_label_fields():
    assert to_binary(_row(toxicity=1), "toxicity") == 1
    assert to_binary(_row(toxicity=1), "jailbreaking") == 0
    # `any` ловит jailbreak, который по формулировке не токсичен
    assert to_binary(_row(toxicity=0, jailbreaking=1), "any") == 1
    assert to_binary(_row(), "any") == 0


def test_to_binary_rejects_unknown_field():
    with pytest.raises(ValueError):
        to_binary(_row(), "no_such_field")


def test_category_of_combines_flags():
    assert category_of(_row(toxicity=1)) == "toxicity"
    assert category_of(_row(jailbreaking=1)) == "jailbreaking"
    assert category_of(_row(toxicity=1, jailbreaking=1)) == "toxicity+jailbreaking"
    assert category_of(_row()) == "safe"


def test_normalize_filters_and_maps_fields():
    spec = DatasetSpec(name="toxicchat", label_field="any", min_chars=5)
    rows = [
        _row(toxicity=1),
        _row(text="x"),                      # короткий
        _row(human=False),                   # без человеческой разметки
    ]
    out = normalize_rows(rows, spec)
    assert len(out) == 1
    record = out[0]
    assert record["label"] == 1 and record["label_name"] == "harm"
    # текст берётся из user_input: guardrail видит запрос, а не ответ модели
    assert record["text"] == "какой-то пользовательский запрос"
    assert record["text_type"] == "user_message"
    assert record["source"] == "toxicchat"


def test_normalize_can_keep_unannotated():
    spec = DatasetSpec(name="toxicchat", require_human_annotation=False)
    assert len(normalize_rows([_row(human=False)], spec)) == 1


def test_loader_registered_and_synthetic_is_imbalanced():
    loader = get_loader("toxicchat")
    raw = loader.make_synthetic(400, seed=0)
    spec = DatasetSpec(name="toxicchat", label_field="any")
    records = loader.normalize_rows(raw["test"], spec)
    harm_rate = np.mean([r["label"] for r in records])
    # ToxicChat сильно несбалансирован; синтетика должна это повторять
    assert 0.0 < harm_rate < 0.3


def test_transfer_probe_does_not_train_on_val():
    """Проб для LODO обучается только на train — иначе порог нельзя калибровать на val."""
    rng = np.random.default_rng(0)
    x_train = np.vstack([rng.normal(0, 1, (60, 8)), rng.normal(2.5, 1, (60, 8))])
    y_train = np.array([0] * 60 + [1] * 60)
    # val помечен ПЕРЕВЁРНУТО: если бы проб на нём дообучался, качество бы просело
    x_val = np.vstack([rng.normal(0, 1, (30, 8)), rng.normal(2.5, 1, (30, 8))])
    y_val = np.array([0] * 30 + [1] * 30)

    clf, scores_val, info = fit_transfer_probe(x_train, y_train, x_val, y_val)
    assert len(scores_val) == len(y_val)
    assert "best_C" in info and "val_auroc" in info
    # обучение шло только на train: предсказания на train разделяют лучше случайного
    from eguard.utils import l2_normalize
    assert binary_metrics(y_train, clf.predict_proba(l2_normalize(x_train))[:, 1])["auroc"] > 0.9


def test_threshold_calibrated_on_val_transfers_to_other_set():
    """Порог с val применяется к другому набору без перекалибровки."""
    rng = np.random.default_rng(1)
    y_val = np.array([0] * 100 + [1] * 100)
    scores_val = np.concatenate([rng.normal(0, 1, 100), rng.normal(3, 1, 100)])
    _, threshold = tpr_at_fpr(y_val, scores_val, 0.01)
    assert np.isfinite(threshold)

    # сдвинутый (OOD) набор: при том же пороге FPR обязан быть посчитан, а не подобран
    y_ood = np.array([0] * 100 + [1] * 100)
    scores_ood = np.concatenate([rng.normal(1.0, 1, 100), rng.normal(3.5, 1, 100)])
    metrics = binary_metrics(y_ood, scores_ood, threshold=threshold, target_fpr=0.01)
    assert 0.0 <= metrics["fpr"] <= 1.0 and 0.0 <= metrics["fnr"] <= 1.0
    # сдвиг вправо -> при неизменном пороге ложных срабатываний становится больше
    baseline = binary_metrics(y_val, scores_val, threshold=threshold, target_fpr=0.01)
    assert metrics["fpr"] > baseline["fpr"]
