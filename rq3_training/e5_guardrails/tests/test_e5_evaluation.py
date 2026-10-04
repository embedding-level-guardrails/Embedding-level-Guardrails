import numpy as np
import pandas as pd
import pytest
from e5_guardrails.evaluation import (
    confusion_at_threshold,
    evaluate_predictions,
    paired_bootstrap,
    select_threshold,
    tpr_at_fpr,
)


def test_confusion_uses_score_ge_threshold_and_ties_flag_together():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.5, 0.5, 0.5, 0.9, 0.2])
    c = confusion_at_threshold(y, s, 0.5)
    assert (c["tp"], c["fp"], c["tn"], c["fn"]) == (2, 2, 1, 1)
    assert c["fpr"] == pytest.approx(2 / 3)
    assert c["fnr"] == pytest.approx(1 / 3)
    assert c["tpr"] == pytest.approx(2 / 3)


def test_tpr_at_fpr_never_splits_a_tie():
    # 100 negatives; one negative ties with two positives at 0.8.
    y = np.array([0] * 100 + [1, 1, 1])
    s = np.array([0.0] * 99 + [0.8] + [0.8, 0.8, 0.95])
    r = tpr_at_fpr(y, s, max_fpr=0.005)  # threshold 0.8 would give FPR 0.01 > 0.005
    assert r["threshold"] == 0.95 and r["tpr"] == pytest.approx(1 / 3) and r["fpr"] == 0.0
    r = tpr_at_fpr(y, s, max_fpr=0.01)
    assert r["threshold"] == 0.8 and r["tpr"] == 1.0 and r["fpr"] == pytest.approx(0.01)


def test_tpr_at_fpr_can_be_zero_when_no_threshold_qualifies():
    y = np.array([0, 1])
    s = np.array([0.9, 0.1])
    r = tpr_at_fpr(y, s, 0.01)
    assert r["tpr"] == 0.0 and np.isinf(r["threshold"])


def _preds(ood_labels):
    rng = np.random.default_rng(0)
    rows = []
    for split, labels in (("calibration", [0] * 200 + [1] * 50), ("toxicchat", ood_labels)):
        labels = np.array(labels)
        scores = labels * 0.5 + rng.random(len(labels))
        for i, (y, s) in enumerate(zip(labels, scores)):
            rows.append({"protocol": "p", "variant": "main", "model": "base", "readout": "linear_probe",
                         "seed": 0, "split": split, "sample_id": f"{split}{i}", "label": int(y),
                         "score": float(s), "adversarial": None})
    return pd.DataFrame(rows)


def test_threshold_comes_from_calibration_and_is_not_reselected_on_ood():
    a = _preds([0] * 100 + [1] * 30)
    b = a.copy()
    flip = b["split"] == "toxicchat"
    b.loc[flip, "label"] = 1 - b.loc[flip, "label"]  # OOD labels must not influence the threshold
    ra, ta = evaluate_predictions(a, 0.01)
    _, tb = evaluate_predictions(b, 0.01)
    assert ta == tb
    assert set(ra["threshold"]) == {select_threshold(a[a["split"] == "calibration"], 0.01)}
    assert ra["test_set"].tolist() == ["toxicchat"]


def test_select_threshold_refuses_non_calibration_rows():
    df = _preds([0, 1])
    with pytest.raises(ValueError):
        select_threshold(df, 0.01)


def test_paired_bootstrap_outputs_intervals_and_paired_differences():
    a = _preds([0] * 100 + [1] * 30)
    b = a.assign(model="ce", score=a["score"] + 0.1)
    preds = pd.concat([a, b], ignore_index=True)
    _, thresholds = evaluate_predictions(preds, 0.01)
    boot = paired_bootstrap(preds, thresholds, 0.01, n_boot=50, seed=0)
    diff = boot[(boot["model"] == "ce") & (boot["minus"] == "base") & (boot["metric"] == "roc_auc")]
    # adding a constant changes no ranking: paired AUC difference is exactly 0
    assert diff["ci_low"].item() == pytest.approx(0.0) and diff["ci_high"].item() == pytest.approx(0.0)
    assert (boot["ci_low"] <= boot["ci_high"]).all()
