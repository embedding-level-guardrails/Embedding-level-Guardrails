"""Collect predictions, evaluate, and write per-seed / aggregate / bootstrap tables."""

from __future__ import annotations

from typing import Any

import pandas as pd

from e5_guardrails.config import paths
from e5_guardrails.evaluation import evaluate_predictions, paired_bootstrap

METRICS = ["fpr", "fnr", "tpr", "tpr_at_fpr", "roc_auc", "avg_precision"]
LOSS_NAMES = {"base": "frozen E5", "ce": "CE", "supcon": "SupCon", "ce_supcon": "CE + SupCon"}


def collect_predictions(cfg: dict[str, Any]) -> pd.DataFrame:
    runs = paths(cfg).root / "runs" / cfg["variant"]
    files = sorted(runs.glob("*/seed=*/predictions.parquet"))
    if not files:
        raise FileNotFoundError(f"No predictions under {runs}; run probe first")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["protocol", "variant", "model", "readout", "test_set", "subset"]
    agg = results.groupby(keys, sort=True)[METRICS].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg["n_seeds"] = results.groupby(keys, sort=True)["seed"].nunique()
    return agg.reset_index()


def _fmt(x: float) -> str:
    return "—" if pd.isna(x) else f"{x:.3f}"


def markdown(results: pd.DataFrame, summary: pd.DataFrame, target_fpr: float) -> str:
    pct = f"{target_fpr:.0%}"
    main = results[(results["readout"] == "linear_probe") & (results["subset"] == "all")]
    lines = [
        "# RQ3 results (linear probe on frozen h)", "",
        (f"FPR/FNR/TPR: source-calibration threshold (max TPR at FPR <= {pct}) carried unchanged to each test set. "
         f"TPR@FPR<={pct}: each test set's own ROC curve. AP = average precision."), "",
        "## Per seed", "",
        f"| protocol | loss | seed | test set | threshold | FPR | FNR | TPR | TP | FP | TN | FN | TPR@FPR<={pct} | ROC-AUC | AP |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in main.sort_values(["protocol", "test_set", "model", "seed"]).itertuples():
        lines.append(
            f"| {r.protocol} | {LOSS_NAMES.get(r.model, r.model)} | {r.seed} | {r.test_set} | {r.threshold:.4f} | "
            f"{_fmt(r.fpr)} | {_fmt(r.fnr)} | {_fmt(r.tpr)} | {r.tp} | {r.fp} | {r.tn} | {r.fn} | "
            f"{_fmt(r.tpr_at_fpr)} | {_fmt(r.roc_auc)} | {_fmt(r.avg_precision)} |")
    s = summary[(summary["readout"] == "linear_probe") & (summary["subset"] == "all")]
    lines += [
        "", "## Mean ± std over training seeds", "",
        f"| protocol | loss | test set | seeds | FPR | FNR | TPR@FPR<={pct} | ROC-AUC | AP |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    def cell(r, m: str) -> str:
        return f"{_fmt(getattr(r, m + '_mean'))} ± {_fmt(getattr(r, m + '_std'))}"

    for r in s.sort_values(["protocol", "test_set", "model"]).itertuples():
        lines.append(f"| {r.protocol} | {LOSS_NAMES.get(r.model, r.model)} | {r.test_set} | {r.n_seeds} | "
                     f"{cell(r, 'fpr')} | {cell(r, 'fnr')} | {cell(r, 'tpr_at_fpr')} | {cell(r, 'roc_auc')} | "
                     f"{cell(r, 'avg_precision')} |")
    return "\n".join(lines) + "\n"


def run_evaluation(cfg: dict[str, Any], bootstrap: bool = True) -> pd.DataFrame:
    target = cfg["evaluation"]["target_fpr"]
    preds = collect_predictions(cfg)
    results, thresholds = evaluate_predictions(preds, target)
    summary = summarize(results)
    out = paths(cfg).results
    out.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out / "predictions.parquet", index=False)
    results.to_csv(out / "results_per_seed.csv", index=False)
    summary.to_csv(out / "results_summary.csv", index=False)
    if bootstrap and cfg["evaluation"]["bootstrap"]["n"] > 0:
        b = cfg["evaluation"]["bootstrap"]
        paired_bootstrap(preds, thresholds, target, b["n"], b["seed"]).to_csv(out / "bootstrap_ci.csv", index=False)
    (out / "results.md").write_text(markdown(results, summary, target))
    return results
