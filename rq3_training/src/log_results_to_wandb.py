"""Log RQ3 results to Weights & Biases from the CSV/JSON files the notebook writes.

Works without the notebook, the training data or a GPU: everything is read from a
`results/` directory, optionally downloaded from the private Hugging Face repo the
notebook uploads to.

    # results already on disk
    python -m rq3_training.src.log_results_to_wandb --project rq3-mmbert-small

    # download from Hugging Face first
    python -m rq3_training.src.log_results_to_wandb \
        --hf-repo <user>/rq3-mmbert-small --project rq3-mmbert-small --entity <team>

Creates one run per (variant, seed) with the training history and final metrics, one
run per sweep configuration, and a `summary` run holding the comparison tables and the
training-curves figure. Model weights are not uploaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
IDENTIFIER_COLUMNS = ("model", "variant", "seed", "scorer", "dataset")
SUMMARY_TABLES = (
    "classification_summary.csv",
    "comparisons_summary.csv",
    "lodo_gap.csv",
    "geometry_before_after.csv",
    "category_fnr.csv",
    "category_diagnostics.csv",
    "scorer_params.csv",
)


def download_results(repo_id: str, destination: Path) -> Path:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, repo_type="model", allow_patterns=["results/*"], local_dir=destination)
    return destination / "results"


def read_frame(results_dir: Path, name: str) -> pd.DataFrame | None:
    path = results_dir / name
    return pd.read_csv(path) if path.exists() else None


def read_json(results_dir: Path, name: str) -> dict:
    path = results_dir / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def log_training_runs(wandb, results_dir: Path, project: str, entity: str | None) -> int:
    history = read_frame(results_dir, "training_geometry_history.csv")
    if history is None:
        print(f"в {results_dir} нет training_geometry_history.csv — нечего логировать")
        return 0

    steps = read_frame(results_dir, "training_steps.csv")
    classification = read_frame(results_dir, "classification_runs.csv")
    collapses = read_json(results_dir, "geometry_collapse.json")
    best_hparams = read_json(results_dir, "best_hparams.json")
    metrics = [column for column in classification.columns if column not in IDENTIFIER_COLUMNS] if classification is not None else []

    for (variant, seed), rows in history.groupby(["variant", "seed"]):
        key = f"{variant}@seed{int(seed)}"
        collapse = collapses.get(key)
        run = wandb.init(
            project=project, entity=entity, name=key, group=variant, job_type="train", reinit=True,
            tags=["main", f"collapsed={collapse is not None}"],
            config={**best_hparams.get(variant, {}), "variant": variant, "seed": int(seed),
                    "steps": int(rows["step"].max())},
        )

        per_step = rows.drop(columns=["variant", "seed"]).set_index("step").add_prefix("eval/")
        if steps is not None:
            run_steps = steps[(steps["variant"] == variant) & (steps["seed"] == seed)].drop(columns=["variant", "seed"])
            per_step = per_step.join(run_steps.set_index("step").add_prefix("train/"), how="outer")
        for step, values in per_step.sort_index().iterrows():
            clean = {name: float(value) for name, value in values.items() if pd.notna(value) and not isinstance(value, str)}
            if clean:
                run.log(clean, step=int(step))

        if classification is not None:
            model_rows = classification[(classification["variant"] == variant) & (classification["seed"] == seed)]
            for _, row in model_rows.iterrows():
                for metric in metrics:
                    run.summary[f"{row['dataset']}/{row['scorer']}/{metric}"] = float(row[metric])
        if collapse:
            run.summary["collapse/step"] = collapse["step"]
            run.summary["collapse/loss"] = collapse["loss at step"]
            run.summary["collapse/reasons"] = "; ".join(collapse["reasons"])
        run.finish()

    return history.groupby(["variant", "seed"]).ngroups


def log_sweep_runs(wandb, results_dir: Path, project: str, entity: str | None) -> int:
    sweep = read_frame(results_dir, "sweep.csv")
    if sweep is None:
        return 0

    for _, row in sweep.iterrows():
        run = wandb.init(
            project=project, entity=entity, name=f"sweep · {row['variant']} · {row['config']}",
            group=row["variant"], job_type="sweep", reinit=True, tags=["sweep"],
            config={"variant": row["variant"], "config": row["config"]},
        )
        run.summary["val AUROC (probe)"] = float(row["val AUROC (probe)"])
        run.summary["collapsed"] = bool(row["collapsed"])
        run.finish()
    return len(sweep)


def log_summary_run(wandb, results_dir: Path, project: str, entity: str | None) -> None:
    run = wandb.init(project=project, entity=entity, name="summary", job_type="report", reinit=True, tags=["summary"])
    for name in SUMMARY_TABLES:
        frame = read_frame(results_dir, name)
        if frame is not None:
            run.log({name.removesuffix(".csv"): wandb.Table(dataframe=frame)})
    curves = results_dir / "training_curves.png"
    if curves.exists():
        run.log({"training_curves": wandb.Image(str(curves))})
    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", required=True, help="проект W&B (создастся, если его нет)")
    parser.add_argument("--entity", default=None, help="команда W&B; по умолчанию — личный аккаунт")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--hf-repo", default=None, help="скачать results/ из этого приватного репозитория HF")
    args = parser.parse_args()

    import wandb

    results_dir = download_results(args.hf_repo, args.results_dir.parent) if args.hf_repo else args.results_dir
    wandb.login()
    trained = log_training_runs(wandb, results_dir, args.project, args.entity)
    if trained:
        swept = log_sweep_runs(wandb, results_dir, args.project, args.entity)
        log_summary_run(wandb, results_dir, args.project, args.entity)
        print(f"залогировано в {args.project}: {trained} запусков обучения, {swept} конфигураций перебора, сводка")


if __name__ == "__main__":
    main()
