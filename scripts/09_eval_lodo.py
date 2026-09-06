"""Шаг 9 (RQ3): сравнение objective на hold-out и OOD (LODO: AEGIS -> ToxicChat).

    python scripts/09_eval_lodo.py --config configs/rq3_contrastive.yaml
    python scripts/09_eval_lodo.py --runs rq3-ce rq3-contrastive rq3-joint

Что и почему меряется именно так.

1. Единый readout. Три objective дают три РАЗНЫХ пространства, и сравнивать их
   «каждый своей головой» некорректно: у чисто contrastive-варианта головы нет
   вообще. Поэтому основная строка — `probe_logreg`: поверх эмбеддингов каждой
   модели обучается ОДИН И ТОТ ЖЕ линейный проб на AEGIS train. Различается только
   пространство, всё остальное зафиксировано — это и есть контролируемое сравнение
   из RQ3. Родная голова и центроидный скор идут дополнительными строками.

2. Порог. FPR/FNR репортятся при пороге, выставленном на FPR=1% по AEGIS **val**,
   и этот же порог применяется к hold-out и к OOD. Подбирать порог на OOD нельзя:
   в проде калибруются на своих данных, а деградация при переносе — это ровно то,
   что мы и хотим увидеть. Рядом лежат AUROC и TPR@FPR=1%, посчитанные на каждом
   наборе отдельно: они показывают качество ранжирования безотносительно порога.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eguard.analysis.probe import binary_metrics, bootstrap_ci, fit_transfer_probe, tpr_at_fpr  # noqa: E402
from eguard.config import load_config  # noqa: E402
from eguard.data import load_split  # noqa: E402
from eguard.training import load_checkpoint, resume_run  # noqa: E402
from eguard.training.data import texts_and_labels  # noqa: E402
from eguard.utils import ensure_dir, get_logger, l2_normalize, save_json, set_seed  # noqa: E402

logger = get_logger("eval_lodo")


def discover_runs(runs_root: Path, names: list[str] | None) -> list[tuple[str, Path]]:
    """(имя, каталог) для каждого прогона с сохранённым чекпоинтом."""
    if names:
        candidates = [runs_root / n for n in names]
    else:
        candidates = sorted(p for p in runs_root.glob("*") if p.is_dir())
    found = []
    for path in candidates:
        if (path / "checkpoint").exists():
            found.append((path.name, path))
        else:
            logger.warning("%s: нет checkpoint/, пропускаю", path)
    return found


def embed_all(model, device, sets: dict[str, list[str]], batch_size: int) -> dict[str, np.ndarray]:
    return {name: model.encode_texts(texts, device, batch_size=batch_size).cpu().numpy()
            for name, texts in sets.items()}


def centroid_scores(x_train: np.ndarray, y_train: np.ndarray, x: np.ndarray) -> np.ndarray:
    mu_safe = l2_normalize(x_train[y_train == 0].mean(0)[None])[0]
    mu_harm = l2_normalize(x_train[y_train == 1].mean(0)[None])[0]
    return x @ mu_harm - x @ mu_safe


def rows_for_readout(model_name: str, readout: str, threshold: float,
                     evals: dict[str, tuple[np.ndarray, np.ndarray]],
                     target_fpr: float, n_boot: int, seed: int) -> list[dict]:
    """По строке на оценочный набор при общем пороге, откалиброванном на val."""
    out = []
    for eval_name, (y, scores) in evals.items():
        metrics = binary_metrics(y, scores, threshold=threshold, target_fpr=target_fpr)
        ci = bootstrap_ci(y, scores, target_fpr, n_boot, seed=seed) if n_boot else {}
        out.append({"model": model_name, "readout": readout, "eval_set": eval_name,
                    "threshold_from_val": round(float(threshold), 6), **metrics, **ci})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rq3_contrastive.yaml")
    ap.add_argument("--ood-config", default="configs/ood_toxicchat.yaml")
    ap.add_argument("--runs", nargs="*", default=None, help="имена каталогов в artifacts/runs")
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--skip-base", action="store_true", help="не считать frozen-базлайн")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ood_cfg = load_config(args.ood_config)
    set_seed(cfg.seed)

    encoder_key = args.encoder or cfg.encoders[0].key
    spec = cfg.encoder(encoder_key)
    target_fpr = cfg.training.get("target_fpr", 0.01)
    device = args.device or cfg.training.get("device", "auto")

    aegis_train = load_split(cfg.paths.processed, cfg.dataset.name, "train")
    aegis_val = load_split(cfg.paths.processed, cfg.dataset.name, "val")
    aegis_test = load_split(cfg.paths.processed, cfg.dataset.name, "test")
    ood_test = load_split(ood_cfg.paths.processed, ood_cfg.dataset.name, "test")

    texts = {}
    labels = {}
    for name, records in (("train", aegis_train), ("val", aegis_val),
                          ("holdout", aegis_test), ("ood", ood_test)):
        texts[name], labels[name] = texts_and_labels(records)
    logger.info("train=%d val=%d holdout=%d ood=%d (harm_rate holdout=%.3f, ood=%.3f)",
                len(texts["train"]), len(texts["val"]), len(texts["holdout"]), len(texts["ood"]),
                labels["holdout"].mean(), labels["ood"].mean())

    runs_root = Path(cfg.training.get("output_dir", "artifacts/runs"))
    models: list[tuple[str, Path | None]] = []
    if not args.skip_base:
        models.append((f"{encoder_key}-frozen", None))
    models.extend(discover_runs(runs_root, args.runs))
    if not models:
        raise SystemExit(f"Нет прогонов с чекпоинтами в {runs_root} — сначала scripts/08_train_contrastive.py")

    rows: list[dict] = []
    for model_name, run_dir in models:
        logger.info("=== %s ===", model_name)
        checkpoint = spec.hf_id if run_dir is None else run_dir / "checkpoint"
        model, resolved = load_checkpoint(checkpoint, spec, device=device)
        embeddings = embed_all(model, resolved, texts, args.batch_size)

        # (1) единый readout: один и тот же проб поверх каждого пространства
        clf, scores_val, probe_info = fit_transfer_probe(
            embeddings["train"], labels["train"], embeddings["val"], labels["val"],
            c_grid=cfg.analysis.get("probe", {}).get("c_grid"), seed=cfg.seed,
        )
        _, threshold = tpr_at_fpr(labels["val"], scores_val, target_fpr)
        if not np.isfinite(threshold):
            threshold = float(np.quantile(scores_val, 1 - target_fpr))
        evals = {
            name: (labels[name], clf.predict_proba(l2_normalize(embeddings[name].astype(np.float64)))[:, 1])
            for name in ("holdout", "ood")
        }
        probe_rows = rows_for_readout(model_name, "probe_logreg", threshold, evals,
                                      target_fpr, args.n_boot, cfg.seed)
        for row in probe_rows:
            row.update(probe_info)
        rows.extend(probe_rows)

        # (2) родная голова, если обучалась
        if model.head is not None:
            with torch.no_grad():
                head_scores = {
                    name: torch.softmax(
                        model.logits(torch.from_numpy(embeddings[name]).to(resolved)).float(), dim=-1
                    )[:, 1].cpu().numpy()
                    for name in ("val", "holdout", "ood")
                }
            _, head_threshold = tpr_at_fpr(labels["val"], head_scores["val"], target_fpr)
            if not np.isfinite(head_threshold):
                head_threshold = float(np.quantile(head_scores["val"], 1 - target_fpr))
            rows.extend(rows_for_readout(
                model_name, "native_head", head_threshold,
                {n: (labels[n], head_scores[n]) for n in ("holdout", "ood")},
                target_fpr, args.n_boot, cfg.seed,
            ))

        # (3) центроидный скор — прообраз distance-based guardrail из RQ5
        train_norm = l2_normalize(embeddings["train"].astype(np.float64))
        cent = {n: centroid_scores(train_norm, labels["train"],
                                   l2_normalize(embeddings[n].astype(np.float64)))
                for n in ("val", "holdout", "ood")}
        _, cent_threshold = tpr_at_fpr(labels["val"], cent["val"], target_fpr)
        if not np.isfinite(cent_threshold):
            cent_threshold = float(np.quantile(cent["val"], 1 - target_fpr))
        rows.extend(rows_for_readout(
            model_name, "centroid_distance", cent_threshold,
            {n: (labels[n], cent[n]) for n in ("holdout", "ood")},
            target_fpr, args.n_boot, cfg.seed,
        ))

        # метрики уезжают в тот же MLflow-прогон, где модель училась
        result_path = (run_dir / "result.json") if run_dir else None
        if result_path and result_path.exists():
            run_id = json.loads(result_path.read_text(encoding="utf-8")).get("mlflow_run_id")
            tracker = resume_run(cfg.mlflow, run_id)
            if tracker.enabled:
                payload = {}
                for row in rows:
                    if row["model"] != model_name:
                        continue
                    prefix = f"{row['eval_set']}_{row['readout']}"
                    for key in ("auroc", "fpr", "fnr", f"tpr_at_fpr_{target_fpr:g}"):
                        payload[f"{prefix}_{key}"] = row[key]
                tracker.log_metrics(payload)
                tracker.finish()

        del model
        if resolved.type == "mps":
            torch.mps.empty_cache()

    tpr_key = f"tpr_at_fpr_{target_fpr:g}"
    df = pd.DataFrame(rows)
    front = ["model", "readout", "eval_set", "auroc", "auroc_lo", "auroc_hi",
             tpr_key, "fpr", "fnr", "f1", "threshold_from_val", "harm_rate", "n_eval"]
    df = df[[c for c in front if c in df.columns] + [c for c in df.columns if c not in front]]
    df = df.round(4)

    out_dir = ensure_dir(Path(cfg.paths.results) / "rq3")
    df.to_csv(out_dir / "lodo.csv", index=False)
    save_json(out_dir / "lodo_meta.json", {
        "encoder": encoder_key, "target_fpr": target_fpr,
        "holdout": {"dataset": cfg.dataset.name, "n": int(len(labels["holdout"])),
                    "harm_rate": round(float(labels["holdout"].mean()), 3)},
        "ood": {"dataset": ood_cfg.dataset.name, "config": ood_cfg.dataset.config,
                "label_field": ood_cfg.dataset.label_field, "n": int(len(labels["ood"])),
                "harm_rate": round(float(labels["ood"].mean()), 3)},
        "models": [m for m, _ in models],
    })

    primary = df[df["readout"] == "probe_logreg"]
    lines = [
        "# RQ3 — objective на hold-out и OOD (LODO: AEGIS -> ToxicChat)\n",
        f"Энкодер: `{encoder_key}`. Hold-out: `{cfg.dataset.name}` test "
        f"(n={len(labels['holdout'])}, harm_rate={labels['holdout'].mean():.3f}). "
        f"OOD: `{ood_cfg.dataset.hf_id}`/`{ood_cfg.dataset.config}` test "
        f"(n={len(labels['ood'])}, harm_rate={labels['ood'].mean():.3f}, "
        f"label_field=`{ood_cfg.dataset.label_field}`).\n",
        f"\nПорог для FPR/FNR выставлен на AEGIS val при FPR={target_fpr:.0%} и применён без "
        "перекалибровки к обоим наборам: подбирать порог на OOD означало бы заглянуть в "
        "тестовые данные. `auroc` и `tpr_at_fpr` считаются на каждом наборе отдельно и от "
        "порога не зависят.\n",
        "\n## Основное сравнение: единый линейный проб поверх каждого пространства\n",
        primary.drop(columns=["readout"]).to_markdown(index=False) + "\n",
        "\nОдин и тот же проб обучен на AEGIS train поверх эмбеддингов каждой модели, "
        "поэтому различие строк — это различие пространств, а не readout-а.\n",
        "\n## Все readout-ы\n", df.to_markdown(index=False) + "\n",
        "\n`native_head` — собственная голова (есть только у CE и joint), "
        "`centroid_distance` — расстояние до центроидов классов, прообраз "
        "distance-based guardrail из RQ5.\n",
    ]
    (out_dir / "LODO.md").write_text("".join(lines), encoding="utf-8")

    print("\n=== Основное сравнение (единый проб) ===")
    show = ["model", "eval_set", "auroc", tpr_key, "fpr", "fnr"]
    print(primary[show].to_string(index=False))
    logger.info("Готово: %s", out_dir / "LODO.md")


if __name__ == "__main__":
    main()
