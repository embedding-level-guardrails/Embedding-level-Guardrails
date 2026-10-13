"""Шаг 10 (RQ3): как objective перестроил пространство эмбеддингов.

    python scripts/10_embedding_geometry.py --config configs/rq3_wildguardmix.yaml
    python scripts/10_embedding_geometry.py --runs wgm-classification wgm-joint --skip-base

`09_eval_lodo.py` отвечает на вопрос «насколько хорошо классифицирует», этот скрипт —
на вопрос «почему»: он смотрит на сами векторы, без классификатора сверху.

Два замера на каждую модель (исходный энкодер + все прогоны с чекпоинтами):

1. Порядок в отложенных триплетах — доля триплетов из `pairs_test.jsonl`, где позитив
   ближе к якорю, чем негатив, по типам пар. Самый показательный тип —
   `adversarial_contrast`: там у якоря с негативом общая форма (jailbreak), а с
   позитивом — общее содержание. Два типа читать с оговоркой: `paraphrase` слишком
   лёгкий (у всех ~100%), а `benign_twin` майнился по близости у исходного e5, так
   что сравнение по нему смещено в пользу исходной модели.

2. Геометрия пространства на hold-out и OOD: доля разброса на первую главную
   компоненту (насколько пространство сплющено в одну ось) и средние косинусы
   между классами и внутри них.

На выходе, в {paths.results}/rq3/:
    geometry.csv        — геометрия по моделям и наборам
    triplet_order.csv   — порядок в триплетах по моделям и типам пар
    GEOMETRY.md         — обе таблицы

Метрики дозаписываются в MLflow-прогон модели, как и в 09_eval_lodo.py.
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

from eguard.analysis.geometry import class_cosine_summary, pc1_share, triplet_margins  # noqa: E402
from eguard.config import load_config  # noqa: E402
from eguard.data import load_split  # noqa: E402
from eguard.pairs import load_pairs  # noqa: E402
from eguard.training import discover_runs, load_checkpoint, resume_run  # noqa: E402
from eguard.utils import ensure_dir, get_logger, save_json, set_seed  # noqa: E402

logger = get_logger("embedding_geometry")

# Типы пар, по которым сравнение моделей смещено или неинформативно (см. докстринг).
CAVEATS = {
    "benign_twin": "майнился по близости у исходного e5 — сравнение смещено в его пользу",
    "paraphrase": "слишком лёгкая проверка — у всех моделей около 100%",
}


def sample_records(records: list[dict], n: int, seed: int) -> list[dict]:
    """Случайная выборка без повторов; 0 или n >= len — все записи."""
    if not n or n >= len(records):
        return records
    rng = np.random.default_rng(seed)
    return [records[i] for i in rng.choice(len(records), n, replace=False)]


def triplet_rows(model_name: str, emb: dict[str, np.ndarray], pairs: list[dict]) -> list[dict]:
    """Порядок в триплетах: по каждому типу пар и по всем вместе."""
    anchor = np.stack([emb[p["anchor"]] for p in pairs])
    positive = np.stack([emb[p["positive"]] for p in pairs])
    negative = np.stack([emb[p["negative"]] for p in pairs])
    margins = triplet_margins(anchor, positive, negative)
    types = np.array([p["pair_type"] for p in pairs])

    rows = []
    for pair_type in [*sorted(set(types)), "all"]:
        m = margins if pair_type == "all" else margins[types == pair_type]
        rows.append({
            "model": model_name,
            "pair_type": pair_type,
            "n": int(len(m)),
            "frac_positive_closer": float((m > 0).mean()),
            "mean_margin": float(m.mean()),
            "caveat": CAVEATS.get(pair_type, ""),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rq3_wildguardmix.yaml")
    ap.add_argument("--ood-config", default="configs/ood_toxicchat.yaml")
    ap.add_argument("--runs", nargs="*", default=None, help="имена каталогов в output_dir прогонов")
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--skip-base", action="store_true", help="не считать исходный энкодер")
    ap.add_argument("--pairs-split", default="test", help="сплит триплетов (по умолчанию отложенный)")
    ap.add_argument("--ood-sample", type=int, default=1500, help="размер выборки OOD; 0 — весь набор")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ood_cfg = load_config(args.ood_config)
    set_seed(cfg.seed)

    encoder_key = args.encoder or cfg.encoders[0].key
    spec = cfg.encoder(encoder_key)
    device = args.device or cfg.training.get("device", "auto")

    pairs_name = cfg.pairs.get("name", "pairs_v1")
    pairs = load_pairs(cfg.paths.processed, pairs_name, args.pairs_split, "pairs")
    triplet_texts = sorted({t for p in pairs for t in (p["anchor"], p["positive"], p["negative"])})

    sets = {
        "holdout": load_split(cfg.paths.processed, cfg.dataset.name, "test"),
        "ood": sample_records(load_split(ood_cfg.paths.processed, ood_cfg.dataset.name, "test"),
                              args.ood_sample, args.sample_seed),
    }
    logger.info("триплетов=%d (уникальных текстов %d), holdout=%d, ood=%d",
                len(pairs), len(triplet_texts), len(sets["holdout"]), len(sets["ood"]))

    runs_root = Path(cfg.training.get("output_dir", "artifacts/runs"))
    models: list[tuple[str, Path | None]] = [] if args.skip_base else [(f"{encoder_key}-frozen", None)]
    models.extend(discover_runs(runs_root, args.runs))
    if not models:
        raise SystemExit(f"Нет прогонов с чекпоинтами в {runs_root} — сначала scripts/08_train_contrastive.py")

    geometry_rows: list[dict] = []
    triplet_all: list[dict] = []
    for model_name, run_dir in models:
        logger.info("=== %s ===", model_name)
        checkpoint = spec.hf_id if run_dir is None else run_dir / "checkpoint"
        model, resolved = load_checkpoint(checkpoint, spec, device=device)

        vectors = model.encode_texts(triplet_texts, resolved, batch_size=args.batch_size).cpu().numpy()
        rows = triplet_rows(model_name, dict(zip(triplet_texts, vectors)), pairs)
        triplet_all.extend(rows)

        for set_name, records in sets.items():
            x = model.encode_texts([r["text"] for r in records], resolved,
                                   batch_size=args.batch_size).cpu().numpy()
            y = np.array([r["label"] for r in records])
            geometry_rows.append({"model": model_name, "eval_set": set_name, "n": int(len(y)),
                                  "pc1_share": pc1_share(x), **class_cosine_summary(x, y)})

        # в тот же MLflow-прогон, где училась модель
        result_path = run_dir / "result.json" if run_dir else None
        if result_path and result_path.exists() and not args.no_mlflow:
            run_id = json.loads(result_path.read_text(encoding="utf-8")).get("mlflow_run_id")
            tracker = resume_run(cfg.mlflow, run_id)
            if tracker.enabled:
                payload = {f"geom_{g['eval_set']}_{k}": g[k] for g in geometry_rows
                           if g["model"] == model_name
                           for k in ("pc1_share", "cos_safe_harm", "cos_within_safe", "cos_within_harm")}
                payload.update({f"triplet_{r['pair_type']}_frac_correct": r["frac_positive_closer"]
                                for r in rows})
                tracker.log_metrics(payload)
                tracker.finish()

        del model
        if resolved.type == "mps":
            torch.mps.empty_cache()
        elif resolved.type == "cuda":
            torch.cuda.empty_cache()

    out_dir = ensure_dir(Path(cfg.paths.results) / "rq3")
    geometry = pd.DataFrame(geometry_rows).round(4)
    triplets = pd.DataFrame(triplet_all).round(4)
    geometry.to_csv(out_dir / "geometry.csv", index=False)
    triplets.to_csv(out_dir / "triplet_order.csv", index=False)
    save_json(out_dir / "geometry_meta.json", {
        "encoder": encoder_key, "pairs": pairs_name, "pairs_split": args.pairs_split,
        "n_triplets": len(pairs), "holdout": cfg.dataset.name, "ood": ood_cfg.dataset.name,
        "ood_sample": len(sets["ood"]), "sample_seed": args.sample_seed,
        "models": [m for m, _ in models],
    })

    order = triplets.pivot(index="model", columns="pair_type", values="frac_positive_closer")
    order = order.reindex([m for m, _ in models])
    caveat_lines = "".join(f"- `{k}`: {v}.\n" for k, v in CAVEATS.items())
    lines = [
        "# RQ3 — геометрия дообученных пространств\n",
        f"Энкодер `{encoder_key}`. Триплеты: `{pairs_name}`/{args.pairs_split} ({len(pairs)} шт.). "
        f"Hold-out: `{cfg.dataset.name}` test ({len(sets['holdout'])}). OOD: `{ood_cfg.dataset.name}` test, "
        f"выборка {len(sets['ood'])} (seed {args.sample_seed}).\n",
        "\n## Порядок в отложенных триплетах\n",
        "Доля триплетов, где позитив ближе к якорю, чем негатив.\n\n",
        (order * 100).round(1).astype(str).add("%").to_markdown() + "\n",
        "\nОговорки:\n", caveat_lines,
        "\n## Геометрия пространства\n",
        "`pc1_share` — доля разброса векторов на первую главную компоненту (чем больше, тем сильнее "
        "пространство сплющено в одну ось); `cos_*` — средние косинусы между классами и внутри них.\n\n",
        geometry.to_markdown(index=False) + "\n",
    ]
    (out_dir / "GEOMETRY.md").write_text("".join(lines), encoding="utf-8")

    print("\n=== Порядок в триплетах ===")
    print((order * 100).round(1).to_string())
    print("\n=== Геометрия ===")
    print(geometry.to_string(index=False))
    logger.info("Готово: %s", out_dir / "GEOMETRY.md")


if __name__ == "__main__":
    main()
