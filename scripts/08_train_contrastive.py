"""Шаг 8 (RQ3): дообучение энкодера на парах из шага 07, с трекингом в MLflow.

    python scripts/08_train_contrastive.py --config configs/rq3_contrastive.yaml
    python scripts/08_train_contrastive.py --objective classification --run-name ce-baseline
    python scripts/08_train_contrastive.py --pair-types safe_harm_contrast jailbreak_variant
    python scripts/08_train_contrastive.py --limit-pairs 200 --epochs 1 --no-mlflow   # быстрый прогон

Контролируемое сравнение RQ3: три запуска с одним и тем же `--config` и разным
`--objective` (contrastive / classification / joint) дают ту самую тройку, которой,
судя по литобзору, в работах нет.

Дообученный чекпоинт кладётся в {output_dir}/{run}/checkpoint и читается обычным
AutoModel — то есть его можно подставить в configs/rq1.yaml как ещё один энкодер
и прогнать весь RQ1-анализ уже на нём.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eguard.config import load_config  # noqa: E402
from eguard.data import load_split  # noqa: E402
from eguard.pairs import load_pairs  # noqa: E402
from eguard.training import GuardEncoder, TrainConfig, build_model, frozen_teacher, start_run, train  # noqa: E402
from eguard.utils import ensure_dir, get_logger, save_json, set_seed  # noqa: E402

logger = get_logger("train_contrastive")


def build_train_config(cfg, args) -> TrainConfig:
    raw = dict(cfg.training)
    raw.pop("device", None)
    raw.pop("output_dir", None)

    overrides = {
        "objective": args.objective, "loss": args.loss, "epochs": args.epochs,
        "batch_size": args.batch_size, "lr": args.lr, "temperature": args.temperature,
        "eval_every": args.eval_every,
    }
    raw.update({k: v for k, v in overrides.items() if v is not None})
    raw["pair_types"] = args.pair_types if args.pair_types is not None else (cfg.pairs.get("types") or [])
    raw["seed"] = cfg.seed
    known = TrainConfig.__dataclass_fields__
    unknown = set(raw) - set(known)
    if unknown:
        logger.warning("Незнакомые ключи в training: %s", sorted(unknown))
    return TrainConfig(**{k: v for k, v in raw.items() if k in known})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rq3_contrastive.yaml")
    ap.add_argument("--encoder", default=None, help="ключ энкодера из конфига")
    ap.add_argument("--objective", default=None, choices=["contrastive", "classification", "joint"])
    ap.add_argument("--loss", default=None,
                    choices=["info_nce", "supervised_contrastive", "triplet_margin"])
    ap.add_argument("--pair-types", nargs="*", default=None, help="абляция RQ2: подмножество типов пар")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--limit-pairs", type=int, default=0, help="обрезать train-пары (быстрый прогон)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    encoder_key = args.encoder or cfg.encoders[0].key
    spec = cfg.encoder(encoder_key)
    train_cfg = build_train_config(cfg, args)

    pairs_name = cfg.pairs.get("name", "pairs_v1")
    pairs_train = load_pairs(cfg.paths.processed, pairs_name, "train", "pairs")
    texts_train = load_pairs(cfg.paths.processed, pairs_name, "train", "texts")
    if args.limit_pairs:
        pairs_train = pairs_train[: args.limit_pairs]
        texts_train = texts_train[: args.limit_pairs * 2]

    # Валидация — исходный AEGIS val, а не пары: метрика должна отражать реальный
    # вход guardrail, а не распределение сконструированных триплетов.
    val_records = load_split(cfg.paths.processed, cfg.dataset.name, "val")
    logger.info("%s | пар=%d, текстов=%d, val=%d", encoder_key,
                len(pairs_train), len(texts_train), len(val_records))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"{encoder_key}-{train_cfg.objective}-{train_cfg.loss}-{stamp}"
    output_dir = ensure_dir(Path(cfg.training.get("output_dir", "artifacts/runs")) / run_name)

    with_head = train_cfg.objective in ("classification", "joint")
    model, device = build_model(spec, device=args.device or cfg.training.get("device", "auto"),
                                with_head=with_head)
    teacher = frozen_teacher(spec, device) if train_cfg.kl_weight > 0 else None
    logger.info("Устройство: %s, голова: %s, KL-якорь: %s", device, with_head, teacher is not None)

    mlflow_cfg = dict(cfg.mlflow)
    if args.no_mlflow:
        mlflow_cfg["enabled"] = False
    tracker = start_run(mlflow_cfg, run_name=run_name,
                        tags={"rq": "rq3", "encoder": encoder_key,
                              "objective": train_cfg.objective, "pairs": pairs_name})

    tracker.log_params({
        "encoder": {"key": encoder_key, "hf_id": spec.hf_id, "pooling": spec.pooling,
                    "max_length": spec.max_length, "prefix": spec.prefix},
        "training": {k: getattr(train_cfg, k) for k in TrainConfig.__dataclass_fields__},
        "data": {"pairs_name": pairs_name, "n_pairs_train": len(pairs_train),
                 "n_texts_train": len(texts_train), "n_val": len(val_records),
                 "pair_types": train_cfg.pair_types or "all"},
        "device": str(device),
    })

    status = "FINISHED"
    try:
        result = train(model, device, train_cfg, pairs_train, texts_train, val_records,
                       tracker, output_dir, teacher=teacher)
    except Exception:
        status = "FAILED"
        tracker.finish(status)
        raise

    save_json(Path(output_dir) / "result.json",
              {"run_name": run_name, "encoder": encoder_key, "objective": train_cfg.objective,
               "loss": train_cfg.loss, "mlflow_run_id": tracker.run_id, **result})
    Path(output_dir, "config_used.json").write_text(
        json.dumps({k: getattr(train_cfg, k) for k in TrainConfig.__dataclass_fields__},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    tracker.log_artifact(Path(output_dir) / "result.json")
    tracker.log_artifact(Path(output_dir) / "config_used.json")
    tracker.log_artifact(args.config, artifact_path="config")
    tracker.finish(status)

    print(f"\n=== {run_name} ===")
    print(f"лучший шаг {result['best']['step']}: "
          f"AUROC={result['best'].get('auroc', float('nan')):.4f}, "
          f"TPR@FPR={result['best']['tpr_at_fpr']:.4f}")
    print(f"чекпоинт: {result['checkpoint']}")
    logger.info("Готово: %s", output_dir)


if __name__ == "__main__":
    main()
