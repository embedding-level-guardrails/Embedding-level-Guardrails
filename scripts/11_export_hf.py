"""Шаг 11: собрать дообученную модель в папку для загрузки на Hugging Face Hub.

    python scripts/11_export_hf.py --run wgm-joint --repo-id USER/e5-small-v2-wildguardmix-joint

Ничего никуда не загружает — только готовит папку (по умолчанию
artifacts/hf_release/<run>/) и проверяет её. Загрузка — scripts/12_push_to_hub.py.

Что делает:
1. Копирует веса и токенизатор из чекпоинта, добавляет vocab.txt и конфиги
   sentence-transformers от базовой модели, перекладывает голову из pickle
   (head.pt) в safetensors.
2. Считает порог головы при FPR = 1% на валидации и метрики на hold-out и ToxicChat
   с этим порогом — они попадают в карточку модели.
3. Пишет README.md (карточку модели).
4. Сверяет собранную папку с исходным чекпоинтом: загружает её обычным
   transformers, как это сделает пользователь, и сравнивает векторы и вероятности.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eguard.analysis.probe import binary_metrics, tpr_at_fpr  # noqa: E402
from eguard.config import load_config  # noqa: E402
from eguard.data import load_split  # noqa: E402
from eguard.hub import assemble_release, compare_with_checkpoint, render_model_card  # noqa: E402
from eguard.training import load_checkpoint  # noqa: E402
from eguard.utils import get_logger  # noqa: E402

logger = get_logger("export_hf")


def head_scores(model, device, records: list[dict]) -> np.ndarray:
    """P(harm) головы на СЫРЫХ векторах — на них голова обучалась."""
    raw = model.encode_texts([r["text"] for r in records], device, normalize=False)
    with torch.no_grad():
        return torch.softmax(model.logits(raw.to(device)).float(), dim=-1)[:, 1].cpu().numpy()


def evaluate_head(cfg, ood_cfg, checkpoint: Path, spec, device: str, target_fpr: float):
    """Порог с валидации и метрики на hold-out и OOD при этом пороге."""
    model, resolved = load_checkpoint(checkpoint, spec, device=device)
    val = load_split(cfg.paths.processed, cfg.dataset.name, "val")
    y_val = np.array([r["label"] for r in val])
    _, threshold = tpr_at_fpr(y_val, head_scores(model, resolved, val), target_fpr)

    rows = []
    for name, records in (
        ("WildGuardTest (in-domain)", load_split(cfg.paths.processed, cfg.dataset.name, "test")),
        ("ToxicChat test (out-of-domain)", load_split(ood_cfg.paths.processed, ood_cfg.dataset.name, "test")),
    ):
        y = np.array([r["label"] for r in records])
        m = binary_metrics(y, head_scores(model, resolved, records), threshold=threshold,
                           target_fpr=target_fpr)
        rows.append({"name": name, "n": len(y), "auroc": m["auroc"],
                     "tpr": m[f"tpr_at_fpr_{target_fpr:g}"], "fpr": m["fpr"], "fnr": m["fnr"]})
        logger.info("%s: AUROC %.3f, FPR %.3f, FNR %.3f", name, m["auroc"], m["fpr"], m["fnr"])
    return float(threshold), rows


def count_lines(path: Path) -> int | str:
    if not path.exists():
        return "about 32.5k"
    with open(path, encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rq3_wildguardmix.yaml")
    ap.add_argument("--ood-config", default="configs/ood_toxicchat.yaml")
    ap.add_argument("--run", default="wgm-joint", help="каталог прогона в output_dir")
    ap.add_argument("--repo-id", default=None,
                    help="будущий репозиторий на Hub, нужен для примеров в README "
                         "(по умолчанию USER/e5-small-v2-wildguardmix-<objective>)")
    ap.add_argument("--out", default=None, help="папка выгрузки (по умолчанию artifacts/hf_release/<run>)")
    ap.add_argument("--skip-eval", action="store_true", help="без порога и метрик в README")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ood_cfg = load_config(args.ood_config)
    spec = cfg.encoders[0]
    run_dir = Path(cfg.training.get("output_dir", "artifacts/runs")) / args.run
    checkpoint = run_dir / "checkpoint"
    if not checkpoint.exists():
        raise SystemExit(f"Нет чекпоинта {checkpoint}")

    trained = json.loads((run_dir / "config_used.json").read_text(encoding="utf-8"))
    objective = trained.get("objective", "unknown")
    repo_id = args.repo_id or f"USER/e5-small-v2-wildguardmix-{objective}"
    out = Path(args.out or f"artifacts/hf_release/{args.run}")

    # 1. файлы
    files = assemble_release(checkpoint, out, base_model_id=spec.hf_id)
    logger.info("Собрано в %s: %s", out, ", ".join(files))

    # 2. порог и метрики головы
    threshold, rows = None, None
    has_head = (checkpoint / "head.pt").exists()
    if has_head and not args.skip_eval:
        threshold, rows = evaluate_head(cfg, ood_cfg, checkpoint, spec,
                                        args.device or cfg.training.get("device", "auto"),
                                        cfg.training.get("target_fpr", 0.01))

    # 3. карточка модели
    pairs_file = Path(cfg.paths.processed) / "pairs" / cfg.pairs.get("name", "") / "pairs_train.jsonl"
    card = render_model_card(
        repo_id=repo_id, objective=objective, base_model=spec.hf_id,
        n_pairs=count_lines(pairs_file), epochs=trained["epochs"], batch_size=trained["batch_size"],
        lr=trained["lr"], temperature=trained["temperature"], threshold=threshold, eval_rows=rows,
    )
    (out / "README.md").write_text(card, encoding="utf-8")

    # 4. сверка с исходным чекпоинтом
    check = compare_with_checkpoint(out, checkpoint, spec)
    ok = check["max_abs_diff"] < 1e-4

    print(f"\nПапка для загрузки: {out}")
    print(f"Файлы: {', '.join(files + ['README.md'])}")
    if threshold is not None:
        print(f"Порог головы (FPR = 1% на валидации): {threshold:.4f}")
    print(f"Сверка с чекпоинтом: макс. расхождение векторов {check['max_abs_diff']:.2e} — "
          f"{'OK' if ok else 'РАСХОЖДЕНИЕ, не загружайте'}")
    for text, p_rel, p_loc in zip(check["texts"], check.get("p_harm_release", []), check.get("p_harm_local", [])):
        print(f"  P(harm) {p_rel:.3f} (чекпоинт {p_loc:.3f})  {text}")
    if not ok:
        raise SystemExit(1)
    print(f"\nДальше: python scripts/12_push_to_hub.py --repo-id {repo_id} --folder {out}")


if __name__ == "__main__":
    main()
