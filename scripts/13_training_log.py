"""Шаг 13: лог обучения -> таблицы по прогонам (шаг, лосс, метрики валидации, чекпоинты).

    python scripts/13_training_log.py --log rq3_pure.log --config configs/rq3_wgm_pure.yaml
    python scripts/13_training_log.py ... --mlflow-uri sqlite:///mlflow.db   # + компоненты лосса

На машине, где шло обучение, — всё сразу, с архивом для отправки:

    python scripts/13_training_log.py --mlflow-uri sqlite:///mlflow.db --archive artifacts/share/rq3_training.zip

Лог `make rq3-pure` содержит все прогоны подряд вперемешку со служебными строками.
Скрипт раскладывает его по прогонам и пишет в {paths.results}/rq3/training/:

    {run}_loss.csv   — лосс каждые log_every шагов (шаг, эпоха, loss)
    {run}_eval.csv   — валидация каждые eval_every шагов и в конце эпох: метрики,
                       изменение TPR@FPR=1% к прошлой проверке, лосс на этом шаге,
                       перезаписан ли на этом шаге сохранённый чекпоинт
    TRAINING.md      — сводка: конфигурация каждого прогона, данные и порядок их
                       использования, таблицы валидации

Важно: чекпоинт у прогона один — лучший по валидации, он перезаписывался при каждом
улучшении. Промежуточные состояния модели не сохранялись; колонка checkpoint_saved
показывает, на каких шагах происходила перезапись, final_checkpoint — какой остался.

Лосс в логе — суммарный. Компоненты (loss_supcon, loss_ce, cos_margin и т.п.) пишутся
только в MLflow; с --mlflow-uri они подтягиваются по mlflow_run_id из result.json, если
прогон есть в этой базе (база лежит на машине, где шло обучение).

--archive собирает zip: таблицы и сводку, config_used.json и result.json каждого
прогона, конфиги, сырой лог, сводку по триплетам и итоговые таблицы оценки. С
--with-checkpoints добавляются и веса (по ~128 МБ на прогон).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eguard.config import load_config  # noqa: E402
from eguard.utils import ensure_dir, get_logger  # noqa: E402

logger = get_logger("training_log")

RE_START = re.compile(r"train_contrastive \| \S+ \| (\w+) \| триплетов=(\d+), текстов=(\d+) \((\w+)\), val=(\d+)")
RE_LOSS = re.compile(r"эпоха (\d+) шаг (\d+) \| loss=([\d.]+)")
RE_EVAL = re.compile(r"шаг (\d+) \| val AUROC=([\d.]+) TPR@FPR=([\d.]+) FPR=([\d.]+) FNR=([\d.]+)")
RE_SAVE = re.compile(r"Чекпоинт сохранён: (\S+)")
RE_DONE = re.compile(r"Готово за ([\d.]+) c\. Лучший шаг (\d+)")

MLFLOW_METRICS = ("train_loss_supcon", "train_loss_ce", "train_loss_contrastive", "train_cos_margin",
                  "train_cos_same_class", "train_cos_other_class", "train_lr")

OBJECTIVE_TEXT = {
    "classification": "CE поверх линейной головы",
    "contrastive": "InfoNCE на триплетах",
    "joint": "InfoNCE на триплетах + CE на текстах",
    "supcon": "SupCon на текстах",
    "supcon_joint": "SupCon + CE на одном батче текстов",
}


def parse_log(path: Path) -> list[dict]:
    """Прогоны в порядке появления в логе."""
    runs: list[dict] = []
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if m := RE_START.search(line):
            current = {"objective": m[1], "n_pairs": int(m[2]), "n_texts": int(m[3]),
                       "text_source": m[4], "n_val": int(m[5]), "loss": [], "eval": [], "run": None}
            runs.append(current)
        elif current is None:
            continue
        elif m := RE_LOSS.search(line):
            current["loss"].append({"step": int(m[2]), "epoch": int(m[1]), "loss": float(m[3])})
        elif m := RE_EVAL.search(line):
            current["eval"].append({"step": int(m[1]), "val_auroc": float(m[2]),
                                    "val_tpr_at_fpr_0.01": float(m[3]), "val_fpr": float(m[4]),
                                    "val_fnr": float(m[5]), "checkpoint_saved": False})
        elif m := RE_SAVE.search(line):
            current["run"] = current["run"] or Path(m[1]).parent.name
            if current["eval"]:
                current["eval"][-1]["checkpoint_saved"] = True
        elif m := RE_DONE.search(line):
            current["seconds"], current["best_step"] = float(m[1]), int(m[2])
            current = None
    return runs


def eval_table(run: dict, steps_per_epoch: int) -> pd.DataFrame:
    loss = pd.DataFrame(run["loss"])
    rows = []
    prev_tpr = None
    for e in run["eval"]:
        # лосс пишется каждые log_every шагов; для шагов конца эпохи берётся ближайший предыдущий
        logged = loss[loss.step <= e["step"]].tail(1)
        rows.append({
            "step": e["step"],
            "epoch": (e["step"] - 1) // steps_per_epoch,
            "train_loss": float(logged.loss.iloc[0]) if len(logged) else None,
            "train_loss_at_step": int(logged.step.iloc[0]) if len(logged) else None,
            **{k: e[k] for k in ("val_auroc", "val_tpr_at_fpr_0.01", "val_fpr", "val_fnr")},
            "delta_val_tpr": None if prev_tpr is None else round(e["val_tpr_at_fpr_0.01"] - prev_tpr, 4),
            "checkpoint_saved": e["checkpoint_saved"],
        })
        prev_tpr = e["val_tpr_at_fpr_0.01"]
    df = pd.DataFrame(rows)
    # Повторная проверка на том же шаге (конец эпохи совпал с eval_every) — оставляем первую.
    df = df.drop_duplicates(subset="step", keep="first").reset_index(drop=True)
    saved = df.index[df.checkpoint_saved]
    df["final_checkpoint"] = False
    if len(saved):
        df.loc[saved[-1], "final_checkpoint"] = True
    return df


def mlflow_components(uri: str, run_id: str | None) -> pd.DataFrame | None:
    if not uri or not run_id:
        return None
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri(uri)
        client = MlflowClient()
        client.get_run(run_id)
    except Exception as exc:                     # прогона нет в этой базе — обычное дело
        logger.info("MLflow-прогон %s недоступен (%s): компоненты лосса не добавлены", run_id, type(exc).__name__)
        return None
    frames = []
    for name in MLFLOW_METRICS:
        hist = client.get_metric_history(run_id, name)
        if hist:
            frames.append(pd.DataFrame({"step": [h.step for h in hist], name: [h.value for h in hist]}).set_index("step"))
    return pd.concat(frames, axis=1).reset_index() if frames else None


def describe_run(name: str, run: dict, cfg_used: dict, result: dict, cfg, steps_per_epoch: int) -> list[str]:
    objective = cfg_used["objective"]
    uses_pairs = objective in ("contrastive", "joint")
    uses_supcon = objective in ("supcon", "supcon_joint")
    lines = [
        f"\n## {name}\n",
        f"- **Схема обучения:** `{objective}` — {OBJECTIVE_TEXT.get(objective, objective)}.\n",
    ]
    loss_bits = []
    if uses_supcon:
        loss_bits.append(f"SupCon, температура {cfg_used.get('supcon_temperature')}")
    if uses_pairs:
        loss_bits.append(f"{cfg_used['loss']}, температура {cfg_used['temperature']}")
    if objective in ("classification", "joint", "supcon_joint"):
        loss_bits.append("CE на сыром векторе через голову Linear(384→2)")
    lines.append(f"- **Лосс:** {'; '.join(loss_bits)}. Веса компонент: CE {cfg_used['ce_weight']}, "
                 f"контрастивная {cfg_used['contrastive_weight']}.\n")
    unused = []
    if not uses_pairs:
        unused.append(f"`loss: {cfg_used['loss']}` и `temperature: {cfg_used['temperature']}` (относятся к триплетам)")
    if not uses_supcon:
        unused.append(f"`supcon_temperature: {cfg_used.get('supcon_temperature')}`")
    if unused:
        lines.append(f"- В `config_used.json` есть поля, которые этот прогон **не использует**: {', '.join(unused)}.\n")
    lines += [
        f"- **Оптимизация:** AdamW, lr {cfg_used['lr']}, weight decay {cfg_used['weight_decay']}, "
        f"warmup {cfg_used['warmup_ratio']:.0%} + косинусное затухание, клиппинг градиента "
        f"{cfg_used['max_grad_norm']}, батч {cfg_used['batch_size']}, {cfg_used['epochs']} эпохи по "
        f"{steps_per_epoch} шагов = {result['steps']} шагов, seed {cfg_used['seed']}, gradient checkpointing.\n",
        f"- **Данные обучения:** {run['n_texts']:,} текстов "
        f"({'train WildGuardMix напрямую' if run['text_source'] == 'dataset' else 'плоский список из триплетов'})"
        + (f" и {run['n_pairs']:,} триплетов `{cfg.pairs.get('name')}` (тексты триплетов совпадают с train: "
           f"{result.get('texts_match_pairs')})" if run["n_pairs"] else ", триплеты не используются") + ".\n",
        f"- **Валидация и выбор чекпоинта:** WildGuardMix val ({run['n_val']:,}), каждые "
        f"{cfg_used['eval_every']} шагов и в конце эпох; критерий — TPR@FPR=1% на val "
        f"({'голова' if objective in ('classification', 'joint', 'supcon_joint') else 'скор по центроидам классов'}).\n",
        f"- **Итог:** {result['steps']} шагов за {run.get('seconds', result['seconds']) / 60:.1f} мин; "
        f"сохранённый чекпоинт — шаг {result['best']['step']}, val AUROC {result['best'].get('auroc', float('nan')):.4f}, "
        f"TPR@FPR=1% {result['best']['tpr_at_fpr']:.4f}. MLflow run id: `{result.get('mlflow_run_id')}`.\n",
        f"- **Файлы:** `{name}_loss.csv`, `{name}_eval.csv`; конфигурация — "
        f"`artifacts/runs_wgm_pure/{name}/config_used.json`, `result.json`.\n",
    ]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="rq3_pure.log")
    ap.add_argument("--config", default="configs/rq3_wgm_pure.yaml")
    ap.add_argument("--out", default=None, help="по умолчанию {paths.results}/rq3/training")
    ap.add_argument("--mlflow-uri", default=None, help="база MLflow с прогонами, чтобы добавить компоненты лосса")
    ap.add_argument("--archive", default=None, help="собрать zip для отправки, например artifacts/share/rq3_training.zip")
    ap.add_argument("--pairs-config", default="configs/rq2_pairs_wgm_pure.yaml", help="кладётся в архив")
    ap.add_argument("--ood-config", default="configs/ood_toxicchat.yaml", help="кладётся в архив")
    ap.add_argument("--with-checkpoints", action="store_true", help="положить в архив и веса (~128 МБ на прогон)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    runs_root = Path(cfg.training.get("output_dir", "artifacts/runs"))
    out = ensure_dir(args.out or Path(cfg.paths.results) / "rq3" / "training")
    runs = parse_log(Path(args.log))
    if not runs:
        raise SystemExit(f"В {args.log} не найдено ни одного прогона")

    md = [
        "# Обучение RQ3: шаги, лосс, валидация, чекпоинты\n",
        f"Источник: лог `{args.log}`, конфигурация `{args.config}`, файлы прогонов в `{runs_root}/`.\n",
        "\n**Чекпоинты.** У каждого прогона сохранён один чекпоинт — лучший по TPR@FPR=1% на валидации. "
        "Он перезаписывался при каждом улучшении, промежуточные состояния не сохранялись. В таблицах "
        "`checkpoint_saved` — шаги, на которых происходила перезапись, `final_checkpoint` — оставшийся.\n",
        "\n**Лосс.** В логе — суммарный лосс каждые 20 шагов, и это значение **на одном батче** "
        "(32 текста) на этом шаге, а не среднее за интервал, поэтому он шумный: следить за обучением "
        "надёжнее по метрикам валидации. На шагах конца эпохи (1344, 2688) показан ближайший "
        "предыдущий записанный (колонка `train_loss_at_step`). Компоненты лосса (SupCon и CE по "
        "отдельности) — только в MLflow на машине, где шло обучение.\n",
        "\n## Данные и порядок их использования\n",
        "1. **Подготовка** (`scripts/00_prepare_data.py`, `configs/rq2_pairs_wgm_pure.yaml`): WildGuardMix "
        "с HF — `wildguardtrain` (86 759 строк) и `wildguardtest` (1 725). Строки дедуплицированы по тексту "
        "промта, промты с противоречивыми метками и без метки выброшены. Из train стратифицированно по "
        "метке отрезано 10% в val (seed 42): train 43 025, val 4 781, test 1 699.\n",
        "2. **Триплеты** (`scripts/07_build_pairs.py`, режим coverage) — только для `contrastive` и `joint`: "
        "каждый текст train — якорь ровно одного триплета.\n",
        "3. **Обучение** (`scripts/08_train_contrastive.py`): train WildGuardMix; порядок батчей — "
        "перемешивание DataLoader в каждой эпохе, генератор зафиксирован `torch.manual_seed(42)` в начале "
        "обучения. Сам порядок батчей не логировался, но воспроизводим при тех же коде, данных и seed.\n",
        "4. **Выбор чекпоинта:** WildGuardMix val, во время обучения.\n",
        "5. **Итоговая оценка** (`scripts/09_eval_lodo.py`): WildGuardTest и ToxicChat test (2 853), "
        "после обучения; результаты — `results/wgm_pure/rq3/LODO.md`.\n",
    ]

    for run in runs:
        name = run["run"]
        run_dir = runs_root / name
        cfg_used = json.loads((run_dir / "config_used.json").read_text(encoding="utf-8"))
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        if cfg_used["objective"] != run["objective"]:
            raise SystemExit(f"{name}: в логе {run['objective']}, в config_used.json {cfg_used['objective']}")
        steps_per_epoch = result["steps"] // cfg_used["epochs"]

        loss = pd.DataFrame(run["loss"])
        extra = mlflow_components(args.mlflow_uri, result.get("mlflow_run_id"))
        if extra is not None:
            loss = loss.merge(extra, on="step", how="left")
        loss.to_csv(out / f"{name}_loss.csv", index=False)

        evals = eval_table(run, steps_per_epoch)
        evals.to_csv(out / f"{name}_eval.csv", index=False)
        final = evals[evals.final_checkpoint]
        if len(final) and int(final.step.iloc[0]) != result["best"]["step"]:
            raise SystemExit(f"{name}: по логу финальный чекпоинт на шаге {int(final.step.iloc[0])}, "
                             f"в result.json — {result['best']['step']}")

        md += describe_run(name, run, cfg_used, result, cfg, steps_per_epoch)
        show = evals.assign(**{"чекпоинт": evals.apply(
            lambda r: "**оставлен**" if r.final_checkpoint else ("перезаписан" if r.checkpoint_saved else ""), axis=1)})
        show = show[["step", "epoch", "train_loss", "val_auroc", "val_tpr_at_fpr_0.01", "delta_val_tpr",
                     "val_fpr", "val_fnr", "чекпоинт"]]
        md.append("\n" + show.to_markdown(index=False) + "\n")
        logger.info("%s: %d точек лосса, %d проверок, финальный чекпоинт — шаг %s",
                    name, len(loss), len(evals), result["best"]["step"])

    (out / "TRAINING.md").write_text("".join(md), encoding="utf-8")
    print(f"Готово: {out}")

    if args.archive:
        names = [r["run"] for r in runs]
        n_with_components = sum(
            1 for name in names
            if len(pd.read_csv(out / f"{name}_loss.csv", nrows=1).columns) > 3)
        build_archive(Path(args.archive), out, runs_root, names, cfg, args)
        print(f"Архив: {args.archive}")
        print(f"Компоненты лосса из MLflow: {n_with_components} из {len(names)} прогонов"
              + ("" if n_with_components == len(names) else
                 " — для остальных их нет в этой базе (запускайте на машине, где шло обучение)"))


def build_archive(path: Path, tables: Path, runs_root: Path, names: list[str], cfg, args) -> None:
    """zip с тем, что нужно для разбора обучения; внутри всё лежит в папке rq3_training/."""
    results = Path(cfg.paths.results)
    pairs_results = Path(load_config(args.pairs_config).paths.results) if Path(args.pairs_config).exists() else None
    entries: list[tuple[Path, str]] = [(f, f"training/{f.name}") for f in sorted(tables.iterdir()) if f.is_file()]
    for name in names:
        for f in ("config_used.json", "result.json"):
            entries.append((runs_root / name / f, f"runs/{name}/{f}"))
        if args.with_checkpoints:
            ckpt = runs_root / name / "checkpoint"
            entries += [(f, f"runs/{name}/checkpoint/{f.name}") for f in sorted(ckpt.iterdir()) if f.is_file()]
    for f in (args.config, args.pairs_config, args.ood_config):
        entries.append((Path(f), f"configs/{Path(f).name}"))
    entries.append((Path(args.log), Path(args.log).name))
    for f in ([pairs_results / "rq2" / "PAIRS.md"] if pairs_results else []) + \
             [results / "rq3" / n for n in ("LODO.md", "lodo.csv", "GEOMETRY.md")]:
        entries.append((f, f.name))

    ensure_dir(path.parent)
    missing = []
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, arc in entries:
            if src.exists():
                zf.write(src, f"rq3_training/{arc}")
            else:
                missing.append(str(src))
    if missing:
        logger.warning("В архив не попали (нет файлов): %s", ", ".join(missing))


if __name__ == "__main__":
    main()
