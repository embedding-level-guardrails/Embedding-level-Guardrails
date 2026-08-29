"""Шаг 5: собирает CSV-результаты и картинки в один results/rq1/REPORT.md."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eguard.config import load_config  # noqa: E402
from eguard.utils import get_logger  # noqa: E402

logger = get_logger("report")


def md_table(path: Path, columns: list[str] | None = None) -> str:
    if not path.exists():
        return f"_No such file {path.name} — corresponding step was not ran._\n"
    df = pd.read_csv(path)
    if columns:
        df = df[[c for c in columns if c in df.columns]]
    return df.to_markdown(index=False) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rq1.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rq1 = Path(cfg.paths.results) / "rq1"
    figures = Path(cfg.paths.figures)

    summary_path = Path(cfg.paths.processed) / cfg.dataset.name / "summary.json"
    data_block = ""
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        data_block = pd.DataFrame(
            [{"split": s["split"], "n": s["n"], "harm_rate": s["harm_rate"],
              "median_chars": s["median_chars"]} for s in summary["splits"]]
        ).to_markdown(index=False)

    target_fpr = cfg.analysis.get("probe", {}).get("fpr_target", 0.01)
    probe_cols = ["model", "probe", "auroc", "auprc", "f1", "fpr", "fnr", f"tpr_at_fpr_{target_fpr:g}"]
    parts = [
        f"# RQ1 — baseline-разделимость safe/harm на frozen-энкодерах\n",
        f"Датасет: `{cfg.dataset.hf_id}`, text_types={cfg.dataset.text_types}, "
        f"caution_policy=`{cfg.dataset.caution_policy}`, seed={cfg.seed}\n",
        "## Данные\n", data_block + "\n",
        "## Косинусная близость intra vs inter\n", md_table(rq1 / "similarity.csv"),
        "\nЧитать так: `gap` — разрыв между средним внутриклассовым и межклассовым косинусом, "
        "`gap_std_units` — тот же разрыв в единицах разброса, `cos_random_pair` — уровень анизотропии "
        "(на сколько уже смещены косинусы случайных пар). Строки `variant=centered` — после вычитания "
        "глобального среднего.\n",
        "\n## Геометрия\n", md_table(rq1 / "geometry.csv"),
        "\n`direction_auroc` — AUROC проекции на разницу центроидов (одномерный сигнал), "
        "`ari_k2` — совпадает ли неразмеченная кластеризация с safety-меткой, "
        "`silhouette_label_cosine` — насколько метки образуют компактные кластеры.\n",
        "\n## Пробы\n", md_table(rq1 / "probe.csv", probe_cols),
        "\nРазрыв между `logreg` и `knn` показывает долю нелинейного сигнала; "
        "`centroid_distance` — прообраз distance-based guardrail из RQ5; "
        "`native_head` — собственная голова guardrail-модели на тех же данных. "
        "Строки `control_tfidf` и `control_length` — нижняя граница: если проб на эмбеддингах "
        "не выигрывает у TF-IDF, разделимость лексическая, а не семантическая. "
        "`auroc_lo`/`auroc_hi` — 95% перцентильный bootstrap-CI.\n",
        "\n## Error analysis\n", md_table(rq1 / "error_summary.csv"),
        "\n`max_cross_class_cosine` — насколько близко энкодер кладёт safe и harm друг к другу; "
        "подробные таблицы hard-пар и ошибок проба лежат в `results/rq1/errors/` "
        "(содержат фрагменты вредоносных текстов, не публиковать).\n",
        "\n## Визуализации\n",
    ]

    out = rq1 / "REPORT.md"
    for png in sorted(figures.glob("*.png")):
        rel = os.path.relpath(png, start=out.parent)
        parts.append(f"![{png.stem}]({Path(rel).as_posix()})\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    logger.info("Отчёт: %s", out)


if __name__ == "__main__":
    main()
