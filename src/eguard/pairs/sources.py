"""Источники примеров для обучающих пар: AEGIS (уже нормализован) и HarmBench.

HarmBench лежит на HF под gated-репозиторием, но канонический CSV с behaviors
открыто доступен в GitHub-репозитории проекта, поэтому берём его оттуда и кешируем
локально — так шаг воспроизводится без HF_TOKEN.

Про сплиты HarmBench. У бенчмарка есть официальное деление val (80 behaviors) /
test (320). В обучение по умолчанию идёт только val: HarmBench задуман как
оценочный набор, и если сжечь его целиком на трейне, не останется OOD-множества
для leave-one-dataset-out оценки, которая и заявлена как вклад работы.
"""
from __future__ import annotations

import csv
import io
import urllib.request
from pathlib import Path

from ..utils import ensure_dir, get_logger

logger = get_logger(__name__)

HARMBENCH_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
    "data/behavior_datasets/harmbench_behaviors_text_{split}.csv"
)
HARMBENCH_SPLITS = ("val", "test", "all")


def fetch_harmbench_csv(split: str = "val", cache_dir: str | Path = "data/raw/harmbench",
                        url_template: str = HARMBENCH_URL, timeout: int = 60) -> Path:
    """Скачивает и кеширует CSV с behaviors. Повторный вызов сеть не трогает."""
    if split not in HARMBENCH_SPLITS:
        raise ValueError(f"Unknown HarmBench split {split!r}. Available: {HARMBENCH_SPLITS}")

    path = ensure_dir(cache_dir) / f"harmbench_behaviors_text_{split}.csv"
    if path.exists():
        logger.info("HarmBench/%s: из кеша %s", split, path)
        return path

    url = url_template.format(split=split)
    logger.info("HarmBench/%s: скачиваю %s", split, url)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = response.read()
    path.write_bytes(payload)
    return path


def load_harmbench(
    split: str = "val",
    cache_dir: str | Path = "data/raw/harmbench",
    include_context: bool = False,
    drop_categories: tuple[str, ...] = (),
) -> list[dict]:
    """Behaviors HarmBench -> записи в том же формате, что и нормализованный AEGIS.

    `include_context` подклеивает ContextString к contextual-behaviors. По умолчанию
    выключен: guardrail на входе видит запрос пользователя, а не приложенный к нему
    документ, и склейка сильно смещает распределение длин относительно AEGIS.
    """
    path = fetch_harmbench_csv(split, cache_dir)
    rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))

    drop = {c.lower() for c in drop_categories}
    records: list[dict] = []
    for row in rows:
        category = (row.get("SemanticCategory") or "unknown").strip()
        if category.lower() in drop:
            continue
        text = (row.get("Behavior") or "").strip()
        if not text:
            continue
        context = (row.get("ContextString") or "").strip()
        if include_context and context:
            text = f"{context}\n\n{text}"

        records.append({
            "id": row.get("BehaviorID") or f"harmbench-{split}-{len(records)}",
            "text": text,
            "label": 1,
            "label_name": "harm",
            "category": category,
            "functional_category": (row.get("FunctionalCategory") or "").strip(),
            "source": "harmbench",
            "harmbench_split": split,
            "n_chars": len(text),
        })

    logger.info("HarmBench/%s: %d behaviors (после фильтров)", split, len(records))
    return records


def load_aegis_records(processed_root: str | Path, dataset: str, split: str) -> list[dict]:
    """Нормализованный AEGIS с диска + пометка источника."""
    from ..data import load_split

    records = []
    for record in load_split(processed_root, dataset, split):
        record = dict(record)
        record.setdefault("source", "aegis")
        records.append(record)
    return records
