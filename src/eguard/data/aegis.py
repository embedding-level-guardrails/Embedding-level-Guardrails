"""Загрузчик и нормализация AEGIS (nvidia/Aegis-AI-Content-Safety-Dataset-1.0).

Сырая строка датасета — это текст плюс до пяти независимых аннотаций
(`labels_0` … `labels_4`). Каждая аннотация — либо `Safe`, либо `Needs Caution`,
либо список категорий вреда через запятую (`Violence, Threat`). Отсюда три шага:

    parse_annotation  — одна аннотация  -> (класс, категории)
    aggregate_row     — пять аннотаций  -> один класс из трёх + согласие аннотаторов
    to_binary         — класс из трёх   -> бинарная метка по caution_policy

Разведение `caution` в отдельный класс принципиально: в AEGIS это ~пятая часть
разметки, и от того, куда её отнести, зависит и harm_rate, и рабочая точка
guardrail. Поэтому политика вынесена в конфиг (exclude / safe / harm), а не
зашита здесь.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable

from ..config import DatasetSpec
from ..utils import get_logger

logger = get_logger(__name__)

SAFE_TOKEN = "safe"
CAUTION_TOKEN = "needs caution"

# Столбцы аннотаторов: labels_0 … labels_4.
LABEL_COLUMN_RE = re.compile(r"^labels?_(\d+)$")

# При равенстве голосов берём более консервативный класс: пропущенный harm
# дороже ложного срабатывания, и так же поступают авторы AEGIS.
CONSERVATISM = {"safe": 0, "caution": 1, "harm": 2}

# Официальный test не трогаем, val пайплайн отрезает от train сам
# (см. scripts/00_prepare_data.py), поэтому готовый validation-сплит не берём.
DEFAULT_SPLITS = ("train", "test")
SPLIT_ALIASES = {"validation": "val", "valid": "val", "dev": "val"}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return not str(value).strip()


def parse_annotation(value: Any) -> tuple[str, list[str]]:
    """Одна аннотация -> (класс, категории).

    Класс: safe | caution | harm | none (последнее — аннотатор не размечал строку).
    """
    if _is_missing(value):
        return "none", []

    parts = [p.strip() for p in str(value).split(",")]
    parts = [p for p in parts if p]
    if not parts:
        return "none", []

    categories = [p for p in parts if p.lower() not in (SAFE_TOKEN, CAUTION_TOKEN)]
    if categories:
        return "harm", categories
    if any(p.lower() == CAUTION_TOKEN for p in parts):
        return "caution", []
    return "safe", []


def annotation_columns(row: dict) -> list[str]:
    """Столбцы аннотаторов в порядке номера, а не в порядке ключей словаря."""
    numbered = []
    for column in row:
        match = LABEL_COLUMN_RE.match(str(column))
        if match:
            numbered.append((int(match.group(1)), str(column)))
    return [column for _, column in sorted(numbered)]


def aggregate_row(row: dict) -> dict[str, Any]:
    """Пять аннотаций -> один класс из трёх + метаданные согласия.

    `agreement` (доля голосов за победивший класс) сохраняется в записи: в error
    analysis видно, что большая часть ошибок проба приходится на строки, где
    аннотаторы сами не сошлись.
    """
    votes: Counter[str] = Counter()
    categories: list[str] = []

    for column in annotation_columns(row):
        klass, cats = parse_annotation(row[column])
        if klass == "none":
            continue
        votes[klass] += 1
        categories.extend(cats)

    n_annotators = sum(votes.values())
    if not n_annotators:
        return {"class3": "none", "categories": [], "category": None,
                "agreement": 0.0, "n_annotators": 0, "votes": {}}

    top = max(votes.values())
    winners = [klass for klass, count in votes.items() if count == top]
    class3 = max(winners, key=lambda k: CONSERVATISM[k])

    ordered = [cat for cat, _ in Counter(categories).most_common()]
    return {
        "class3": class3,
        "categories": ordered,
        "category": ordered[0] if ordered else None,
        "agreement": top / n_annotators,
        "n_annotators": n_annotators,
        "votes": dict(votes),
    }


def to_binary(class3: str, policy: str = "exclude") -> int | None:
    """Класс из трёх -> бинарная метка. None означает «выбросить строку»."""
    if class3 == "safe":
        return 0
    if class3 == "harm":
        return 1
    if class3 == "caution":
        if policy == "exclude":
            return None
        if policy == "safe":
            return 0
        if policy == "harm":
            return 1
        raise ValueError(f"Unknown caution_policy: {policy!r} (exclude | safe | harm)")
    return None


def normalize_rows(rows: Iterable[dict], spec: DatasetSpec) -> list[dict]:
    """Сырые строки AEGIS -> плоские записи пайплайна.

    Фильтры: тип текста, длина, политика по caution. Порядок записей сохраняется —
    на него завязан порядок строк в кешах эмбеддингов.
    """
    keep_types = {t.strip().lower() for t in spec.text_types}
    all_types = "all" in keep_types

    out: list[dict] = []
    for i, raw in enumerate(rows):
        row = dict(raw)

        text_type = str(row.get("text_type") or "").strip()
        if not all_types and text_type.lower() not in keep_types:
            continue

        text = str(row.get("text") or "").strip()
        if not spec.min_chars <= len(text) <= spec.max_chars:
            continue

        agg = aggregate_row(row)
        label = to_binary(agg["class3"], spec.caution_policy)
        if label is None:
            continue

        if label == 1:
            category = agg["category"] or ("Needs Caution" if agg["class3"] == "caution" else "Other")
        else:
            category = "safe"

        out.append({
            "id": str(row.get("id", i)),
            "text": text,
            "text_type": text_type,
            "label": label,
            "label_name": "harm" if label == 1 else "safe",
            "class3": agg["class3"],
            "category": category,
            "categories": agg["categories"],
            "agreement": round(float(agg["agreement"]), 3),
            "n_annotators": agg["n_annotators"],
            "n_chars": len(text),
        })

    return out


def load_raw(spec: DatasetSpec, splits: tuple[str, ...] = DEFAULT_SPLITS) -> dict[str, list[dict]]:
    """Скачивает датасет с HF Hub и отдаёт сырые строки по сплитам."""
    from datasets import load_dataset

    dataset = load_dataset(spec.hf_id)
    out: dict[str, list[dict]] = {}

    for name in dataset:
        canonical = SPLIT_ALIASES.get(name.lower(), name.lower())
        if canonical not in splits:
            logger.warning("%s: сплит '%s' пропущен (val отрезается от train)", spec.hf_id, name)
            continue
        out[canonical] = [dict(row) for row in dataset[name]]
        logger.info("%s/%s: %d сырых строк", spec.hf_id, canonical, len(out[canonical]))

    if not out:
        available = ", ".join(dataset)
        raise ValueError(f"{spec.hf_id}: нет сплитов {splits}. Доступны: {available}")

    sample = next(iter(out.values()))[0]
    if "text" not in sample or not annotation_columns(sample):
        raise ValueError(
            f"{spec.hf_id}: ожидались столбцы 'text' и 'labels_N', получены: {sorted(sample)}"
        )
    return out


# --- Синтетика для офлайн-прогона пайплайна ------------------------------------
# Заглушки в формате сырого AEGIS, нужные только чтобы прогнать `make smoke` без
# сети и без torch. В экспериментах не участвуют.
#
# Классы разводятся статистикой словаря, а не маркером в тексте, и часть словаря
# у них общая. Это принципиально: на линейно разделимой синтетике все пробы дают
# AUROC=1.0, ошибок нет, и смоук перестаёт ловить регрессии в анализе. Здесь же
# probe выходит в район 0.85–0.97, контроль по длине остаётся около 0.5, а общая
# часть словаря даёт непустые FP/FN для 06_error_analysis.

_SYNTH_CATEGORIES = [
    "Violence", "Threat", "Criminal Planning/Confessions", "Harassment",
    "Hate/Identity Hate", "Guns and Illegal Weapons", "Controlled/Regulated Substances",
    "Suicide and Self Harm", "PII/Privacy", "Sexual",
]
_SAFE_WORDS = ["рецепта", "поездки", "договора", "отчёта", "ремонта", "тренировки",
               "бюджета", "резюме", "переезда", "собеседования"]
_HARM_WORDS = ["конфликта", "угрозы", "оружия", "взлома", "мошенничества", "слежки",
               "шантажа", "вербовки", "подделки", "травли"]
# Общая часть: из неё берутся слова для обоих классов, отсюда и берутся ошибки.
_SHARED_WORDS = ["инструкции", "плана", "доступа", "системы", "сообщения",
                 "денег", "адреса", "инструмента"]

# Шаблоны общие для обоих классов — иначе разделимость ловилась бы по длине.
_TEMPLATES = [
    "нужен короткий разбор: {a} и {b}, с чего начать",
    "объясни простыми словами, как связаны {a} и {b}",
    "составь план на неделю: сначала {a}, потом {b}",
    "какие типичные ошибки бывают, когда речь про {a} и {b}",
    "подскажи чек-лист по теме {a} с учётом {b}",
]

_SHARED_WORD_RATE = 0.4   # доля слов из общего пула у обычной строки
_HARD_ROW_RATE = 0.12     # доля строк целиком из общего пула: неразличимы по тексту


def _pick(rng, pool: list[str]) -> str:
    return pool[int(rng.integers(len(pool)))]


def _synthetic_text(class3: str, rng) -> str:
    own = _HARM_WORDS if class3 == "harm" else _SAFE_WORDS
    hard = rng.random() < _HARD_ROW_RATE
    words = [
        _pick(rng, _SHARED_WORDS) if hard or rng.random() < _SHARED_WORD_RATE else _pick(rng, own)
        for _ in range(2)
    ]
    return _pick(rng, _TEMPLATES).format(a=words[0], b=words[1])


def _synthetic_annotations(class3: str, category: str, rng) -> list[str | None]:
    """Пять аннотаций с шумом: 1-2 несогласных, чтобы agreement не был всегда 1.0."""
    main = {"safe": "Safe", "caution": "Needs Caution", "harm": category}[class3]
    other = {"safe": "Needs Caution", "caution": "Safe", "harm": "Safe"}[class3]

    n_annotators = int(rng.integers(3, 6))
    n_dissent = int(rng.integers(0, max(n_annotators // 2, 1)))
    labels: list[str | None] = [main] * (n_annotators - n_dissent) + [other] * n_dissent
    rng.shuffle(labels)
    return labels + [None] * (5 - n_annotators)


def make_synthetic(n: int = 600, seed: int = 42, test_fraction: float = 0.2) -> dict[str, list[dict]]:
    """N строк в формате сырого AEGIS: train/test, три класса, два типа текста."""
    import numpy as np

    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for i in range(n):
        roll = rng.random()
        class3 = "safe" if roll < 0.45 else ("harm" if roll < 0.9 else "caution")
        category = _pick(rng, _SYNTH_CATEGORIES)
        text = _synthetic_text(class3, rng)
        if class3 == "caution":
            text = f"на грани: {text}"

        labels = _synthetic_annotations(class3, category, rng)
        rows.append({
            "id": f"synth-{i}",
            "text": text,
            # Часть строк — ответы модели: так проверяется фильтр по text_type.
            "text_type": "llm_response" if rng.random() < 0.15 else "user_message",
            **{f"labels_{j}": labels[j] for j in range(5)},
        })

    cut = int(round(n * (1 - test_fraction)))
    return {"train": rows[:cut], "test": rows[cut:]}
