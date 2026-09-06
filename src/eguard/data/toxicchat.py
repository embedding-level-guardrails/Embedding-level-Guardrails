"""ToxicChat (lmsys/toxic-chat) — OOD-набор для LODO-оценки.

Зачем именно он. RQ3 сравнивает objective на hold-out AEGIS, но обзор (раздел 1.4)
прямо фиксирует, что детекторы поверх эмбеддингов работают in-domain и «резко
деградируют на реальном трафике (ToxicChat)». Поэтому LODO здесь не формальность:
обучаемся на AEGIS, тестируем на ToxicChat и смотрим, какой objective лучше держит
рабочую точку при переносе.

Отличия от AEGIS, из-за которых перенос и нетривиален:
  * это реальные пользовательские промты к чат-боту, а не собранный safety-набор;
  * дисбаланс сильный (примерно 7% toxic против 58% harm в AEGIS), поэтому AUROC
    здесь мало что говорит, а FPR при зафиксированном пороге — очень много;
  * разметка бинарная и своя, категорий вреда AEGIS в ней нет.
"""
from __future__ import annotations

from typing import Iterable

from ..config import DatasetSpec
from ..utils import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG = "toxicchat0124"
LABEL_FIELDS = ("toxicity", "jailbreaking", "any")


def to_binary(row: dict, label_field: str = "toxicity") -> int | None:
    """Бинарная метка harm из полей ToxicChat."""
    if label_field == "any":
        values = [row.get("toxicity"), row.get("jailbreaking")]
        if all(v is None for v in values):
            return None
        return int(any(int(v or 0) == 1 for v in values))

    if label_field not in LABEL_FIELDS:
        raise ValueError(f"Unknown label_field {label_field!r}. Available: {LABEL_FIELDS}")

    value = row.get(label_field)
    return None if value is None else int(int(value) == 1)


def category_of(row: dict) -> str:
    """Грубая категория для error analysis: своей таксономии у ToxicChat нет."""
    toxic = int(row.get("toxicity") or 0) == 1
    jailbreak = int(row.get("jailbreaking") or 0) == 1
    if toxic and jailbreak:
        return "toxicity+jailbreaking"
    if jailbreak:
        return "jailbreaking"
    if toxic:
        return "toxicity"
    return "safe"


def normalize_rows(rows: Iterable[dict], spec: DatasetSpec) -> list[dict]:
    """Сырые строки ToxicChat -> записи в формате пайплайна.

    Текст — `user_input`: guardrail стоит перед моделью и видит запрос
    пользователя, а не ответ. Это же делает сопоставимым с AEGIS/user_message.
    """
    out: list[dict] = []
    for i, raw in enumerate(rows):
        row = dict(raw)

        if spec.require_human_annotation and not bool(row.get("human_annotation", True)):
            continue

        text = str(row.get("user_input") or "").strip()
        if not spec.min_chars <= len(text) <= spec.max_chars:
            continue

        label = to_binary(row, spec.label_field)
        if label is None:
            continue

        out.append({
            "id": str(row.get("conv_id", i)),
            "text": text,
            "text_type": "user_message",
            "label": label,
            "label_name": "harm" if label == 1 else "safe",
            "class3": "harm" if label == 1 else "safe",
            "category": category_of(row) if label == 1 else "safe",
            "categories": [],
            "agreement": 1.0 if row.get("human_annotation", True) else 0.0,
            "n_annotators": 1,
            "n_chars": len(text),
            "source": "toxicchat",
        })
    return out


def load_raw(spec: DatasetSpec, splits: tuple[str, ...] = ("train", "test")) -> dict[str, list[dict]]:
    from datasets import load_dataset

    config = spec.config or DEFAULT_CONFIG
    dataset = load_dataset(spec.hf_id, config)

    out: dict[str, list[dict]] = {}
    for name in dataset:
        if name.lower() not in splits:
            continue
        out[name.lower()] = [dict(row) for row in dataset[name]]
        logger.info("%s/%s/%s: %d сырых строк", spec.hf_id, config, name, len(out[name.lower()]))

    if not out:
        raise ValueError(f"{spec.hf_id}/{config}: нет сплитов {splits}. Доступны: {', '.join(dataset)}")
    return out


def make_synthetic(n: int = 400, seed: int = 42, test_fraction: float = 0.5) -> dict[str, list[dict]]:
    """Заглушки в формате ToxicChat для офлайн-прогона (дисбаланс как в оригинале)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        toxic = int(rng.random() < 0.07)
        jailbreak = int(rng.random() < 0.02)
        rows.append({
            "conv_id": f"tc-synth-{i}",
            "user_input": (f"SYNTHETIC-TOXIC-{i}: заглушка запроса" if toxic
                           else f"обычный вопрос номер {i} про повседневную тему"),
            "model_output": "",
            "human_annotation": True,
            "toxicity": toxic,
            "jailbreaking": jailbreak,
        })
    cut = int(round(n * (1 - test_fraction)))
    return {"train": rows[:cut], "test": rows[cut:]}
