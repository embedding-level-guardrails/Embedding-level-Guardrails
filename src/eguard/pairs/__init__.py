"""Сборка обучающих пар для RQ2/RQ3.

Раскладка на диске:
    data/processed/pairs/{name}/pairs_{split}.jsonl   # триплеты
    data/processed/pairs/{name}/texts_{split}.jsonl   # те же тексты плоско, с меткой
    data/processed/pairs/{name}/summary.json          # зафиксированные размеры
"""
from __future__ import annotations

from pathlib import Path

from ..utils import get_logger, read_jsonl
from .builders import BUILDERS, PairContext, deduplicate, flatten_texts, make_pair
from .sources import load_aegis_records, load_harmbench
from .transforms import JAILBREAKS, PARAPHRASES, load_external_variants

logger = get_logger(__name__)


def pairs_dir(processed_root: str | Path, name: str) -> Path:
    return Path(processed_root) / "pairs" / name


def pairs_path(processed_root: str | Path, name: str, split: str, kind: str = "pairs") -> Path:
    return pairs_dir(processed_root, name) / f"{kind}_{split}.jsonl"


def load_pairs(processed_root: str | Path, name: str, split: str, kind: str = "pairs") -> list[dict]:
    path = pairs_path(processed_root, name, split, kind)
    if not path.exists():
        raise FileNotFoundError(f"No {path}. Run scripts/07_build_pairs.py")
    return read_jsonl(path)


def dedupe_pool(records: list[dict]) -> list[dict]:
    """Один текст — одна запись.

    В AEGIS один и тот же текст встречается по нескольку раз (2318 строк train при
    2169 уникальных текстах). Без этого якорь и «позитив того же класса» могут
    оказаться одной и той же строкой с разными id.
    """
    seen: set[str] = set()
    out = []
    for record in records:
        key = record["text"].strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def exclude_texts(records: list[dict], blocked: set[str]) -> list[dict]:
    """Убирает записи, чей текст встречается в другом сплите.

    Дубликаты в AEGIS пересекают и официальную границу train/test (21 текст), так
    что без этой фильтрации train-пары протекают в val/test.
    """
    return [r for r in records if r["text"].strip() not in blocked]


def split_texts(records: list[dict]) -> set[str]:
    return {r["text"].strip() for r in records}


def build_split(ctx: PairContext, sizes: dict[str, int]) -> list[dict]:
    """Прогоняет все запрошенные типы пар для одного сплита."""
    pairs: list[dict] = []
    for pair_type, n in sizes.items():
        if n <= 0:
            continue
        builder = BUILDERS.get(pair_type)
        if builder is None:
            raise KeyError(f"Unknown pair_type '{pair_type}'. Available: {sorted(BUILDERS)}")
        produced = builder(ctx, int(n))
        logger.info("%s/%s: запрошено %d, построено %d", ctx.split, pair_type, n, len(produced))
        pairs.extend(produced)
    return deduplicate(pairs)


__all__ = [
    "BUILDERS", "JAILBREAKS", "PARAPHRASES", "PairContext", "build_split", "dedupe_pool",
    "deduplicate", "exclude_texts", "split_texts",
    "flatten_texts", "load_aegis_records", "load_external_variants", "load_harmbench",
    "load_pairs", "make_pair", "pairs_dir", "pairs_path",
]
