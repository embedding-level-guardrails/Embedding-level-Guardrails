"""Строители обучающих пар (RQ2) — по одному на тип пары.

Формат записи — триплет (anchor, positive, negative) с меткой типа. Триплет, а не
(a, b, label), потому что он одинаково ложится и на InfoNCE с in-batch negatives,
и на triplet margin, и на SupCon, а абляция «какой тип пар сколько даёт» сводится
к фильтру по полю `pair_type`.

Отдельно и намеренно: рядом с триплетами каждый строитель отдаёт все порождённые
тексты с бинарной меткой (см. `flatten_texts`). Без этого RQ3 не поставить честно —
сравнение contrastive против classification head требует, чтобы у обеих веток были
РОВНО одни и те же обучающие тексты, включая парафразы и jailbreak-варианты.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..utils import get_logger, l2_normalize
from .transforms import JAILBREAKS, PARAPHRASES, apply_named

logger = get_logger(__name__)


@dataclass
class PairContext:
    """Всё, из чего строятся пары для одного сплита."""
    split: str
    aegis: list[dict]
    harmbench: list[dict] = field(default_factory=list)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(42))
    cfg: dict = field(default_factory=dict)
    external_variants: dict[str, list[str]] = field(default_factory=dict)
    # (эмбеддинги, записи) фиксированного энкодера — нужны только benign_twin
    mining: tuple[np.ndarray, list[dict]] | None = None

    def pool(self, label: int, include_harmbench: bool = True) -> list[dict]:
        records = [r for r in self.aegis if r["label"] == label]
        if label == 1 and include_harmbench:
            records = records + self.harmbench
        return records


def _pick(rng: np.random.Generator, items: list) -> dict:
    return items[int(rng.integers(len(items)))]


def _pair_id(pair_type: str, anchor: str, positive: str, negative: str) -> str:
    digest = hashlib.sha1(f"{pair_type}\x00{anchor}\x00{positive}\x00{negative}".encode()).hexdigest()
    return f"{pair_type}-{digest[:16]}"


def make_pair(pair_type: str, split: str, anchor: dict, positive_text: str, negative: dict,
              anchor_label: int, variant: str = "", positive_source: str = "record",
              positive_record: dict | None = None) -> dict:
    """Один триплет + вся метаинформация, нужная для абляций и error analysis."""
    return {
        "pair_id": _pair_id(pair_type, anchor["text"], positive_text, negative["text"]),
        "pair_type": pair_type,
        "split": split,
        "anchor": anchor["text"],
        "positive": positive_text,
        "negative": negative["text"],
        "anchor_label": int(anchor_label),
        "negative_label": int(negative["label"]),
        "variant": variant,
        "positive_source": positive_source,
        "anchor_id": anchor.get("id"),
        "positive_id": (positive_record or {}).get("id"),
        "negative_id": negative.get("id"),
        "anchor_source": anchor.get("source", "aegis"),
        "negative_source": negative.get("source", "aegis"),
        "category": anchor.get("category"),
    }


def _positive_from_same_class(rng, anchor: dict, pool: list[dict]) -> dict | None:
    """Второй пример того же класса, по возможности из той же категории вреда."""
    category = anchor.get("category")
    same_category = [r for r in pool if r.get("category") == category and r["id"] != anchor["id"]]
    candidates = same_category if len(same_category) >= 1 else [r for r in pool if r["id"] != anchor["id"]]
    return _pick(rng, candidates) if candidates else None


# --- Типы пар ------------------------------------------------------------------

def build_safe_harm_contrast(ctx: PairContext, n: int) -> list[dict]:
    """Базовый контраст класса: якорь и позитив одного класса, негатив — другого.

    Часть пар строится от safe-якоря: если тянуть только harm, обучение стягивает
    вредоносный кластер, но ничего не говорит о структуре безопасной области, а
    именно её размытость и даёт ложные пропуски (см. 1.3 описания проекта).
    """
    harm, safe = ctx.pool(1), ctx.pool(0)
    if not harm or not safe:
        return []

    safe_fraction = float(ctx.cfg.get("safe_anchor_fraction", 0.5))
    out = []
    for _ in range(n):
        anchor_label = 0 if ctx.rng.random() < safe_fraction else 1
        anchor_pool, negative_pool = (safe, harm) if anchor_label == 0 else (harm, safe)
        anchor = _pick(ctx.rng, anchor_pool)
        positive = _positive_from_same_class(ctx.rng, anchor, anchor_pool)
        if positive is None:
            continue
        out.append(make_pair("safe_harm_contrast", ctx.split, anchor, positive["text"],
                             _pick(ctx.rng, negative_pool), anchor_label,
                             positive_source="record", positive_record=positive))
    return out


def build_paraphrase(ctx: PairContext, n: int) -> list[dict]:
    """Инвариантность к форме: позитив — парафраз якоря, негатив — другой класс."""
    # Обёртки вежливости и смена наклонения предполагают, что текст — это запрос.
    # На коротких репликах диалога («Great let's hear one.») они дают бессмыслицу,
    # поэтому короткие тексты в якоря парафраз не берём.
    min_chars = int(ctx.cfg.get("paraphrase_min_chars", 25))
    harm = [r for r in ctx.pool(1) if len(r["text"]) >= min_chars]
    safe = [r for r in ctx.pool(0) if len(r["text"]) >= min_chars]
    if not harm or not safe:
        logger.warning("%s: нет текстов длиннее %d символов для парафраз", ctx.split, min_chars)
        return []

    names = ctx.cfg.get("paraphrase_variants")
    safe_fraction = float(ctx.cfg.get("safe_anchor_fraction", 0.5))
    out = []
    for _ in range(n):
        anchor_label = 0 if ctx.rng.random() < safe_fraction else 1
        anchor_pool, negative_pool = (safe, harm) if anchor_label == 0 else (harm, safe)
        anchor = _pick(ctx.rng, anchor_pool)

        external = ctx.external_variants.get(str(anchor.get("id")))
        if external:
            positive_text, variant = _pick(ctx.rng, external), "external"
        else:
            positive_text, variant = apply_named(PARAPHRASES, names, anchor["text"], ctx.rng)

        if positive_text.strip() == anchor["text"].strip():
            continue
        out.append(make_pair("paraphrase", ctx.split, anchor, positive_text,
                             _pick(ctx.rng, negative_pool), anchor_label,
                             variant=variant, positive_source="transform"))
    return out


def build_jailbreak_variant(ctx: PairContext, n: int) -> list[dict]:
    """Обёртка не делает вредоносный запрос безопасным.

    Ключевая деталь — негатив. Если негативом всегда брать обычный safe-текст,
    модель выучит саму обёртку («ignore previous instructions», ролевая рамка) как
    признак вреда и начнёт валить безобидные запросы в такой же обёртке. Поэтому по
    умолчанию половина негативов — это БЕЗОПАСНЫЙ текст в той же самой обёртке:
    так единственный различающий сигнал остаётся содержательным.
    """
    harm_seed = ctx.harmbench or ctx.pool(1)
    safe = ctx.pool(0)
    if not harm_seed or not safe:
        return []

    names = ctx.cfg.get("jailbreak_variants")
    negative_mode = str(ctx.cfg.get("jailbreak_negative", "mix"))
    out = []
    for _ in range(n):
        anchor = _pick(ctx.rng, harm_seed)
        positive_text, variant = apply_named(JAILBREAKS, names, anchor["text"], ctx.rng)

        wrap_negative = negative_mode == "wrapped_safe" or (
            negative_mode == "mix" and ctx.rng.random() < 0.5
        )
        negative = _pick(ctx.rng, safe)
        if wrap_negative:
            wrapped, _ = apply_named(JAILBREAKS, [variant], negative["text"], ctx.rng)
            negative = {**negative, "text": wrapped, "id": f"{negative.get('id')}::{variant}"}

        out.append(make_pair("jailbreak_variant", ctx.split, anchor, positive_text, negative,
                             anchor_label=1, variant=variant, positive_source="transform"))
    return out


def build_benign_twin(ctx: PairContext, n: int) -> list[dict]:
    """Hard negatives: лексически ближайший safe к вредоносному якорю.

    Это ровно те пары, которые RQ1 находит в `06_error_analysis.py` как hard_pairs,
    и на которых Safe-Embed фиксирует систематические ошибки готовых энкодеров.
    Требует посчитанных эмбеддингов фиксированного энкодера (mining.encoder).
    """
    if ctx.mining is None:
        logger.warning("%s: benign_twin пропущен — нет эмбеддингов для майнинга", ctx.split)
        return []

    x, records = ctx.mining
    vectors = l2_normalize(x.astype(np.float64))
    labels = np.array([r["label"] for r in records])
    harm_idx = np.where(labels == 1)[0]
    safe_idx = np.where(labels == 0)[0]
    if not len(harm_idx) or not len(safe_idx):
        return []

    take = min(n, len(harm_idx))
    chosen = ctx.rng.choice(harm_idx, size=take, replace=False)
    cross = vectors[chosen] @ vectors[safe_idx].T      # [take, n_safe]
    within = vectors[chosen] @ vectors[harm_idx].T     # [take, n_harm]

    out = []
    for row, anchor_pos in enumerate(chosen):
        anchor = records[anchor_pos]
        negative = records[safe_idx[int(np.argmax(cross[row]))]]
        order = np.argsort(within[row])[::-1]
        positive = next(
            (records[harm_idx[j]] for j in order if harm_idx[j] != anchor_pos), None
        )
        if positive is None:
            continue
        out.append(make_pair("benign_twin", ctx.split, anchor, positive["text"], negative,
                             anchor_label=1, variant="cosine_mined",
                             positive_source="record", positive_record=positive))
    return out


BUILDERS: dict[str, Callable[[PairContext, int], list[dict]]] = {
    "safe_harm_contrast": build_safe_harm_contrast,
    "paraphrase": build_paraphrase,
    "jailbreak_variant": build_jailbreak_variant,
    "benign_twin": build_benign_twin,
}


# --- Сборка --------------------------------------------------------------------

def deduplicate(pairs: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for pair in pairs:
        if pair["anchor"].strip() == pair["positive"].strip():
            continue
        if pair["pair_id"] in seen:
            continue
        seen.add(pair["pair_id"])
        out.append(pair)
    return out


def flatten_texts(pairs: list[dict]) -> list[dict]:
    """Все тексты из пар с бинарной меткой, без повторов.

    Это обучающее множество для classification-ветки RQ3: те же самые тексты, что
    видит contrastive-ветка, иначе сравнение объективов не контролируемое.
    """
    seen: set[str] = set()
    out: list[dict] = []
    for pair in pairs:
        items = [
            (pair["anchor"], pair["anchor_label"], pair["anchor_source"], pair["pair_type"], pair["variant"]),
            (pair["positive"], pair["anchor_label"], pair["anchor_source"], pair["pair_type"], pair["variant"]),
            (pair["negative"], pair["negative_label"], pair["negative_source"], pair["pair_type"], pair["variant"]),
        ]
        for text, label, source, pair_type, variant in items:
            key = hashlib.sha1(text.strip().encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "id": key[:16],
                "text": text,
                "label": int(label),
                "label_name": "harm" if label == 1 else "safe",
                "source": source,
                "origin_pair_type": pair_type,
                "variant": variant,
                "split": pair["split"],
                "n_chars": len(text),
            })
    return out
