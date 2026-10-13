"""Сборка триплетов из «чистого» датасета, режим coverage.

Чем отличается от прежней сборки (builders.py, размеры из `sizes`), и какую проблему
закрывает каждое отличие:

1. Только настоящие тексты датасета. Нет шаблонных парафраз и jailbreak-обёрток,
   нет HarmBench. Раньше шаблонные тексты занимали ~24% train.

2. Каждый текст train — якорь ровно одного триплета. Раньше якоря выбирались
   случайно с возвращением: ~14% текстов не попадали в обучение вовсе, а другие
   повторялись. Теперь множество текстов в триплетах совпадает с train целиком,
   и CE / SupCon можно учить прямо на train — это те же тексты. Заодно число
   триплетов равно числу текстов, и у всех объективов одинаковое число шагов
   (раньше у CE было 3060 против 2032).

3. Подозрение на ошибку разметки. Поиск ближайших соседей непропорционально часто
   вытаскивает ошибочно размеченные тексты: «безопасный» текст, который на самом
   деле вредный, ближе всех к вредным якорям (так «How to make a bomb?» с меткой
   unharmful стоял негативом в 13 триплетах). Поэтому для каждого текста
   считается вероятность противоположного класса out-of-fold логистической
   регрессией поверх эмбеддингов исходного e5. Тексты, где она выше порога,
   остаются в данных как якоря с исходной меткой (датасет не правится), но не
   используются как позитивы и негативы.

4. Ограничение на переиспользование негативов. Один и тот же «хаб» больше не может
   стоять негативом в десятках триплетов: не больше `max_negative_uses` раз.

5. Выбор из нескольких ближайших, а не самого ближайшего: случайный из `top_k`
   подходящих, с лимитом повторов и для негативов, и для позитивов
   (`max_negative_uses`, `max_positive_uses`). Меньше упора в один и тот же «хаб».

Что проверили и НЕ стали делать. Позитив adversarial-якоря пробовали брать как
ближайший по эмбеддингу vanilla-промт того же класса — в надежде найти «исходную»
формулировку того же запроса. Не работает: исходный e5 видит прежде всего форму,
и ближайшим оказывается другой jailbreak-похожий текст (один длинный ролевой промт
стал позитивом для 554 якорей). Поэтому позитив adversarial-якоря — случайный
vanilla-промт того же класса и категории, как и раньше.

Что осталось как было: соседи ищутся один раз по исходному e5 (статический майнинг),
а не текущей моделью в процессе обучения. И детектор ошибок разметки работает на
тех же эмбеддингах, поэтому ловит только то, что для исходного e5 уверенно выглядит
другим классом: «How to make a bomb?» он не ловит (P(harm) = 0.21) — от таких
текстов защищает только лимит повторов.

Типы триплетов и якоря:
  adversarial_contrast — все adversarial-промты (оба класса):
      позитив — случайный vanilla того же класса (для вредных — той же категории),
      негатив — один из ближайших adversarial другого класса (та же форма, другое
      содержание).
  benign_twin — доля `benign_twin_fraction` vanilla-вредных промтов:
      позитив — ближайший вредный, негатив — ближайший безопасный.
  safe_harm_contrast — все остальные vanilla-промты (безопасные и вредные):
      позитив — случайный того же класса (для вредных — той же категории),
      негатив — случайный другого класса.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from ..utils import get_logger, l2_normalize
from .builders import make_pair

logger = get_logger(__name__)


@dataclass
class CoverageConfig:
    benign_twin_fraction: float = 0.3   # доля vanilla-вредных якорей, идущих в benign_twin
    top_k: int = 5                      # выбор случайного из k ближайших подходящих
    search_depth: int = 100             # сколько ближайших смотреть, если первые k заняты
    max_negative_uses: int = 3          # сколько раз текст может быть майненым негативом
    max_positive_uses: int = 3          # сколько раз текст может быть майненым позитивом
    noise_threshold: float = 0.9        # P(другой класс) выше — текст «подозрительный»
    noise_folds: int = 5
    noise_c: float = 1.0
    block_size: int = 512

    @classmethod
    def from_dict(cls, raw: dict | None) -> "CoverageConfig":
        raw = dict(raw or {})
        known = cls.__dataclass_fields__
        unknown = set(raw) - set(known)
        if unknown:
            raise ValueError(f"Незнакомые ключи в pairs.coverage: {sorted(unknown)}")
        return cls(**raw)


@dataclass
class CoverageReport:
    n_anchors: int = 0
    n_pairs: int = 0
    by_type: dict = field(default_factory=dict)
    n_suspect: int = 0
    suspect_examples: list = field(default_factory=list)
    n_fallback: int = 0
    max_negative_uses: int = 0
    max_positive_uses: int = 0
    frozen_negative_closer: float = float("nan")
    texts_cover_pool: bool = False


# --- Подозрение на ошибку разметки --------------------------------------------------

def out_of_fold_harm_probability(x: np.ndarray, y: np.ndarray, folds: int = 5, c: float = 1.0,
                                 seed: int = 42) -> np.ndarray:
    """P(harm) для каждого текста от модели, которая этот текст не видела.

    Логистическая регрессия на нормированных эмбеддингах исходного e5, k-fold: каждый
    текст оценивается моделью, обученной на остальных фолдах. Без этого ошибочно
    размеченный текст был бы в обучающей выборке своего же классификатора.
    """
    v = l2_normalize(x.astype(np.float64))
    folds = int(min(folds, np.bincount(y).min()))
    if folds < 2:
        return np.where(y == 1, 1.0, 0.0)
    p = np.zeros(len(y), dtype=np.float64)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_idx, held_idx in splitter.split(v, y):
        clf = LogisticRegression(C=c, max_iter=2000, class_weight="balanced", random_state=seed)
        clf.fit(v[train_idx], y[train_idx])
        p[held_idx] = clf.predict_proba(v[held_idx])[:, 1]
    return p


# --- Поиск ближайших с ограничением --------------------------------------------------

class NeighbourPicker:
    """Случайный из top_k ближайших подходящих кандидатов с учётом лимита повторов.

    Ближайшие кандидаты для всех якорей одной группы считаются блоками заранее
    (`search_depth` штук), дальше выбор идёт по порядку якорей — детерминированно
    при фиксированном генераторе. Повторы считаются отдельно по ролям.
    """

    def __init__(self, vectors: np.ndarray, rng: np.random.Generator, cfg: CoverageConfig):
        self.v = vectors
        self.rng = rng
        self.cfg = cfg
        self.usage: dict[str, Counter] = {"positive": Counter(), "negative": Counter()}

    def nearest(self, anchors: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """[len(anchors), depth] индексов кандидатов по убыванию косинуса (−1 = нет)."""
        depth = min(self.cfg.search_depth, len(candidates))
        out = np.full((len(anchors), self.cfg.search_depth), -1, dtype=np.int64)
        if depth == 0 or len(anchors) == 0:
            return out
        cand = self.v[candidates]
        for start in range(0, len(anchors), self.cfg.block_size):
            rows = anchors[start : start + self.cfg.block_size]
            sim = self.v[rows] @ cand.T
            top = np.argpartition(-sim, depth - 1, axis=1)[:, :depth]
            order = np.argsort(-np.take_along_axis(sim, top, axis=1), axis=1, kind="stable")
            out[start : start + len(rows), :depth] = candidates[np.take_along_axis(top, order, axis=1)]
        return out

    def pick(self, ranked: np.ndarray, exclude: int, role: str) -> int | None:
        """Случайный из первых top_k подходящих в списке `ranked` для роли positive/negative."""
        limit = self.cfg.max_positive_uses if role == "positive" else self.cfg.max_negative_uses
        usage = self.usage[role]
        eligible = []
        for j in ranked:
            if j < 0:
                break
            if j == exclude or usage[j] >= limit:
                continue
            eligible.append(int(j))
            if len(eligible) == self.cfg.top_k:
                break
        if not eligible:
            return None
        choice = eligible[int(self.rng.integers(len(eligible)))]
        usage[choice] += 1
        return choice


# --- Сборка ------------------------------------------------------------------------

def build_split_coverage(split: str, records: list[dict], x: np.ndarray, rng: np.random.Generator,
                         cfg: CoverageConfig, seed: int = 42) -> tuple[list[dict], CoverageReport]:
    """Триплеты для одного сплита: каждая запись — якорь ровно одного триплета.

    `records` — очищенный пул сплита (без повторов и пересечений с другими сплитами),
    `x` — эмбеддинги исходного e5 построчно к `records`.
    """
    if len(records) != len(x):
        raise ValueError(f"{split}: записей {len(records)}, эмбеддингов {len(x)}")
    n = len(records)
    v = l2_normalize(x.astype(np.float64))
    y = np.array([int(r["label"]) for r in records])
    adv = np.array([bool(r.get("adversarial")) for r in records])
    report = CoverageReport(n_anchors=n)

    # 1. подозрение на ошибку разметки
    p_harm = out_of_fold_harm_probability(v, y, cfg.noise_folds, cfg.noise_c, seed)
    p_other = np.where(y == 1, 1.0 - p_harm, p_harm)
    suspect = p_other > cfg.noise_threshold
    report.n_suspect = int(suspect.sum())
    for i in np.argsort(-p_other)[: min(10, report.n_suspect)]:
        report.suspect_examples.append({"text": records[i]["text"][:160], "label": int(y[i]),
                                        "p_other_class": round(float(p_other[i]), 4)})

    # 2. группы кандидатов (подозрительные не бывают позитивами и негативами)
    ok = ~suspect
    group = {(lab, a): np.where((y == lab) & (adv == a) & ok)[0] for lab in (0, 1) for a in (False, True)}
    by_label = {lab: np.where((y == lab) & ok)[0] for lab in (0, 1)}

    # 3. тип триплета для каждого якоря
    vanilla_harm = np.where((y == 1) & ~adv)[0]
    twin_anchors = set(rng.choice(vanilla_harm, size=int(round(cfg.benign_twin_fraction * len(vanilla_harm))),
                                  replace=False).tolist()) if len(vanilla_harm) else set()
    kind = np.array(["adversarial_contrast" if adv[i] else
                     "benign_twin" if i in twin_anchors else
                     "safe_harm_contrast" for i in range(n)], dtype=object)

    # 4. заранее — ближайшие кандидаты для майненых типов
    picker = NeighbourPicker(v, rng, cfg)
    ranked: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray]] = {}
    for lab in (0, 1):
        anchors = np.where((kind == "adversarial_contrast") & (y == lab))[0]
        ranked[("adv", lab, "neg")] = (anchors, picker.nearest(anchors, group[(1 - lab, True)]))
    twins = np.where(kind == "benign_twin")[0]
    ranked[("twin", 1, "pos")] = (twins, picker.nearest(twins, by_label[1]))
    ranked[("twin", 1, "neg")] = (twins, picker.nearest(twins, by_label[0]))
    row_of = {key: {int(a): k for k, a in enumerate(anchors)} for key, (anchors, _) in ranked.items()}

    def mined(key: tuple, anchor: int, role: str) -> int | None:
        anchors, table = ranked[key]
        return picker.pick(table[row_of[key][anchor]], exclude=anchor, role=role)

    def random_other(pool: np.ndarray, exclude: int) -> int | None:
        if len(pool) == 0 or (len(pool) == 1 and pool[0] == exclude):
            return None
        while True:                                   # исключаем сам якорь
            j = int(pool[int(rng.integers(len(pool)))])
            if j != exclude:
                return j

    # индексы по категории для случайного позитива той же категории
    def category_index(pool: np.ndarray) -> dict[object, np.ndarray]:
        cats = np.array([records[i].get("category") for i in pool], dtype=object)
        return {cat: pool[cats == cat] for cat in set(cats.tolist())}

    by_category = {lab: category_index(by_label[lab]) for lab in (0, 1)}
    vanilla_by_category = {lab: category_index(group[(lab, False)]) for lab in (0, 1)}

    def random_same_class(i: int, lab: int, pool: np.ndarray, index: dict) -> int | None:
        """Случайный того же класса; для вредных — по возможности той же категории."""
        if lab == 1:
            same = index.get(records[i].get("category"), np.array([], dtype=np.int64))
            j = random_other(same, i)
            if j is not None:
                return j
        return random_other(pool, i)

    # 5. сборка, по порядку записей — детерминированно
    pairs: list[dict] = []
    for i in range(n):
        lab, t = int(y[i]), kind[i]
        variant = "nearest"
        if t == "adversarial_contrast":
            variant = "nearest_negative"
            pos = random_same_class(i, lab, group[(lab, False)], vanilla_by_category)
            neg = mined(("adv", lab, "neg"), i, role="negative")
        elif t == "benign_twin":
            pos = mined(("twin", 1, "pos"), i, role="positive")
            neg = mined(("twin", 1, "neg"), i, role="negative")
        else:
            variant = "random"
            pos = random_same_class(i, lab, by_label[lab], by_category)
            neg = random_other(by_label[1 - lab], i)

        if pos is None or neg is None:                # кандидаты кончились — случайные
            report.n_fallback += 1
            variant = "fallback"
            pos = pos if pos is not None else random_other(by_label[lab], i)
            neg = neg if neg is not None else random_other(by_label[1 - lab], i)
            if pos is None or neg is None:
                continue

        pair = make_pair(t, split, records[i], records[pos]["text"], records[neg], anchor_label=lab,
                         variant=variant, positive_source="record", positive_record=records[pos])
        pair["frozen_cos_positive"] = round(float(v[i] @ v[pos]), 4)
        pair["frozen_cos_negative"] = round(float(v[i] @ v[neg]), 4)
        pairs.append(pair)

    # 6. отчёт
    report.n_pairs = len(pairs)
    report.by_type = dict(Counter(p["pair_type"] for p in pairs))
    report.max_negative_uses = max(picker.usage["negative"].values(), default=0)
    report.max_positive_uses = max(picker.usage["positive"].values(), default=0)
    mined_pairs = [p for p in pairs if p["variant"] in ("nearest", "nearest_negative")]
    if mined_pairs:
        report.frozen_negative_closer = float(np.mean(
            [p["frozen_cos_negative"] > p["frozen_cos_positive"] for p in mined_pairs]))
    pair_texts = {p[k].strip() for p in pairs for k in ("anchor", "positive", "negative")}
    report.texts_cover_pool = pair_texts == {r["text"].strip() for r in records}
    return pairs, report
