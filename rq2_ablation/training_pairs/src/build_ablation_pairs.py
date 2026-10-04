"""Данные для абляции RQ2: какие тексты добавлять в обучение CE.

Пять плеч на одном и том же наборе из N harm-промптов. Отличаются только тем, какие safe-промпты и
какие обёрнутые тексты добавлены к этим harm:

    plain             случайные safe
    contrast          safe из safe_harm_contrast — самые похожие на harm по TF-IDF (hard negatives)
    plain+wrap        plain + jailbreak_variant + benign_twin
    contrast+wrap     contrast + jailbreak_variant + benign_twin
    plain+jailbreak   plain + только jailbreak_variant — проверка, зачем нужен benign_twin

CE учится на каждом тексте независимо и спаривания не видит, поэтому основной выход — texts.jsonl
(уровень текстов). pairs.jsonl хранит происхождение текстов и match_score контрастов.

Что держится постоянным, чтобы сравнения были чистыми:

* набор harm-промптов одинаков во всех плечах — plain и contrast отличаются только выбором safe;
* обёртки добавляются к сырым промптам, а не заменяют их — у плеч с обёртками те же исходные запросы;
* jailbreak_variant (какие harm оборачиваются и в какие шаблоны) одинаков во всех плечах с обёртками;
  benign_twin подбирается из safe своего плеча по длине и оборачивается в те же шаблоны;
* разбиение на категории и фильтры пула — те же, что в основном эксперименте (build_pairs_v2).

Запуск из корня репозитория:

    python rq2_ablation/training_pairs/src/build_ablation_pairs.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from training_pairs.src.data.build_pairs_v2 import (  # noqa: E402
    HARM,
    MAX_WRAPPED_PROMPT_CHARS,
    PROTECTED_CATEGORIES,
    SAFE,
    _clean,
    _side,
    build_safe_harm_pairs,
    is_jailbreak_like,
    load_aegis_split,
    make_pair,
    prepare_prompt_pool,
    sample_by_category,
    wrap_with_template,
)
from training_pairs.src.data.sources import load_jailbreak_templates  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"
WRAPPED_SOURCE = "aegis+harmbench_templates"

ARMS = {
    "plain": {"safe": "random", "jailbreak_variant": False, "benign_twin": False},
    "contrast": {"safe": "contrast", "jailbreak_variant": False, "benign_twin": False},
    "plain+wrap": {"safe": "random", "jailbreak_variant": True, "benign_twin": True},
    "contrast+wrap": {"safe": "contrast", "jailbreak_variant": True, "benign_twin": True},
    "plain+jailbreak": {"safe": "random", "jailbreak_variant": True, "benign_twin": False},
}


def raw_texts(rows: pd.DataFrame, origin: str) -> list[dict]:
    """Сырые промпты пула как тексты обучения."""
    return [
        {
            "text": _clean(row["prompt"]),
            "label": row["label"],
            "category": row["primary_category"],
            "source_id": str(row["id"]),
            "kind": "raw",
            "origin": origin,
            "template_id": None,
            "match_score": None,
        }
        for row in rows.to_dict("records")
    ]


def pair_side_texts(pairs: list[dict], side: str, kind: str) -> list[dict]:
    """Одна сторона пар как тексты обучения; origin — тип пары, который этот текст привёл."""
    return [
        {
            # очистка сразу при создании: дедупликация и отпечатки видят тот же текст, что запишется
            "text": _clean(pair[f"{side}_text"]),
            "label": pair[f"{side}_label"],
            "category": pair[f"{side}_category"],
            "source_id": pair[f"{side}_source_id"],
            "kind": kind,
            "origin": pair["pair_type"],
            "template_id": pair.get("template_id"),
            "match_score": pair.get("match_score"),
        }
        for pair in pairs
    ]


def wrapping_plan(harm: pd.DataFrame, templates: list[str], n_prompts: int,
                  templates_per_prompt: int, rng: np.random.Generator) -> list[tuple[dict, list[int]]]:
    """Какие harm оборачиваются и в какие шаблоны. Строится один раз и общий для всех плеч.

    Джейлбрейк не оборачивается в другой джейлбрейк, длинные запросы не оборачиваются — те же
    ограничения, что в build_wrapped_pairs основного эксперимента.
    """
    eligible = harm[(harm["chars"] <= MAX_WRAPPED_PROMPT_CHARS) & ~harm["prompt"].map(is_jailbreak_like)]
    chosen = sample_by_category(eligible, n_prompts, rng)
    size = min(templates_per_prompt, len(templates))
    return [(row, rng.choice(len(templates), size=size, replace=False).tolist()) for row in chosen.to_dict("records")]


def jailbreak_variant_pairs(plan: list[tuple[dict, list[int]]], templates: list[str]) -> list[dict]:
    return [
        make_pair("jailbreak_variant", _side(row), _side(row, wrap_with_template(templates[template_id], row["prompt"])),
                  WRAPPED_SOURCE, template_id=template_id)
        for row, template_ids in plan
        for template_id in template_ids
    ]


def benign_twin_pairs(plan: list[tuple[dict, list[int]]], safe: pd.DataFrame, templates: list[str],
                      seed: int) -> list[dict]:
    """Каждому harm из плана — неиспользованный safe своего плеча ближайшей длины в тех же шаблонах.

    Так обёрнутые harm и safe отличаются только содержанием запроса, как в основном эксперименте.
    Порядок safe перемешивается, чтобы равные по длине кандидаты выбирались случайно.
    """
    eligible = safe[(safe["chars"] <= MAX_WRAPPED_PROMPT_CHARS) & ~safe["prompt"].map(is_jailbreak_like)]
    eligible = eligible.iloc[np.random.default_rng(seed).permutation(len(eligible))].reset_index(drop=True)
    lengths = eligible["chars"].to_numpy()
    used = np.zeros(len(eligible), dtype=bool)

    pairs = []
    for harm_row, template_ids in plan:
        gaps = np.where(used, np.inf, np.abs(lengths - harm_row["chars"]))
        if not np.isfinite(gaps).any():
            break
        match = int(np.argmin(gaps))
        used[match] = True
        safe_row = eligible.iloc[match].to_dict()
        pairs += [
            make_pair("benign_twin", _side(safe_row),
                      _side(safe_row, wrap_with_template(templates[template_id], safe_row["prompt"])),
                      WRAPPED_SOURCE, template_id=template_id)
            for template_id in template_ids
        ]
    return pairs


def deduplicate(texts: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(texts).drop_duplicates("text", keep="first").reset_index(drop=True)
    frame["template_id"] = frame["template_id"].astype("Int64")   # иначе None у сырых превратит колонку во float
    return frame


def build_arms(pool: pd.DataFrame, templates: list[str], n_harm: int, n_wrapped_prompts: int,
               templates_per_prompt: int = 2, seed: int = 42, arms: tuple[str, ...] = tuple(ARMS)) -> dict:
    """Все плечи абляции. Без ввода-вывода — чтобы тесты могли вызывать напрямую."""
    harm_pool = pool[pool["label"] == HARM]
    safe_pool = pool[pool["label"] == SAFE]

    harm = sample_by_category(harm_pool, n_harm, np.random.default_rng(seed))
    n_safe = len(harm)
    if len(safe_pool) < n_safe:
        raise ValueError(f"safe в пуле {len(safe_pool)}, нужно {n_safe}")

    safe_random = safe_pool.iloc[np.random.default_rng(seed + 1).permutation(len(safe_pool))[:n_safe]]

    contrast_pairs = []
    if any(ARMS[arm]["safe"] == "contrast" for arm in arms):
        # пул ограничен ровно этим harm-набором, поэтому build_safe_harm_pairs подбирает пары к нему
        contrast_pairs = build_safe_harm_pairs(pd.concat([harm, safe_pool]), n_safe, seed=seed)
    contrast_ids = {pair["pair_source_id"] for pair in contrast_pairs}
    safe_contrast = safe_pool[safe_pool["id"].astype(str).isin(contrast_ids)]

    plan = wrapping_plan(harm, templates, n_wrapped_prompts, templates_per_prompt, np.random.default_rng(seed + 2))
    jailbreak_pairs = jailbreak_variant_pairs(plan, templates)

    result = {}
    for arm in arms:
        spec = ARMS[arm]
        safe = safe_random if spec["safe"] == "random" else safe_contrast
        texts = raw_texts(harm, "harm_base")
        pairs: list[dict] = []
        if spec["safe"] == "random":
            texts += raw_texts(safe_random, "safe_random")
        else:
            texts += pair_side_texts(contrast_pairs, "pair", "raw")
            pairs += contrast_pairs
        if spec["jailbreak_variant"]:
            texts += pair_side_texts(jailbreak_pairs, "pair", "wrapped")
            pairs += jailbreak_pairs
        if spec["benign_twin"]:
            twins = benign_twin_pairs(plan, safe, templates, seed + 3)
            texts += pair_side_texts(twins, "pair", "wrapped")
            pairs += twins
        result[arm] = {"texts": deduplicate(texts), "pairs": pairs}
    return result


def fingerprint(values) -> str:
    return hashlib.sha1("\n".join(sorted(map(str, values))).encode("utf-8")).hexdigest()[:16]


def describe(arm: str, texts: pd.DataFrame, pairs: list[dict]) -> dict:
    """Что лежит в плече: всё, что надо показывать рядом с результатами, чтобы видеть различия."""
    harm = texts[(texts["label"] == HARM) & (texts["kind"] == "raw")]
    scores = [pair["match_score"] for pair in pairs if pair["pair_type"] == "safe_harm_contrast"]
    return {
        "arm": arm,
        "spec": ARMS[arm],
        "texts": len(texts),
        "texts_by_label": texts["label"].value_counts().to_dict(),
        "texts_by_kind": texts["kind"].value_counts().to_dict(),
        "texts_by_origin": texts["origin"].value_counts().to_dict(),
        "distinct_requests": int(texts["source_id"].nunique()),
        "wrapped_share": round(float((texts["kind"] == "wrapped").mean()), 4),
        "pairs_by_type": pd.Series([pair["pair_type"] for pair in pairs], dtype=object).value_counts().to_dict(),
        "harm_categories": harm["category"].value_counts().to_dict(),
        "harm_fingerprint": fingerprint(harm["source_id"]),
        "contrast_match_score": (
            {key: round(float(value), 4) for key, value in pd.Series(scores).describe(percentiles=[.25, .5, .75]).items()}
            if scores else None
        ),
        "median_chars": float(texts["text"].str.len().median()),
    }


def to_json(value):
    """Та же очистка, что в основном сборщике: numpy → Python, NaN → None, одиночные суррогаты
    в шаблонах HarmBench → «?». Иначе обёрнутые тексты разошлись бы с основным набором побайтово."""
    return None if value is pd.NA else _clean(value)


def write_arm(output_dir: Path, arm: str, texts: pd.DataFrame, pairs: list[dict], extra: dict) -> dict:
    arm_dir = output_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    with (arm_dir / "texts.jsonl").open("w", encoding="utf-8") as file:
        for record in texts.to_dict("records"):
            file.write(json.dumps({key: to_json(value) for key, value in record.items()}, ensure_ascii=False) + "\n")
    with (arm_dir / "pairs.jsonl").open("w", encoding="utf-8") as file:
        for index, record in enumerate(pairs):
            clean = {key: to_json(value) for key, value in record.items()}
            file.write(json.dumps({"id": index, **clean}, ensure_ascii=False, default=str) + "\n")
    manifest = {**describe(arm, texts, pairs), **extra}
    (arm_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-harm", type=int, default=5000, help="harm-промптов, общих для всех плеч; safe столько же")
    parser.add_argument("--n-wrapped-prompts", type=int, default=600,
                        help="harm-промптов, которые оборачиваются (и столько же safe-двойников)")
    parser.add_argument("--templates-per-prompt", type=int, default=2)
    parser.add_argument("--min-category-prompts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42, help="seed данных; seed обучения задаётся в блокноте отдельно")
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    args = parser.parse_args()

    # пул собирается так же, как в основном эксперименте: те же фильтры, то же слияние категорий,
    # персонажные джейлбрейки исключены (вариант A), промпты val и test исключены
    train = load_aegis_split("train")
    held_out = pd.concat([load_aegis_split("validation")["prompt"], load_aegis_split("test")["prompt"]])
    exclude = frozenset(held_out.dropna().astype(str).str.strip())
    pool, pool_info = prepare_prompt_pool(train, exclude, args.min_category_prompts,
                                          protected_categories=PROTECTED_CATEGORIES, drop_persona_jailbreaks=True)
    templates = load_jailbreak_templates()

    arms = build_arms(pool, templates, args.n_harm, args.n_wrapped_prompts, args.templates_per_prompt,
                      args.seed, tuple(args.arms))
    shared = {"parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "category_map": pool_info["category_map"], "template_count": len(templates)}

    summary = {}
    for arm, data in arms.items():
        manifest = write_arm(args.output_dir, arm, data["texts"], data["pairs"], shared)
        summary[arm] = {key: manifest[key] for key in ("texts", "texts_by_label", "distinct_requests",
                                                       "wrapped_share", "harm_fingerprint", "median_chars")}
        print(f"{arm:16s} текстов {manifest['texts']:6d} · harm/safe "
              f"{manifest['texts_by_label'].get(HARM, 0)}/{manifest['texts_by_label'].get(SAFE, 0)} · "
              f"запросов {manifest['distinct_requests']:6d} · обёрнутых {manifest['wrapped_share']:.0%}")

    fingerprints = {item["harm_fingerprint"] for item in summary.values()}
    checks = {"harm_set_identical_across_arms": len(fingerprints) == 1}
    jailbreak_sets = {
        arm: fingerprint(data["texts"].loc[data["texts"]["origin"] == "jailbreak_variant", "text"])
        for arm, data in arms.items() if ARMS[arm]["jailbreak_variant"]
    }
    if jailbreak_sets:
        checks["jailbreak_variant_identical_across_arms"] = len(set(jailbreak_sets.values())) == 1
    (args.output_dir / "summary.json").write_text(
        json.dumps({"arms": summary, "checks": checks, "pool": pool_info["stats"]}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print("\nпроверки:", checks)


if __name__ == "__main__":
    main()
