"""Build training pairs (safe/harm contrast, paraphrase, jailbreak variant)
from AEGIS 2.0 and HarmBench for RQ4 (п.4 плана: сбор пар для обучения).

Pair types
----------
safe_harm_contrast:
    (unsafe prompt, safe prompt) sampled from AEGIS. The core contrastive
    signal: two different-label points that should end up far apart.

paraphrase:
    (prompt, prompt) pairs mined *within* AEGIS: for each (label, category)
    group, near-duplicate prompts are found via TF-IDF cosine similarity in a
    similarity band that excludes both unrelated prompts (too dissimilar) and
    literal duplicates (too similar). Both sides keep the same label — this
    pair type tests whether an encoder keeps same-intent reformulations close.

jailbreak_variant:
    (behavior, jailbreak-wrapped behavior) pairs built from HarmBench: a
    canonical harmful behavior vs. the same behavior wrapped in one of the 114
    static human-jailbreak templates HarmBench ships with. Both sides are
    "harm" by intent — this pair type tests whether an encoder is fooled by
    adversarial surface style into looking safe.

Sizes are fixed via CLI defaults (documented in rq4_training_pairs/data/README.md)
so every run of this script with the default seed reproduces the same dataset sizes.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rq4_training_pairs.src.data.sources import (
    load_aegis,
    load_harmbench_behaviors,
    load_jailbreak_templates,
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MAX_GROUP_SIZE = 300  # bounds the O(group^2) TF-IDF similarity computation
MIN_PROMPT_LEN = 15


def build_safe_harm_pairs(aegis_df: pd.DataFrame, n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    unsafe = aegis_df[aegis_df["prompt_label"] == "unsafe"].to_dict("records")
    safe = aegis_df[aegis_df["prompt_label"] == "safe"].to_dict("records")
    rng.shuffle(unsafe)
    rng.shuffle(safe)
    k = min(n, len(unsafe), len(safe))
    pairs = []
    for i in range(k):
        u, s = unsafe[i], safe[i]
        pairs.append(
            {
                "pair_type": "safe_harm_contrast",
                "anchor_text": u["prompt"],
                "anchor_label": "harm",
                "pair_text": s["prompt"],
                "pair_label": "safe",
                "category": u["violated_categories"],
                "source": "aegis",
            }
        )
    return pairs


def build_paraphrase_pairs(
    aegis_df: pd.DataFrame,
    n: int,
    seed: int = 42,
    min_sim: float = 0.35,
    max_sim: float = 0.95,
) -> list[dict]:
    rng = random.Random(seed)
    df = aegis_df[aegis_df["prompt"].str.len() >= MIN_PROMPT_LEN]
    candidates: list[dict] = []

    for (label, category), group in df.groupby(["prompt_label", "violated_categories"]):
        if len(group) < 2:
            continue
        if len(group) > MAX_GROUP_SIZE:
            group = group.sample(n=MAX_GROUP_SIZE, random_state=seed)
        texts = group["prompt"].tolist()

        tfidf = TfidfVectorizer().fit_transform(texts)
        sim = cosine_similarity(tfidf)
        np.fill_diagonal(sim, 0.0)

        above_i, above_j = np.where((sim >= min_sim) & (sim <= max_sim))
        for i, j in zip(above_i, above_j):
            if i >= j:
                continue
            candidates.append(
                {
                    "pair_type": "paraphrase",
                    "anchor_text": texts[i],
                    "anchor_label": label,
                    "pair_text": texts[j],
                    "pair_label": label,
                    "category": category,
                    "source": "aegis",
                    "_score": float(sim[i, j]),
                }
            )

    rng.shuffle(candidates)
    selected = candidates[:n]
    for c in selected:
        c.pop("_score", None)
    return selected


def build_jailbreak_pairs(
    behaviors_df: pd.DataFrame,
    templates: list[str],
    n: int,
    templates_per_behavior: int = 5,
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)
    behaviors = behaviors_df.to_dict("records")
    rng.shuffle(behaviors)

    pairs: list[dict] = []
    for behavior_row in behaviors:
        if len(pairs) >= n:
            break
        behavior = behavior_row["Behavior"]
        context = behavior_row.get("ContextString")
        if isinstance(context, str) and context:
            behavior = f"{context}\n\n---\n\n{behavior}"

        picked_templates = rng.sample(templates, k=min(templates_per_behavior, len(templates)))
        for template in picked_templates:
            if len(pairs) >= n:
                break
            pairs.append(
                {
                    "pair_type": "jailbreak_variant",
                    "anchor_text": behavior,
                    "anchor_label": "harm",
                    "pair_text": f"{template}\n\n{behavior}",
                    "pair_label": "harm",
                    "category": behavior_row.get("SemanticCategory"),
                    "source": "harmbench",
                }
            )
    return pairs


def _sanitize(value):
    """Drop lone UTF-16 surrogates that sneak in from source text encoding issues."""
    if isinstance(value, str):
        return value.encode("utf-8", "replace").decode("utf-8")
    return value


def write_dataset(pairs: list[dict], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "pairs.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(pairs):
            clean_row = {k: _sanitize(v) for k, v in row.items()}
            f.write(json.dumps({"id": i, **clean_row}, ensure_ascii=False) + "\n")

    counts_by_type: dict[str, int] = {}
    counts_by_type_and_category: dict[str, dict[str, int]] = {}
    for row in pairs:
        pt = row["pair_type"]
        counts_by_type[pt] = counts_by_type.get(pt, 0) + 1
        cat = row.get("category") or "(none)"
        counts_by_type_and_category.setdefault(pt, {})
        counts_by_type_and_category[pt][cat] = counts_by_type_and_category[pt].get(cat, 0) + 1

    manifest = {
        "total_pairs": len(pairs),
        "counts_by_type": counts_by_type,
        "counts_by_type_and_category": counts_by_type_and_category,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-safe-harm", type=int, default=1000)
    parser.add_argument("--n-paraphrase", type=int, default=500)
    parser.add_argument("--n-jailbreak", type=int, default=500)
    parser.add_argument("--templates-per-behavior", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    aegis_df = load_aegis()
    behaviors_df = load_harmbench_behaviors()
    templates = load_jailbreak_templates()

    pairs = []
    pairs += build_safe_harm_pairs(aegis_df, args.n_safe_harm, seed=args.seed)
    pairs += build_paraphrase_pairs(aegis_df, args.n_paraphrase, seed=args.seed)
    pairs += build_jailbreak_pairs(
        behaviors_df,
        templates,
        args.n_jailbreak,
        templates_per_behavior=args.templates_per_behavior,
        seed=args.seed,
    )

    manifest = write_dataset(pairs, args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
