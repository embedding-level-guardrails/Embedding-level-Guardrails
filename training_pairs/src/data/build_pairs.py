"""Build training pairs (safe/harm contrast, paraphrase, jailbreak variant,
benign twin) from AEGIS 2.0 and HarmBench — data prep for RQ3 (contrastive
fine-tuning). Also builds a separate, comparably-structured pair set for the
code domain from a small hand-authored bank (`code_bank.py`), since none of
the general-purpose sources cover it densely enough on their own.

Pair types (main set)
----------------------
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

benign_twin:
    (safe prompt, same prompt wrapped in a jailbreak template) pairs — the
    TwinBreak-style twin of jailbreak_variant: same template family, but
    wrapped around an AEGIS *safe* prompt instead of a HarmBench harmful
    behavior, both sides labeled "safe". Paired with jailbreak_variant for
    RQ2: isolates whether the wrapper/style alone shifts an encoder's
    embedding, independent of the underlying harmful content.

The code-domain set mirrors all four pair types (`code_safe_harm_contrast`,
`code_paraphrase`, `code_jailbreak_variant`, `code_benign_twin`) built purely
from `code_bank.CODE_PAIRS`, written separately to `data/processed/code/`.

Sizes are fixed via CLI defaults (documented in training_pairs/data/README.md)
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

from training_pairs.src.data.code_bank import CODE_PAIRS
from training_pairs.src.data.sources import (
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


def build_benign_twin_pairs(
    aegis_df: pd.DataFrame,
    templates: list[str],
    n: int,
    templates_per_prompt: int = 5,
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)
    safe_prompts = aegis_df[aegis_df["prompt_label"] == "safe"]["prompt"].tolist()
    rng.shuffle(safe_prompts)

    pairs: list[dict] = []
    for prompt in safe_prompts:
        if len(pairs) >= n:
            break
        picked_templates = rng.sample(templates, k=min(templates_per_prompt, len(templates)))
        for template in picked_templates:
            if len(pairs) >= n:
                break
            pairs.append(
                {
                    "pair_type": "benign_twin",
                    "anchor_text": prompt,
                    "anchor_label": "safe",
                    "pair_text": f"{template}\n\n{prompt}",
                    "pair_label": "safe",
                    "category": None,
                    "source": "aegis+harmbench",
                }
            )
    return pairs


def build_code_contrast_pairs(code_bank: list[dict] = CODE_PAIRS, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    pairs = []
    for entry in code_bank:
        for m in entry["malicious"]:
            for b in entry["benign_twin"]:
                pairs.append(
                    {
                        "pair_type": "code_safe_harm_contrast",
                        "anchor_text": m,
                        "anchor_label": "harm",
                        "pair_text": b,
                        "pair_label": "safe",
                        "category": entry["category"],
                        "source": "code_bank",
                    }
                )
    rng.shuffle(pairs)
    return pairs


def build_code_paraphrase_pairs(code_bank: list[dict] = CODE_PAIRS, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    pairs = []
    for entry in code_bank:
        for key, label in (("malicious", "harm"), ("benign_twin", "safe")):
            texts = entry[key]
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    pairs.append(
                        {
                            "pair_type": "code_paraphrase",
                            "anchor_text": texts[i],
                            "anchor_label": label,
                            "pair_text": texts[j],
                            "pair_label": label,
                            "category": entry["category"],
                            "source": "code_bank",
                        }
                    )
    rng.shuffle(pairs)
    return pairs


def build_code_jailbreak_pairs(
    templates: list[str],
    code_bank: list[dict] = CODE_PAIRS,
    templates_per_text: int = 3,
    seed: int = 42,
    twin: bool = False,
) -> list[dict]:
    """`twin=False` builds code_jailbreak_variant (malicious text + wrapper, both
    "harm"); `twin=True` builds code_benign_twin (benign_twin text + the same
    wrapper family, both "safe") — the code-domain analog of jailbreak_variant
    vs. benign_twin above.
    """
    rng = random.Random(seed)
    key = "benign_twin" if twin else "malicious"
    label = "safe" if twin else "harm"
    pair_type = "code_benign_twin" if twin else "code_jailbreak_variant"

    pairs: list[dict] = []
    for entry in code_bank:
        for text in entry[key]:
            picked_templates = rng.sample(templates, k=min(templates_per_text, len(templates)))
            for template in picked_templates:
                pairs.append(
                    {
                        "pair_type": pair_type,
                        "anchor_text": text,
                        "anchor_label": label,
                        "pair_text": f"{template}\n\n{text}",
                        "pair_label": label,
                        "category": entry["category"],
                        "source": "code_bank",
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
    parser.add_argument("--n-safe-harm", type=int, default=500)
    parser.add_argument("--n-paraphrase", type=int, default=500)
    parser.add_argument("--n-jailbreak", type=int, default=500)
    parser.add_argument("--n-benign-twin", type=int, default=500)
    parser.add_argument("--templates-per-behavior", type=int, default=5)
    parser.add_argument("--code-templates-per-text", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-code", action="store_true", help="skip the separate code-domain pair set")
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
    pairs += build_benign_twin_pairs(
        aegis_df,
        templates,
        args.n_benign_twin,
        templates_per_prompt=args.templates_per_behavior,
        seed=args.seed,
    )

    manifest = write_dataset(pairs, args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

    if not args.skip_code:
        code_pairs = []
        code_pairs += build_code_contrast_pairs(seed=args.seed)
        code_pairs += build_code_paraphrase_pairs(seed=args.seed)
        code_pairs += build_code_jailbreak_pairs(
            templates, templates_per_text=args.code_templates_per_text, seed=args.seed, twin=False
        )
        code_pairs += build_code_jailbreak_pairs(
            templates, templates_per_text=args.code_templates_per_text, seed=args.seed, twin=True
        )
        code_manifest = write_dataset(code_pairs, args.output_dir / "code")
        print(json.dumps(code_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
