"""Build training pairs v2 for RQ3 — a single source of requests (AEGIS 2.0).

What changed compared to v1 (`build_pairs.py`, left untouched):

* `jailbreak_variant` and `benign_twin` take requests for *both* classes from
  AEGIS 2.0 train (unsafe / safe); HarmBench is used only for its 114 static
  `HumanJailbreaks` wrappers. In v1 the harmful side came from HarmBench and the
  safe side from AEGIS, so the unwrapped request style (imperative vs.
  conversational) leaked the label.
* The request is substituted into the template's `{0}` placeholder instead of
  being appended after the template with the literal `{0}` left in the text.
  Templates without a placeholder get the request appended.
* Every harmful wrapped prompt gets a safe twin of similar length wrapped in the
  *same* templates, so wrapped harm and wrapped safe differ only by content.
* Prompts are filtered before any pair is built: empty / `REDACTED`, prompts
  with contradictory labels, duplicates, prompts that also occur in AEGIS
  validation or test, prompts that already look like jailbreaks, too short or
  too long prompts.
* Harm categories: `primary_category` is the first violated category after
  dropping the vague `Needs Caution` (AEGIS orders categories by annotation
  frequency); categories with fewer than `--min-category-prompts` prompts are
  merged into `Other`. The full raw list is kept in `*_categories`.
* Harmful prompts are sampled with square-root-of-frequency stratification over
  `primary_category`, so large categories do not dominate.
* `safe_harm_contrast` pairs every unsafe prompt with the most TF-IDF-similar
  unused safe prompt (hard negatives instead of random ones).
* `paraphrase` pairs are mined within one `primary_category` (harm) or among
  safe prompts, half of each.
* The code-domain set is rebuilt from `code_bank.CODE_PAIRS` with the same
  placeholder fix and matched templates; its harmful side is mapped to the AEGIS
  category `Malware`.

Output: `training_pairs/data/processed/v2/{pairs.jsonl,manifest.json}` and
`.../v2/code/{pairs.jsonl,manifest.json}`. The record schema is documented in
`training_pairs_v2.md` at the repository root.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from training_pairs.src.data.code_bank import CODE_PAIRS
from training_pairs.src.data.sources import RAW_DIR, _download, load_jailbreak_templates

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "v2"
AEGIS_SPLIT_URL = (
    "https://huggingface.co/api/datasets/nvidia/"
    "Aegis-AI-Content-Safety-Dataset-2.0/parquet/default/{split}/0.parquet"
)

HARM = "harm"
SAFE = "safe"
REDACTED = "REDACTED"
NEEDS_CAUTION = "Needs Caution"
OTHER_CATEGORY = "Other"
CODE_CATEGORY = "Malware"
# never merged into Other: the code-domain set relies on it
PROTECTED_CATEGORIES = frozenset({CODE_CATEGORY})

MIN_PROMPT_CHARS = 15
MAX_PROMPT_CHARS = 2000
MAX_WRAPPED_PROMPT_CHARS = 1000
MAX_GROUP_SIZE = 2000  # bounds the O(group^2) TF-IDF similarity computation
TEMPLATE_PLACEHOLDER = "{0}"

# Strong signals of a prompt that is already a jailbreak; `DAN` is case-sensitive so the name "Dan" does not match.
JAILBREAK_PATTERN = (
    r"(?-i:\bDAN\b)|do anything now|jailbr[eo]a?k"
    r"|ignore (?:all |any |your |the )?(?:previous |prior |above )?(?:instructions|rules|guidelines|restrictions)"
    r"|developer mode|dev mode|god mode|unfiltered|uncensored"
    r"|no (?:ethical|moral) (?:guidelines|restrictions)|without (?:any )?(?:filters|restrictions|limitations)"
)
_JAILBREAK_RE = re.compile(JAILBREAK_PATTERN, flags=re.IGNORECASE)


def load_aegis_split(split: str, cache_dir: Path = RAW_DIR) -> pd.DataFrame:
    """AEGIS 2.0 split (train / validation / test): id, prompt, prompt_label, violated_categories."""
    path = _download(AEGIS_SPLIT_URL.format(split=split), cache_dir / f"aegis_{split}.parquet")
    df = pd.read_parquet(path, columns=["id", "prompt", "prompt_label", "violated_categories"])
    return df[df["prompt_label"].isin(["safe", "unsafe"])].reset_index(drop=True)


def is_jailbreak_like(text: str) -> bool:
    return _JAILBREAK_RE.search(text) is not None


def wrap_with_template(template: str, request: str) -> str:
    if TEMPLATE_PLACEHOLDER in template:
        return template.replace(TEMPLATE_PLACEHOLDER, request)
    return f"{template}\n\n{request}"


def category_list(violated_categories) -> list[str]:
    if not isinstance(violated_categories, str):
        return []
    return [category.strip() for category in violated_categories.split(",") if category.strip()]


def primary_category(violated_categories) -> str | None:
    """First category after dropping Needs Caution; None if nothing is left."""
    categories = [category for category in category_list(violated_categories) if category != NEEDS_CAUTION]
    return categories[0] if categories else None


def prepare_prompt_pool(
    train_df: pd.DataFrame,
    exclude_prompts: frozenset[str] = frozenset(),
    min_category_prompts: int = 100,
    min_chars: int = MIN_PROMPT_CHARS,
    max_chars: int = MAX_PROMPT_CHARS,
    protected_categories: frozenset[str] = PROTECTED_CATEGORIES,
) -> tuple[pd.DataFrame, dict]:
    """Filtered AEGIS prompts with binary labels and merged primary categories.

    Returns the pool (id, prompt, label, categories, primary_category, chars) and a
    dict with per-filter removal counts and the category merge map.
    """
    stats: dict = {"rows": len(train_df)}
    df = train_df[["id", "prompt", "prompt_label", "violated_categories"]].copy()
    df = df[df["prompt"].notna()]
    df["prompt"] = df["prompt"].astype(str).str.strip()

    def drop(mask: pd.Series, name: str) -> None:
        nonlocal df
        stats[f"removed_{name}"] = int(mask.sum())
        df = df[~mask]

    drop((df["prompt"] == "") | (df["prompt"] == REDACTED), "redacted_or_empty")
    drop(df.groupby("prompt")["prompt_label"].transform("nunique") > 1, "inconsistent_label")
    drop(df.duplicated("prompt", keep="first"), "duplicate")
    drop(df["prompt"].isin(exclude_prompts), "in_validation_or_test")
    drop(df["prompt"].map(is_jailbreak_like), "jailbreak_like")
    lengths = df["prompt"].str.len()
    drop((lengths < min_chars) | (lengths > max_chars), "length")

    df = df.reset_index(drop=True)
    is_harm = (df["prompt_label"] == "unsafe").to_numpy()
    df["label"] = np.where(is_harm, HARM, SAFE)
    df["categories"] = [category_list(value) if harm else [] for value, harm in zip(df["violated_categories"], is_harm)]
    raw_primary = [
        (primary_category(value) or OTHER_CATEGORY) if harm else None
        for value, harm in zip(df["violated_categories"], is_harm)
    ]

    counts = Counter(category for category in raw_primary if category is not None)
    category_map = {
        category: category
        if count >= min_category_prompts or category in protected_categories or category == OTHER_CATEGORY
        else OTHER_CATEGORY
        for category, count in sorted(counts.items())
    }
    df["primary_category"] = pd.Series(
        [category_map[category] if category is not None else None for category in raw_primary],
        index=df.index,
        dtype=object,
    )
    df["chars"] = df["prompt"].str.len()

    stats["pool_harm"] = int(is_harm.sum())
    stats["pool_safe"] = int((~is_harm).sum())
    stats["harm_prompts_by_raw_category"] = dict(sorted(counts.items(), key=lambda item: -item[1]))
    stats["harm_prompts_by_category"] = df.loc[is_harm, "primary_category"].value_counts().to_dict()
    pool = df[["id", "prompt", "label", "categories", "primary_category", "chars"]]
    return pool, {"stats": stats, "category_map": category_map}


def allocate_by_category(counts: dict[str, int], n: int) -> dict[str, int]:
    """Split n across categories proportionally to sqrt(count), never above what a category has."""
    allocation = {category: 0 for category in counts}
    remaining = min(n, sum(counts.values()))
    while remaining > 0:
        open_categories = sorted(category for category in counts if allocation[category] < counts[category])
        if not open_categories:
            break
        weights = np.sqrt([counts[category] for category in open_categories])
        shares = weights / weights.sum() * remaining
        extra = np.floor(shares).astype(int)
        for index in np.argsort(-(shares - extra), kind="stable")[: remaining - int(extra.sum())]:
            extra[index] += 1

        added = 0
        for category, amount in zip(open_categories, extra):
            take = min(int(amount), counts[category] - allocation[category])
            allocation[category] += take
            added += take
        if added == 0:
            break
        remaining -= added
    return allocation


def sample_by_category(frame: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """sqrt-stratified sample of harmful prompts over primary_category, shuffled."""
    allocation = allocate_by_category(frame["primary_category"].value_counts().to_dict(), n)
    parts = []
    for category in sorted(allocation):
        if allocation[category] == 0:
            continue
        rows = frame[frame["primary_category"] == category]
        parts.append(rows.iloc[rng.permutation(len(rows))[: allocation[category]]])
    if not parts:
        return frame.iloc[:0]
    sample = pd.concat(parts)
    return sample.iloc[rng.permutation(len(sample))]


def _side(row: dict, text: str | None = None) -> dict:
    return {
        "text": row["prompt"] if text is None else text,
        "label": row["label"],
        "category": row["primary_category"],
        "categories": list(row["categories"]),
        "source_id": str(row["id"]),
    }


def make_pair(pair_type: str, anchor: dict, pair: dict, source: str, **extra) -> dict:
    record = {"pair_type": pair_type}
    for prefix, side in (("anchor", anchor), ("pair", pair)):
        for key, value in side.items():
            record[f"{prefix}_{key}"] = value
    record.update({"source": source, "template_id": None, "match_score": None, "subcategory": None})
    record.update(extra)
    return record


def build_safe_harm_pairs(pool: pd.DataFrame, n: int, seed: int = 42, max_similarity: float = 0.95) -> list[dict]:
    """(unsafe, most TF-IDF-similar unused safe) pairs — hard negatives."""
    rng = np.random.default_rng(seed)
    harm = sample_by_category(pool[pool["label"] == HARM], n, rng)
    safe = pool[pool["label"] == SAFE].reset_index(drop=True)
    harm = harm.iloc[: min(len(harm), len(safe))]
    if harm.empty:
        return []

    vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit(pd.concat([harm["prompt"], safe["prompt"]]))
    similarity = cosine_similarity(vectorizer.transform(harm["prompt"]), vectorizer.transform(safe["prompt"]))
    similarity[similarity >= max_similarity] = -1.0  # near-duplicates are not useful negatives

    used = np.zeros(len(safe), dtype=bool)
    pairs = []
    for index, harm_row in enumerate(harm.to_dict("records")):
        match = int(np.argmax(np.where(used, -np.inf, similarity[index])))
        used[match] = True
        pairs.append(
            make_pair(
                "safe_harm_contrast",
                _side(harm_row),
                _side(safe.iloc[match].to_dict()),
                "aegis",
                match_score=float(similarity[index, match]),
            )
        )
    return pairs


def mine_similar_pairs(texts: list[str], min_similarity: float, max_similarity: float) -> list[tuple[int, int, float]]:
    if len(texts) < 2:
        return []
    matrix = TfidfVectorizer(sublinear_tf=True).fit_transform(texts)
    similarity = cosine_similarity(matrix)
    rows, columns = np.triu_indices(len(texts), k=1)
    scores = similarity[rows, columns]
    keep = (scores >= min_similarity) & (scores <= max_similarity)
    return list(zip(rows[keep].tolist(), columns[keep].tolist(), scores[keep].tolist()))


def build_paraphrase_pairs(
    pool: pd.DataFrame,
    n: int,
    seed: int = 42,
    min_similarity: float = 0.35,
    max_similarity: float = 0.95,
    max_group_size: int = MAX_GROUP_SIZE,
) -> list[dict]:
    """Similar-but-not-duplicate prompts: half within one harm category, half among safe prompts."""
    rng = np.random.default_rng(seed)

    def candidates_for(group: pd.DataFrame) -> list[tuple[dict, dict, float]]:
        group = group.iloc[rng.permutation(len(group))[:max_group_size]]
        records = group.to_dict("records")
        mined = mine_similar_pairs(group["prompt"].tolist(), min_similarity, max_similarity)
        return [(records[first], records[second], score) for first, second, score in mined]

    harm = pool[pool["label"] == HARM]
    harm_candidates = {
        category: candidates_for(group) for category, group in harm.groupby("primary_category", sort=True)
    }
    safe_candidates = candidates_for(pool[pool["label"] == SAFE])

    selected = []
    allocation = allocate_by_category({category: len(items) for category, items in harm_candidates.items()}, n // 2)
    for category in sorted(allocation):
        items = harm_candidates[category]
        selected += [items[index] for index in rng.permutation(len(items))[: allocation[category]]]
    selected += [safe_candidates[index] for index in rng.permutation(len(safe_candidates))[: n - len(selected)]]

    pairs = [make_pair("paraphrase", _side(first), _side(second), "aegis", match_score=score) for first, second, score in selected]
    return [pairs[index] for index in rng.permutation(len(pairs))]


def build_wrapped_pairs(
    pool: pd.DataFrame,
    templates: list[str],
    n: int,
    templates_per_prompt: int = 5,
    seed: int = 42,
    max_prompt_chars: int = MAX_WRAPPED_PROMPT_CHARS,
) -> tuple[list[dict], list[dict]]:
    """Matched jailbreak_variant / benign_twin pairs.

    Each sampled unsafe prompt gets the unused safe prompt closest in length; both are
    wrapped in the same randomly chosen templates, so pair i of both lists shares a template.
    """
    rng = np.random.default_rng(seed)
    eligible = pool[pool["chars"] <= max_prompt_chars]
    harm = sample_by_category(eligible[eligible["label"] == HARM], math.ceil(n / templates_per_prompt), rng)
    safe = eligible[eligible["label"] == SAFE]
    safe = safe.iloc[rng.permutation(len(safe))].reset_index(drop=True)
    safe_lengths = safe["chars"].to_numpy()
    used = np.zeros(len(safe), dtype=bool)

    jailbreak_pairs, benign_twin_pairs = [], []
    for harm_row in harm.to_dict("records"):
        if len(jailbreak_pairs) >= n:
            break
        gaps = np.where(used, np.inf, np.abs(safe_lengths - harm_row["chars"]))
        if not np.isfinite(gaps).any():
            break
        match = int(np.argmin(gaps))
        used[match] = True
        safe_row = safe.iloc[match].to_dict()

        template_ids = rng.choice(len(templates), size=min(templates_per_prompt, len(templates)), replace=False)
        for template_id in template_ids.tolist():
            if len(jailbreak_pairs) >= n:
                break
            template = templates[template_id]
            jailbreak_pairs.append(
                make_pair(
                    "jailbreak_variant",
                    _side(harm_row),
                    _side(harm_row, wrap_with_template(template, harm_row["prompt"])),
                    "aegis+harmbench_templates",
                    template_id=template_id,
                )
            )
            benign_twin_pairs.append(
                make_pair(
                    "benign_twin",
                    _side(safe_row),
                    _side(safe_row, wrap_with_template(template, safe_row["prompt"])),
                    "aegis+harmbench_templates",
                    template_id=template_id,
                )
            )
    return jailbreak_pairs, benign_twin_pairs


def build_code_pairs(
    templates: list[str],
    code_bank: list[dict] = CODE_PAIRS,
    templates_per_text: int = 3,
    seed: int = 42,
) -> list[dict]:
    """Code-domain pairs from the hand-authored bank; the harmful side is category Malware."""
    rng = np.random.default_rng(seed)

    def side(entry: dict, kind: str, index: int, text: str | None = None) -> dict:
        harmful = kind == "malicious"
        return {
            "text": entry[kind][index] if text is None else text,
            "label": HARM if harmful else SAFE,
            "category": CODE_CATEGORY if harmful else None,
            "categories": [CODE_CATEGORY] if harmful else [],
            "source_id": f"code_bank:{entry['category']}:{kind}:{index}",
        }

    contrast, paraphrase, jailbreak, benign_twin = [], [], [], []
    for entry in code_bank:
        subcategory = entry["category"]
        for malicious in range(len(entry["malicious"])):
            for benign in range(len(entry["benign_twin"])):
                contrast.append(
                    make_pair(
                        "code_safe_harm_contrast",
                        side(entry, "malicious", malicious),
                        side(entry, "benign_twin", benign),
                        "code_bank",
                        subcategory=subcategory,
                    )
                )
        for kind in ("malicious", "benign_twin"):
            for first in range(len(entry[kind])):
                for second in range(first + 1, len(entry[kind])):
                    paraphrase.append(
                        make_pair(
                            "code_paraphrase", side(entry, kind, first), side(entry, kind, second), "code_bank", subcategory=subcategory
                        )
                    )
        # malicious[i] and benign_twin[i] are parallel phrasings — wrap them in the same templates
        for index in range(min(len(entry["malicious"]), len(entry["benign_twin"]))):
            template_ids = rng.choice(len(templates), size=min(templates_per_text, len(templates)), replace=False)
            for template_id in template_ids.tolist():
                for kind, pair_type, target in (
                    ("malicious", "code_jailbreak_variant", jailbreak),
                    ("benign_twin", "code_benign_twin", benign_twin),
                ):
                    text = entry[kind][index]
                    target.append(
                        make_pair(
                            pair_type,
                            side(entry, kind, index),
                            side(entry, kind, index, wrap_with_template(templates[template_id], text)),
                            "code_bank+harmbench_templates",
                            template_id=template_id,
                            subcategory=subcategory,
                        )
                    )

    contrast = [contrast[index] for index in rng.permutation(len(contrast))]
    paraphrase = [paraphrase[index] for index in rng.permutation(len(paraphrase))]
    return contrast + paraphrase + jailbreak + benign_twin


def _clean(value):
    """JSON-safe value: numpy scalars to Python, NaN to None, lone surrogates replaced."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        return value.encode("utf-8", "replace").decode("utf-8")
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _harm_category(record: dict) -> str:
    for prefix in ("anchor", "pair"):
        if record[f"{prefix}_label"] == HARM:
            return record[f"{prefix}_category"] or OTHER_CATEGORY
    return "(safe only)"


def write_dataset(pairs: list[dict], output_dir: Path, extra: dict | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "pairs.jsonl").open("w", encoding="utf-8") as file:
        for index, record in enumerate(pairs):
            clean = {key: _clean(value) for key, value in record.items()}
            file.write(json.dumps({"id": index, **clean}, ensure_ascii=False) + "\n")

    by_type_and_category: dict[str, Counter] = {}
    unique_texts: dict[str, str] = {}
    for record in pairs:
        by_type_and_category.setdefault(record["pair_type"], Counter())[_harm_category(record)] += 1
        for prefix in ("anchor", "pair"):
            unique_texts.setdefault(record[f"{prefix}_text"], record[f"{prefix}_label"])

    manifest = {
        "schema_version": 2,
        "total_pairs": len(pairs),
        "counts_by_type": dict(Counter(record["pair_type"] for record in pairs)),
        "counts_by_type_and_harm_category": {
            pair_type: dict(counter.most_common()) for pair_type, counter in by_type_and_category.items()
        },
        "unique_texts_by_label": dict(Counter(unique_texts.values())),
        **(extra or {}),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-safe-harm", type=int, default=500)
    parser.add_argument("--n-paraphrase", type=int, default=500)
    parser.add_argument("--n-wrapped", type=int, default=500, help="pairs of each of jailbreak_variant and benign_twin")
    parser.add_argument("--templates-per-prompt", type=int, default=5)
    parser.add_argument("--code-templates-per-text", type=int, default=3)
    parser.add_argument("--min-category-prompts", type=int, default=100)
    parser.add_argument("--max-wrapped-prompt-chars", type=int, default=MAX_WRAPPED_PROMPT_CHARS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-code", action="store_true", help="skip the separate code-domain pair set")
    args = parser.parse_args()

    train = load_aegis_split("train")
    held_out = pd.concat([load_aegis_split("validation")["prompt"], load_aegis_split("test")["prompt"]])
    exclude = frozenset(held_out.dropna().astype(str).str.strip())
    pool, pool_info = prepare_prompt_pool(train, exclude, args.min_category_prompts)
    templates = load_jailbreak_templates()

    pairs = build_safe_harm_pairs(pool, args.n_safe_harm, seed=args.seed)
    pairs += build_paraphrase_pairs(pool, args.n_paraphrase, seed=args.seed)
    jailbreak_pairs, benign_twin_pairs = build_wrapped_pairs(
        pool, templates, args.n_wrapped, args.templates_per_prompt, seed=args.seed,
        max_prompt_chars=args.max_wrapped_prompt_chars,
    )
    pairs += jailbreak_pairs + benign_twin_pairs

    template_info = {
        "template_count": len(templates),
        "templates_with_placeholder": sum(TEMPLATE_PLACEHOLDER in template for template in templates),
    }
    manifest = write_dataset(
        pairs,
        args.output_dir,
        {
            "parameters": vars(args),
            "filters": pool_info["stats"],
            "category_map": pool_info["category_map"],
            "dropped_category": NEEDS_CAUTION,
            "other_category": OTHER_CATEGORY,
            "jailbreak_pattern": JAILBREAK_PATTERN,
            **template_info,
        },
    )
    print(json.dumps({key: manifest[key] for key in ("total_pairs", "counts_by_type", "unique_texts_by_label")}, indent=2))

    if not args.skip_code:
        code_manifest = write_dataset(
            build_code_pairs(templates, templates_per_text=args.code_templates_per_text, seed=args.seed),
            args.output_dir / "code",
            {"parameters": vars(args), "code_category": CODE_CATEGORY, **template_info},
        )
        print(json.dumps({key: code_manifest[key] for key in ("total_pairs", "counts_by_type")}, indent=2))


if __name__ == "__main__":
    main()
