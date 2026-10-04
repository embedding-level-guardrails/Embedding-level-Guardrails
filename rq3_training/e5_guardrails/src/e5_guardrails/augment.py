"""Train-only augmentation and pair construction (RQ2), built after splitting.

What each pair type changes:

* ``jailbreak_variant`` / ``benign_twin`` add *new texts*: a train anchor wrapped
  in a HarmBench template. The wrapped text inherits the anchor's label and
  family (so it can never leak into another split). The inherited label is an
  assumption, flagged by ``label_inherited``: a wrapper can add harmful content
  of its own, so a wrapped safe request is not guaranteed to be safe.
* ``paraphrase_candidate`` adds no texts: TF-IDF-similar same-label train rows.
  They are *candidates*, not verified paraphrases.
* ``safe_harm_contrast`` adds no texts: random harmful/safe train pairs.

Pairs only act through the sampler (co-location in a batch). Binary SupCon
already treats every same-class row in a batch as a positive and every
other-class row as a negative, so pair metadata changes nothing unless it
changes the texts or the batches. Torch-free.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from e5_guardrails import jailbreak_templates as jt
from e5_guardrails import preprocess

WRAPPER_TYPES = {"jailbreak_variant": 1, "benign_twin": 0}  # pair type -> anchor label
PAIR_TYPES = (*WRAPPER_TYPES, "paraphrase_candidate", "safe_harm_contrast")


def load_templates(url: str, cache_dir: Path) -> tuple[list[tuple[str, str]], dict[str, str]]:
    cache = cache_dir / "harmbench_jailbreaks.py"
    if not cache.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            cache.write_bytes(response.read())
    return jt.usable_templates(jt.parse_harmbench_templates(cache.read_text(encoding="utf-8")))


def build_training_set(
    train: pd.DataFrame,
    cfg: dict[str, Any],
    templates: list[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return (training rows, pairs over row positions, report)."""
    aug, pcfg = cfg["augmentation"], cfg["pairs"]
    seed = cfg["preprocess"]["split_seed"]
    rng = np.random.default_rng(seed)
    unknown = set(aug["wrapper_types"]) - set(WRAPPER_TYPES) | set(pcfg["types"]) - set(PAIR_TYPES)
    if unknown:
        raise ValueError(f"Unknown pair types: {sorted(unknown)}")

    rows = train.assign(row_kind="anchor", anchor_sample_id=train["sample_id"], template_id=None,
                        label_inherited=False).reset_index(drop=True)
    pairs: list[tuple[int, int, str]] = []
    new_rows = []
    if aug["wrapper_types"]:
        if not templates:
            raise ValueError("Wrapper augmentation requested but no usable templates given")
        fill = jt.fill_template
        for pair_type in aug["wrapper_types"]:
            pool = np.flatnonzero(rows["label"].to_numpy() == WRAPPER_TYPES[pair_type])
            n = round(len(pool) * aug["anchor_fraction"])
            for anchor in sorted(rng.choice(pool, size=n, replace=False).tolist()):
                src = rows.iloc[anchor]
                k = min(aug["wrappers_per_anchor"], len(templates))
                for t in rng.choice(len(templates), size=k, replace=False):
                    tid, template = templates[int(t)]
                    new_rows.append({
                        **src.to_dict(),
                        "sample_id": f"{src['sample_id']}#{pair_type}:{tid}",
                        "text": fill(template, src["text"]),
                        "row_kind": pair_type,
                        "anchor_sample_id": src["sample_id"],
                        "template_id": tid,
                        "label_inherited": True,
                    })
                    pairs.append((anchor, len(rows) + len(new_rows) - 1, pair_type))
    if new_rows:
        rows = pd.concat([rows, pd.DataFrame(new_rows)], ignore_index=True)

    anchors = rows.index[rows["row_kind"] == "anchor"].to_numpy()
    if "paraphrase_candidate" in pcfg["types"]:
        texts = rows.loc[anchors, "text"].tolist()
        cand = preprocess.near_duplicate_pairs(texts, threshold=pcfg["paraphrase_min_sim"])
        cand = cand[cand["similarity"] <= pcfg["paraphrase_max_sim"]]
        labels = rows.loc[anchors, "label"].to_numpy()
        for i, j in zip(cand["i"].astype(int), cand["j"].astype(int)):
            if labels[i] == labels[j]:
                pairs.append((int(anchors[i]), int(anchors[j]), "paraphrase_candidate"))
    if "safe_harm_contrast" in pcfg["types"]:
        harm = anchors[rows.loc[anchors, "label"].to_numpy() == 1]
        safe = anchors[rows.loc[anchors, "label"].to_numpy() == 0]
        n = min(pcfg["n_contrast"], len(harm), len(safe))
        for i, j in zip(rng.choice(harm, n, replace=False), rng.choice(safe, n, replace=False)):
            pairs.append((int(i), int(j), "safe_harm_contrast"))

    pair_df = pd.DataFrame(pairs, columns=["i", "j", "pair_type"])
    used = pair_df[pair_df["pair_type"].isin(pcfg["types"])] if len(pair_df) else pair_df
    report = {
        "unique_anchors": len(anchors),
        "rows": len(rows),
        "rows_by_kind": rows["row_kind"].value_counts().to_dict(),
        "rows_by_label": {int(k): int(v) for k, v in rows["label"].value_counts().items()},
        "pairs_built_by_type": pair_df["pair_type"].value_counts().to_dict() if len(pair_df) else {},
        "pairs_used_for_sampling_by_type": used["pair_type"].value_counts().to_dict() if len(used) else {},
    }
    return rows, used.reset_index(drop=True), report


def partners_from_pairs(pairs: pd.DataFrame) -> dict[int, list[int]]:
    partners: dict[int, list[int]] = {}
    for i, j in zip(pairs["i"].astype(int), pairs["j"].astype(int)):
        partners.setdefault(i, []).append(j)
        partners.setdefault(j, []).append(i)
    return partners
