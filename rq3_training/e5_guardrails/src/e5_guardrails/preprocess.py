"""Cleaning, deduplication, label-conflict checks, near-duplicate families and
overlap detection against evaluation sets. Torch-free."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def normalize_text(text: str) -> str:
    """Exact-duplicate key: case- and whitespace-insensitive."""
    return " ".join(text.split()).casefold()


def clean(df: pd.DataFrame, placeholders: Sequence[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop non-string / empty texts and placeholder stubs such as ``REDACTED``."""
    is_str = df["text"].map(lambda t: isinstance(t, str))
    stripped = df["text"].where(is_str, "").str.strip()
    empty = stripped.str.len() == 0
    stub = stripped.str.casefold().isin({p.casefold() for p in placeholders}) & ~empty
    report = {"empty_text": int(empty.sum()), "placeholder_text": int(stub.sum())}
    return df[~empty & ~stub].reset_index(drop=True), report


def dedup(df: pd.DataFrame, priority: Sequence[str] = ()) -> tuple[pd.DataFrame, dict[str, int]]:
    """One row per normalized text; texts seen with conflicting labels are dropped.

    Covers rows that repeat one prompt with different responses. When copies sit
    in several official splits, the copy from the earliest split in ``priority``
    is kept (e.g. keep the official validation copy over the train copy).
    """
    key = df["text"].map(normalize_text)
    n_labels = df.groupby(key)["label"].transform("nunique")
    conflict = n_labels > 1
    rank = df["source_split"].map({s: i for i, s in enumerate(priority)}).fillna(len(priority))
    kept = (
        df[~conflict]
        .assign(_key=key[~conflict], _rank=rank[~conflict], _pos=np.arange(len(df))[~conflict])
        .sort_values(["_rank", "_pos"], kind="stable")
        .drop_duplicates("_key")
        .sort_values("_pos")
        .drop(columns=["_key", "_rank", "_pos"])
        .reset_index(drop=True)
    )
    report = {
        "conflicting_label_rows": int(conflict.sum()),
        "conflicting_label_texts": int(key[conflict].nunique()),
        "duplicate_rows_removed": int((~conflict).sum() - len(kept)),
    }
    return kept, report


def _tfidf(texts_a: Sequence[str], texts_b: Sequence[str] | None):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True, dtype=np.float32)
    vectorizer.fit(list(texts_a) + list(texts_b or []))
    a = vectorizer.transform(texts_a)
    return a, (a if texts_b is None else vectorizer.transform(texts_b))


def near_duplicate_pairs(
    texts_a: Sequence[str],
    texts_b: Sequence[str] | None = None,
    threshold: float = 0.9,
    chunk_size: int = 512,
) -> pd.DataFrame:
    """Pairs (i, j, similarity) with TF-IDF cosine >= threshold.

    Within one collection (``texts_b is None``) only ``i < j`` pairs are returned.
    """
    columns = ["i", "j", "similarity"]
    if len(texts_a) == 0 or (texts_b is not None and len(texts_b) == 0):
        return pd.DataFrame(columns=columns)
    try:
        a, b = _tfidf(texts_a, texts_b)
    except ValueError:  # empty vocabulary (e.g. only stop words)
        return pd.DataFrame(columns=columns)
    bt = b.T.tocsc()
    rows = []
    for start in range(0, a.shape[0], chunk_size):
        sims = (a[start:start + chunk_size] @ bt).tocoo()
        keep = sims.data >= threshold
        i, j, s = sims.row[keep] + start, sims.col[keep], sims.data[keep]
        if texts_b is None:
            upper = i < j
            i, j, s = i[upper], j[upper], s[upper]
        rows.append(pd.DataFrame({"i": i, "j": j, "similarity": s}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)


def assign_families(df: pd.DataFrame, threshold: float, enabled: bool = True) -> tuple[pd.DataFrame, dict[str, int]]:
    """``family_id`` = connected component of near-duplicate links (union-find).

    A family is the unit of splitting: near-duplicates never straddle splits.
    """
    parent = np.arange(len(df))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    n_links = 0
    if enabled and len(df) > 1:
        pairs = near_duplicate_pairs(df["text"].tolist(), threshold=threshold)
        n_links = len(pairs)
        for i, j in zip(pairs["i"].astype(int), pairs["j"].astype(int)):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(len(df))], dtype=int)
    df = df.assign(family_id=df["sample_id"].to_numpy()[roots] if len(df) else [])
    sizes = df["family_id"].value_counts()
    mixed = df.groupby("family_id")["label"].nunique()
    report = {
        "near_duplicate_links": int(n_links),
        "families": len(sizes),
        "multi_member_families": int((sizes > 1).sum()),
        "rows_in_multi_member_families": int(sizes[sizes > 1].sum()),
        "families_with_mixed_labels": int((mixed > 1).sum()),
    }
    return df, report


def eval_overlap(
    source: pd.DataFrame,
    eval_sets: dict[str, pd.DataFrame],
    threshold: float,
    near_enabled: bool = True,
) -> pd.DataFrame:
    """Exact (normalized) and near-duplicate matches between source rows and evaluation sets."""
    columns = ["source_sample_id", "source_family_id", "eval_set", "eval_sample_id", "kind", "similarity"]
    rows = []
    source_keys = source["text"].map(normalize_text)
    for name, ev in eval_sets.items():
        ev_ids_by_key = ev.assign(_key=ev["text"].map(normalize_text)).groupby("_key")["sample_id"].first()
        exact = source_keys.isin(ev_ids_by_key.index)
        for idx in np.flatnonzero(exact.to_numpy()):
            rows.append((source["sample_id"].iat[idx], source["family_id"].iat[idx], name,
                         ev_ids_by_key[source_keys.iat[idx]], "exact", 1.0))
        if near_enabled:
            pairs = near_duplicate_pairs(source["text"].tolist(), ev["text"].tolist(), threshold=threshold)
            for i, j, s in zip(pairs["i"].astype(int), pairs["j"].astype(int), pairs["similarity"]):
                if exact.iat[i]:
                    continue
                rows.append((source["sample_id"].iat[i], source["family_id"].iat[i], name,
                             ev["sample_id"].iat[j], "near", float(s)))
    return pd.DataFrame(rows, columns=columns)
