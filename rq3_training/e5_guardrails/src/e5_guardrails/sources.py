"""Raw dataset loaders -> one canonical schema.

Only the request text and its own request-level label are used:

* AEGIS 2.0: ``prompt`` / ``prompt_label`` (safe, unsafe)
* WildGuardMix: ``prompt`` / ``prompt_harm_label`` (unharmful, harmful)
* ToxicChat toxicchat0124 test: ``user_input`` / ``toxicity`` (0, 1)

Responses, refusal labels and ToxicChat ``jailbreaking`` are never targets;
``jailbreaking`` / WildGuardMix ``adversarial`` are kept as metadata for subset
reporting. Rows with a missing or unknown label are dropped and counted; no
label is ever reconstructed from categories.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

SCHEMA = ["sample_id", "text", "label", "adversarial", "categories", "source", "source_split"]

LABEL_MAPS: dict[str, dict[Any, int]] = {
    "aegis": {"safe": 0, "unsafe": 1},
    "wildguardmix": {"unharmful": 0, "harmful": 1},
    "toxicchat": {0: 0, 1: 1},
}


def _require(raw: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in raw.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}; got {list(raw.columns)}")


def map_labels(values: pd.Series, mapping: dict[Any, int]) -> tuple[pd.Series, dict[str, int]]:
    """Map known label values; anything else becomes NA. Returns per-reason counts."""
    mapped = values.map(lambda v: mapping.get(v) if not pd.isna(v) else None)
    missing = int(values.isna().sum())
    unknown = int(mapped.isna().sum()) - missing
    return mapped, {"missing_label": missing, "unknown_label": unknown}


def _finish(df: pd.DataFrame, mapped: pd.Series, report: dict[str, int]) -> tuple[pd.DataFrame, dict[str, int]]:
    df = df.assign(label=mapped)
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    report = {"raw_rows": len(mapped), **report, "kept_rows": len(df)}
    return df[SCHEMA].reset_index(drop=True), report


def aegis_frame(raw: pd.DataFrame, source_split: str) -> tuple[pd.DataFrame, dict[str, int]]:
    _require(raw, ["id", "prompt", "prompt_label", "violated_categories"], "AEGIS")
    mapped, report = map_labels(raw["prompt_label"], LABEL_MAPS["aegis"])
    df = pd.DataFrame(
        {
            "sample_id": "aegis:" + raw["id"].astype(str),
            "text": raw["prompt"],
            "adversarial": None,  # not annotated in AEGIS
            "categories": raw["violated_categories"].fillna("").astype(str),
            "source": "aegis",
            "source_split": source_split,
        }
    )
    return _finish(df, mapped, report)


def wildguardmix_frame(raw: pd.DataFrame, source_split: str) -> tuple[pd.DataFrame, dict[str, int]]:
    _require(raw, ["prompt", "prompt_harm_label", "adversarial", "subcategory"], "WildGuardMix")
    mapped, report = map_labels(raw["prompt_harm_label"], LABEL_MAPS["wildguardmix"])
    # No id column upstream: row position within the pinned revision is stable.
    df = pd.DataFrame(
        {
            "sample_id": f"wgm-{source_split}:" + pd.Series(range(len(raw)), index=raw.index).astype(str),
            "text": raw["prompt"],
            "adversarial": raw["adversarial"].astype(object).where(raw["adversarial"].notna(), None),
            "categories": raw["subcategory"].fillna("").astype(str),
            "source": "wildguardmix",
            "source_split": source_split,
        }
    )
    return _finish(df, mapped, report)


def toxicchat_frame(raw: pd.DataFrame, source_split: str = "test") -> tuple[pd.DataFrame, dict[str, int]]:
    _require(raw, ["conv_id", "user_input", "toxicity", "jailbreaking"], "ToxicChat")
    mapped, report = map_labels(raw["toxicity"], LABEL_MAPS["toxicchat"])
    df = pd.DataFrame(
        {
            "sample_id": "toxicchat:" + raw["conv_id"].astype(str),
            "text": raw["user_input"],
            "adversarial": raw["jailbreaking"].map({0: False, 1: True}),  # metadata only
            "categories": "",
            "source": "toxicchat",
            "source_split": source_split,
        }
    )
    return _finish(df, mapped, report)


def _read(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".json"):
        return pd.read_json(path)
    raise ValueError(f"Unsupported file type: {path}")


FRAMERS: dict[str, Callable[[pd.DataFrame, str], tuple[pd.DataFrame, dict[str, int]]]] = {
    "aegis": aegis_frame,
    "wildguardmix": wildguardmix_frame,
    "toxicchat": toxicchat_frame,
}


def load(source: str, path: str, source_split: str) -> tuple[pd.DataFrame, dict[str, int]]:
    return FRAMERS[source](_read(path), source_split)
