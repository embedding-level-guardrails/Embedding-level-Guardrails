"""Fixed, family-grouped splits per protocol and their manifest. Torch-free.

Source splits: ``train`` (encoder + probe fitting), ``validation`` (checkpoint,
hyperparameter and probe-C selection), ``calibration`` (operating threshold
only). Test sets: ``holdout`` (internal), protocol-specific extra sets, and
``toxicchat`` (OOD, shared by both protocols).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from e5_guardrails import preprocess, sources
from e5_guardrails.config import protocol_cfg

SOURCE_SPLITS = ("train", "validation", "calibration")
OOD_SET = "toxicchat"
COLUMNS = [*sources.SCHEMA, "family_id", "split"]


def _strata(df: pd.DataFrame) -> pd.Series:
    adv = df["adversarial"].map(lambda v: "na" if v is None or pd.isna(v) else str(bool(v)))
    return df["label"].astype(str) + "|" + adv


def grouped_stratified_split(
    df: pd.DataFrame, fractions: dict[str, float], seed: int, rest: str = "train"
) -> dict[str, pd.DataFrame]:
    """Assign whole families to splits, stratified by (label, adversarial) of
    the family's first member. Deterministic for a seed regardless of row order."""
    if fractions and (sum(fractions.values()) >= 1.0 or min(fractions.values()) <= 0):
        raise ValueError("Fractions must be positive and sum to less than 1")
    first = df.sort_values("sample_id", kind="stable").drop_duplicates("family_id")
    families = pd.DataFrame({"family_id": first["family_id"], "stratum": _strata(first)})
    families = families.sort_values("family_id", kind="stable").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    assignment = pd.Series(rest, index=families["family_id"], dtype=object)
    for _, group in families.groupby("stratum", sort=True):
        ids = group["family_id"].to_numpy()[rng.permutation(len(group))]
        start = 0
        for name, fraction in fractions.items():
            n = round(len(ids) * fraction)
            assignment[ids[start:start + n]] = name
            start += n
    split_of_row = df["family_id"].map(assignment)
    return {name: df[split_of_row == name].reset_index(drop=True) for name in (rest, *fractions)}


def cap_rows(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    """Stratified (by label) cap, used only for smoke tests."""
    if n is None or len(df) <= n:
        return df
    frac = n / len(df)
    parts = [g.sample(n=max(1, round(len(g) * frac)), random_state=seed) for _, g in df.groupby("label", sort=True)]
    return pd.concat(parts).sort_index().reset_index(drop=True)


def _prepare(df: pd.DataFrame, report: dict[str, Any], placeholders, priority=()) -> pd.DataFrame:
    df, r = preprocess.clean(df, placeholders)
    report.update(r)
    df, r = preprocess.dedup(df, priority)
    report.update(r)
    return df


def build_splits(cfg: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, Any], pd.DataFrame]:
    pcfg = protocol_cfg(cfg)
    pre = cfg["preprocess"]
    near = pre["near_duplicate"]
    reports: dict[str, Any] = {}

    def load(source: str, path: str, name: str, priority=()) -> pd.DataFrame:
        df, report = sources.load(source, path, name)
        reports[name] = report
        return _prepare(df, report, pre["placeholders"], priority)

    files = pcfg["files"]
    pool_names = [n for n in ("validation", "train") if n in files]
    pool_raw = []
    for name in pool_names:
        df, report = sources.load(pcfg["source"], files[name], name)
        reports[name] = report
        pool_raw.append(df)
    pool_report: dict[str, Any] = {}
    pool = _prepare(pd.concat(pool_raw, ignore_index=True), pool_report, pre["placeholders"], priority=pool_names)
    reports["source_pool"] = pool_report
    pool, fam_report = preprocess.assign_families(pool, near["threshold"], near["enabled"])
    reports["source_pool"]["families"] = fam_report

    tests: dict[str, pd.DataFrame] = {}
    if "holdout" in files:
        tests["holdout"] = load(pcfg["source"], files["holdout"], "holdout")
    for name, path in pcfg["extra_eval"].items():
        tests[name] = load(pcfg["source"], path, name)
    tests[OOD_SET] = load("toxicchat", cfg["ood"]["toxicchat"], OOD_SET)
    for df in tests.values():
        df["family_id"] = df["sample_id"]

    overlap = preprocess.eval_overlap(pool, tests, near["threshold"], near["enabled"])
    flagged = set(overlap["source_family_id"])
    action = pre["eval_overlap_action"]
    if action not in ("drop", "report"):
        raise ValueError("preprocess.eval_overlap_action must be drop or report")
    overlap["action"] = "dropped_source_family" if action == "drop" else "reported_only"
    if action == "drop":
        pool = pool[~pool["family_id"].isin(flagged)].reset_index(drop=True)

    # Official validation: a family with any member from it goes there whole.
    in_val = pool.groupby("family_id")["source_split"].transform(lambda s: (s == "validation").any())
    official_val = pool[in_val].reset_index(drop=True)
    carved = grouped_stratified_split(pool[~in_val].reset_index(drop=True), pcfg["carve"], seed=pre["split_seed"])
    splits = {"train": carved.pop("train")}
    if "validation" in files:
        splits["validation"] = official_val
    splits.update(carved)
    for name, df in tests.items():
        if name in splits:
            raise ValueError(f"Split {name!r} is both carved and loaded")
        splits[name] = df
    if missing := [n for n in (*SOURCE_SPLITS, "holdout", OOD_SET) if n not in splits]:
        raise ValueError(f"Protocol {cfg['protocol']!r} does not define splits {missing}")

    cap = pre["max_rows_per_split"]
    splits = {
        name: cap_rows(df, cap, pre["split_seed"]).assign(split=name)[COLUMNS]
        for name, df in splits.items()
    }
    manifest = {
        "protocol": cfg["protocol"],
        "protocol_config": pcfg,
        "ood": cfg["ood"],
        "preprocess": pre,
        "model": {k: cfg["model"][k] for k in ("name", "revision", "prefix", "max_length")},
        "source_splits": list(SOURCE_SPLITS),
        "test_sets": [n for n in splits if n not in SOURCE_SPLITS],
        "loading_and_cleaning": reports,
        "eval_overlap": {
            "action": action,
            "matches_by_set_and_kind": overlap.groupby(["eval_set", "kind"]).size().to_dict() if len(overlap) else {},
            "source_families_flagged": len(flagged),
        },
        "counts": describe(splits),
        "ids": {name: df["sample_id"].tolist() for name, df in splits.items()},
    }
    manifest["eval_overlap"]["matches_by_set_and_kind"] = {
        f"{k[0]}/{k[1]}": int(v) for k, v in manifest["eval_overlap"]["matches_by_set_and_kind"].items()
    }
    manifest["test_set_exact_overlap"] = test_set_overlap(splits)
    check_disjoint(splits, texts_vs_tests=action == "drop")
    return splits, manifest, overlap


def describe(splits: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    out = {}
    for name, df in splits.items():
        adv = df["adversarial"].dropna()
        out[name] = {
            "n": len(df),
            "safe": int((df["label"] == 0).sum()),
            "harmful": int((df["label"] == 1).sum()),
            "families": int(df["family_id"].nunique()),
            "adversarial": int(adv.astype(bool).sum()) if len(adv) else None,
        }
    return out


def check_disjoint(splits: dict[str, pd.DataFrame], texts_vs_tests: bool = True) -> None:
    """No sample, family or normalized text may be shared by two source splits,
    or (when ``texts_vs_tests``) by a source split and a test set. Overlap
    between two test sets is only reported (see ``test_set_overlap``)."""
    keys = {
        name: {
            "sample_id": set(df["sample_id"]),
            "family_id": set(df["family_id"]),
            "text": set(df["text"].map(preprocess.normalize_text)),
        }
        for name, df in splits.items()
    }
    names = list(splits)
    for a_pos, a in enumerate(names):
        for b in names[a_pos + 1:]:
            if a not in SOURCE_SPLITS and b not in SOURCE_SPLITS:
                continue
            both_source = a in SOURCE_SPLITS and b in SOURCE_SPLITS
            for kind in ("sample_id", "family_id", "text"):
                if kind == "text" and not (both_source or texts_vs_tests):
                    continue
                shared = keys[a][kind] & keys[b][kind]
                if shared:
                    raise AssertionError(f"{len(shared)} {kind} values shared by {a!r} and {b!r}, e.g. {next(iter(shared))!r}")


def test_set_overlap(splits: dict[str, pd.DataFrame]) -> dict[str, int]:
    tests = [n for n in splits if n not in SOURCE_SPLITS]
    keys = {n: set(splits[n]["text"].map(preprocess.normalize_text)) for n in tests}
    return {f"{a}/{b}": len(keys[a] & keys[b]) for i, a in enumerate(tests) for b in tests[i + 1:]}


def save_splits(splits: dict[str, pd.DataFrame], manifest: dict[str, Any], overlap: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
    overlap.to_csv(out_dir / "eval_overlap_report.csv", index=False)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))


def load_manifest(splits_dir: Path) -> dict[str, Any]:
    path = splits_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run prepare-data first")
    return json.loads(path.read_text())


def load_split(splits_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_parquet(splits_dir / f"{name}.parquet")
