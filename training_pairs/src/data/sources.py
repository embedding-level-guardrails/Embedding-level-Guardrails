"""Downloaders/loaders for the raw datasets used to build training pairs.

Sources:
- AEGIS 2.0 (nvidia/Aegis-AI-Content-Safety-Dataset-2.0): human+LLM-jury labeled
  prompt/response pairs with a binary safe/unsafe label per prompt.
- HarmBench (centerforaisafety/HarmBench): 400 canonical harmful behaviors, plus
  a bank of 114 static human-written jailbreak templates used by the
  `HumanJailbreaks` baseline method to wrap a behavior into an adversarial prompt.
- WildGuardMix (allenai/wildguardmix): optional enrichment source, gated on HF
  (requires accepting the license + an HF_TOKEN). See load_wildguardmix() docstring
  — this loader is UNVERIFIED (couldn't be tested without a token) and off by
  default; whoever has access should smoke-test the column mapping first.
"""

from __future__ import annotations

import ast
import functools
import os
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

AEGIS_TRAIN_PARQUET_URL = (
    "https://huggingface.co/api/datasets/nvidia/"
    "Aegis-AI-Content-Safety-Dataset-2.0/parquet/default/train/0.parquet"
)
HARMBENCH_BEHAVIORS_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
    "data/behavior_datasets/harmbench_behaviors_text_all.csv"
)
HARMBENCH_JAILBREAKS_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
    "baselines/human_jailbreaks/jailbreaks.py"
)
WILDGUARDMIX_TRAIN_PARQUET_URL = (
    "https://huggingface.co/api/datasets/allenai/wildguardmix/parquet/wildguardtrain/train/0.parquet"
)


def _download(url: str, dest: Path, timeout: int = 60, headers: dict | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()
        dest.write_bytes(response.content)
    return dest


@functools.lru_cache(maxsize=1)
def load_aegis(cache_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load AEGIS 2.0 train split: columns prompt, prompt_label, violated_categories."""
    path = _download(AEGIS_TRAIN_PARQUET_URL, cache_dir / "aegis_train.parquet")
    df = pd.read_parquet(path, columns=["id", "prompt", "prompt_label", "violated_categories"])
    df = df[df["prompt_label"].isin(["safe", "unsafe"])].reset_index(drop=True)
    return df


@functools.lru_cache(maxsize=1)
def load_harmbench_behaviors(cache_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load the 400 HarmBench text behaviors (canonical harmful requests)."""
    path = _download(HARMBENCH_BEHAVIORS_URL, cache_dir / "harmbench_behaviors_text_all.csv")
    return pd.read_csv(path)


@functools.lru_cache(maxsize=1)
def load_jailbreak_templates(cache_dir: Path = RAW_DIR) -> list[str]:
    """Load the 114 static human-jailbreak templates from HarmBench.

    Parsed with `ast.literal_eval` (not exec/import) since this is remote code
    we don't want to execute, only read the `JAILBREAKS = [...]` list literal from.
    """
    path = _download(HARMBENCH_JAILBREAKS_URL, cache_dir / "harmbench_jailbreaks.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "JAILBREAKS" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("JAILBREAKS list not found in jailbreaks.py")


@functools.lru_cache(maxsize=1)
def load_wildguardmix(cache_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load WildGuardMix (wildguardtrain split), normalized to AEGIS's schema.

    UNVERIFIED: allenai/wildguardmix is gated on HuggingFace (requires accepting
    the license and an approved HF_TOKEN). This sandbox has no such token, so
    the column mapping below (based on the public WildGuard paper description —
    `prompt`, `prompt_harm_label` in {"yes","no"}, `subcategory`) could not be
    tested against real data. Whoever runs this with real HF access should
    verify the mapping and fix it if the actual columns differ before relying
    on the output.

    Raises RuntimeError with setup instructions if HF_TOKEN is not set.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "WildGuardMix is gated on HuggingFace. Accept the license at "
            "https://huggingface.co/datasets/allenai/wildguardmix, create an "
            "access token at https://huggingface.co/settings/tokens, and set "
            "HF_TOKEN in the environment before calling load_wildguardmix()."
        )
    path = _download(
        WILDGUARDMIX_TRAIN_PARQUET_URL,
        cache_dir / "wildguardmix_train.parquet",
        headers={"Authorization": f"Bearer {token}"},
    )
    raw = pd.read_parquet(path)

    label_col = next((c for c in ("prompt_harm_label", "prompt_harm") if c in raw.columns), None)
    if label_col is None:
        raise RuntimeError(
            f"load_wildguardmix: expected a prompt-harm-label column, got columns "
            f"{list(raw.columns)!r} — fix the column mapping in sources.py."
        )

    def _to_binary_label(v) -> str | None:
        s = str(v).strip().lower()
        if s in {"yes", "harmful", "unsafe", "1", "true"}:
            return "unsafe"
        if s in {"no", "unharmful", "safe", "0", "false"}:
            return "safe"
        return None

    df = pd.DataFrame(
        {
            "id": raw.get("id", pd.Series(range(len(raw)))).astype(str),
            "prompt": raw["prompt"],
            "prompt_label": raw[label_col].map(_to_binary_label),
            "violated_categories": raw.get("subcategory", ""),
        }
    )
    df = df[df["prompt_label"].isin(["safe", "unsafe"])].reset_index(drop=True)
    return df
