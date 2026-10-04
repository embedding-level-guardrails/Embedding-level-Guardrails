from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PACKAGE_ROOT / "conf" / "default.yaml"
MODES = ("ce", "supcon", "ce_supcon")
ENCODERS = ("base", *MODES)  # "base" = E5 without fine-tuning


def apply_override(cfg: dict[str, Any], key: str, value: Any) -> None:
    *parents, leaf = key.split(".")
    node = cfg
    for part in parents:
        if not isinstance(node.get(part), dict):
            raise KeyError(f"Unknown config section {part!r} in {key!r}")
        node = node[part]
    if leaf not in node:
        raise KeyError(f"Unknown config key {key!r}")
    node[leaf] = value


def parse_override(override: str) -> tuple[str, Any]:
    """``dotted.key=value`` with the value parsed as YAML."""
    key, sep, raw_value = override.partition("=")
    if not sep or not key:
        raise ValueError(f"Override must look like key=value, got {override!r}")
    return key, yaml.safe_load(raw_value)


def load_config(path: str | Path = DEFAULT_CONFIG, overrides: list[str] | None = None) -> dict[str, Any]:
    with open(path) as f:
        cfg = copy.deepcopy(yaml.safe_load(f))
    for override in overrides or []:
        apply_override(cfg, *parse_override(override))
    if cfg["protocol"] not in cfg["protocols"]:
        raise KeyError(f"Unknown protocol {cfg['protocol']!r}; known: {sorted(cfg['protocols'])}")
    return cfg


def protocol_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg["protocols"][cfg["protocol"]]


@dataclass(frozen=True)
class Paths:
    root: Path
    variant: str

    @property
    def splits(self) -> Path:
        return self.root / "splits"

    def run(self, encoder: str, seed: int) -> Path:
        return self.root / "runs" / self.variant / encoder / f"seed={seed}"

    def checkpoint(self, encoder: str, seed: int) -> Path:
        return self.run(encoder, seed) / "checkpoint"

    def predictions(self, encoder: str, seed: int) -> Path:
        return self.run(encoder, seed) / "predictions.parquet"

    def training_set(self, spec_hash: str) -> Path:
        return self.root / "training_sets" / spec_hash

    @property
    def results(self) -> Path:
        return self.root / "results" / self.variant

    @property
    def sweeps(self) -> Path:
        return self.root / "sweeps" / self.variant

    @property
    def cache(self) -> Path:
        return self.root.parent / "cache"


def paths(cfg: dict[str, Any]) -> Paths:
    root = Path(cfg["output_dir"])
    if not root.is_absolute():
        root = PACKAGE_ROOT / root
    return Paths(root / cfg["protocol"], cfg["variant"])
