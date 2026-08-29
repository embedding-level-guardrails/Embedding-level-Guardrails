"""Загрузка конфигурации эксперимента."""
from __future__ import annotations

from dataclasses import dataclass, field 
from pathlib import Path
from typing import Any

import yaml 


@dataclass
class EncoderSpec:
    key: str
    hf_id: str
    pooling: str = "mean"
    max_length: int = 512
    prefix: str = ""
    head: str | None = None 
    gated: bool = False 
    trust_remote_code: bool = False 


@dataclass
class DatasetSpec:
    name: str = "aegis"
    hf_id: str = "nvidia/Aegis-AI-Content-Safety-Dataset-1.0"
    text_types: list[str] = field(default_factory=lambda: ["user_message"])
    caution_policy: str = "exclude"
    min_chars: int = 3 
    max_chars: int = 8000
    val_fraction: float = 0.1


@dataclass
class Paths:
    processed: Paths = Path("data/processed")
    embeddings: Path = Path("artifacts/embeddings")
    results: Path = Path("results")
    figures: Path = Path("results/figures")

    def __post_init__(self) -> None:
        for f in ("processed", "embeddings", "results", "figures"):
            setattr(self, f, Path(getattr(self, f)))


@dataclass
class Config:
    seed: int = 42
    paths: Paths = field(default_factory=Path)
    dataset: DatasetSpec = field(default_factory=DatasetSpec)
    encoders: list[EncoderSpec] = field(default_factory=list)
    embed: dict[str, Any] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)
    viz: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def encoder(self, key: str) -> EncoderSpec:
        for spec in self.encoders:
            if spec.key == key:
                return spec 
        known = ", ".join(s.key for s in self.encoders)
        raise KeyError(f"Encoder '{key}' not found in config. Available: {known}")

    def select_encoders(self, keys: list[str] | None) -> list[EncoderSpec]:
        if not keys:
            return list(self.encoders)
        return [self.encoder(k) for k in keys]


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) 

    return Config(
        seed=raw.get("seed", 42),
        paths=Paths(**raw.get("paths", {})),
        dataset=DatasetSpec(**raw.get("dataset", {})),
        encoders=[EncoderSpec(**e) for e in raw.get("encoders", [])],
        embed=raw.get("emged", {}),
        analysis=raw.get("analysis", {}),
        viz=raw.get("viz", {}),
        raw=raw,
    )
