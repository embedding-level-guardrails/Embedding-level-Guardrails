from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from ettin_guardrails.model import Embedder


def load_embedder(
    checkpoint_path: str | Path, *, weights_only: bool = True,
) -> tuple[Embedder, AutoTokenizer, DictConfig]:
    """Load an embedder; disable weights_only only for trusted legacy checkpoints."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=weights_only,
    )
    config = OmegaConf.create(checkpoint["config"])
    embedder = Embedder.load_embedder(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    return embedder, tokenizer, config


def save_checkpoint(training_checkpoint: dict[str, Any], output_path: str | Path):
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(training_checkpoint, output_dir / "last.pt")
