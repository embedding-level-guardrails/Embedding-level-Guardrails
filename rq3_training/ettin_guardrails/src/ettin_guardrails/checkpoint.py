from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ettin_guardrails.model import Embedder, Classifier


def save_checkpoint(training_checkpoint: dict[str, Any], output_path: str | Path):
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    torch.save(training_checkpoint, output_dir / f"checkpoint_{timestamp}.pt")


def load_classifier(checkpoint_path: str | Path) -> tuple[Classifier, PreTrainedTokenizerBase, DictConfig]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )
    config = OmegaConf.create(checkpoint["config"])
    classifier = Classifier.load(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(config.models.backbone)
    return classifier, tokenizer, config


def load_embedder(checkpoint_path: str | Path) -> tuple[Embedder, AutoTokenizer, DictConfig]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )
    config = OmegaConf.create(checkpoint["config"])
    embedder = Embedder.load(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(config.models.backbone)
    return embedder, tokenizer, config
