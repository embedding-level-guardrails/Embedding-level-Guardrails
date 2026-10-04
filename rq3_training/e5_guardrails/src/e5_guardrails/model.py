"""E5 backbone with separate interfaces.

* ``encode(x) -> h``: E5 sentence embedding: mean pooling over non-padding
  tokens of the last hidden state, then L2 normalization (as in the model card).
  ``h`` is what the linear probe sees.
* ``project(h) -> z``: MLP projection head, L2-normalized; used only by SupCon.
* ``classify(h) -> logits``: linear safe/harm head; used only by CE.

In ``ce_supcon`` both heads read the same ``h`` from one encoder pass.
Model id, revision, tokenizer, prefix and max length all come from one config
section, so nothing silently falls back to another backbone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

USES_CE = ("ce", "ce_supcon")
USES_SUPCON = ("supcon", "ce_supcon")


def load_tokenizer(name_or_path: str | Path, revision: str | None = None) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(str(name_or_path), revision=revision)


def tokenize(tokenizer: PreTrainedTokenizerBase, texts: list[str], prefix: str, max_length: int):
    return tokenizer([prefix + t for t in texts], padding=True, truncation=True,
                     max_length=max_length, return_tensors="pt")


def token_lengths(tokenizer: PreTrainedTokenizerBase, texts: list[str], prefix: str) -> np.ndarray:
    """Untruncated token counts including special tokens."""
    encoded = tokenizer([prefix + t for t in texts], add_special_tokens=True, truncation=False)
    return np.array([len(ids) for ids in encoded["input_ids"]])


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


class E5Encoder(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    @classmethod
    def from_pretrained(cls, name_or_path: str | Path, revision: str | None = None) -> E5Encoder:
        return cls(AutoModel.from_pretrained(str(name_or_path), revision=revision))

    @property
    def hidden_size(self) -> int:
        return self.backbone.config.hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return F.normalize(mean_pool(hidden.float(), attention_mask), dim=-1)


class GuardrailModel(nn.Module):
    def __init__(self, encoder: E5Encoder, mode: str, projection_dim: int) -> None:
        super().__init__()
        if mode not in USES_CE + USES_SUPCON:
            raise ValueError(f"Unknown mode {mode!r}")
        self.mode = mode
        self.encoder = encoder
        d = encoder.hidden_size
        self.classifier = nn.Linear(d, 2) if mode in USES_CE else None
        self.projector = (
            nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, projection_dim))
            if mode in USES_SUPCON else None
        )

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_ids, attention_mask)

    def project(self, h: torch.Tensor) -> torch.Tensor:
        if self.projector is None:
            raise RuntimeError(f"mode {self.mode!r} has no projection head")
        return F.normalize(self.projector(h.float()), dim=-1)

    def classify(self, h: torch.Tensor) -> torch.Tensor:
        if self.classifier is None:
            raise RuntimeError(f"mode {self.mode!r} has no classification head")
        return self.classifier(h.float())

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encode(input_ids, attention_mask)
        out = {"h": h}
        if self.projector is not None:
            out["z"] = self.project(h)
        if self.classifier is not None:
            out["logits"] = self.classify(h)
        return out


def save_checkpoint(
    model: GuardrailModel, tokenizer: PreTrainedTokenizerBase, cfg: dict[str, Any], out_dir: Path, meta: dict[str, Any]
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.backbone.save_pretrained(out_dir / "encoder")
    tokenizer.save_pretrained(out_dir / "encoder")
    heads = {
        "classifier": model.classifier.state_dict() if model.classifier is not None else None,
        "projector": model.projector.state_dict() if model.projector is not None else None,
    }
    torch.save(heads, out_dir / "heads.pt")
    (out_dir / "checkpoint.json").write_text(json.dumps(
        {"mode": model.mode, "config": cfg, "meta": meta}, indent=2, ensure_ascii=False, default=str))


def load_checkpoint(path: Path) -> tuple[GuardrailModel, PreTrainedTokenizerBase, dict[str, Any], dict[str, Any]]:
    info = json.loads((path / "checkpoint.json").read_text())
    cfg = info["config"]
    encoder = E5Encoder.from_pretrained(path / "encoder")
    model = GuardrailModel(encoder, info["mode"], cfg["model"]["projection_dim"])
    heads = torch.load(path / "heads.pt", map_location="cpu")
    for name in ("classifier", "projector"):
        module = getattr(model, name)
        if (module is None) != (heads[name] is None):
            raise ValueError(f"Checkpoint head {name!r} does not match mode {info['mode']!r}")
        if module is not None:
            module.load_state_dict(heads[name])
    return model, load_tokenizer(path / "encoder"), cfg, info["meta"]


def load_frozen_encoder(
    encoder: str, cfg: dict[str, Any], checkpoint_dir: Path | None
) -> tuple[E5Encoder, GuardrailModel | None, PreTrainedTokenizerBase]:
    """``base`` -> pretrained E5 at the configured revision; otherwise the saved
    checkpoint. The encoder is put in eval mode with gradients disabled."""
    if encoder == "base":
        enc = E5Encoder.from_pretrained(cfg["model"]["name"], cfg["model"]["revision"])
        model, tokenizer = None, load_tokenizer(cfg["model"]["name"], cfg["model"]["revision"])
    else:
        model, tokenizer, ckpt_cfg, _ = load_checkpoint(checkpoint_dir)
        for key in ("name", "revision", "prefix", "max_length"):
            if ckpt_cfg["model"][key] != cfg["model"][key]:
                raise ValueError(f"Checkpoint model.{key}={ckpt_cfg['model'][key]!r} != config {cfg['model'][key]!r}")
        model.eval().requires_grad_(False)
        enc = model.encoder
    enc.eval().requires_grad_(False)
    return enc, model, tokenizer
