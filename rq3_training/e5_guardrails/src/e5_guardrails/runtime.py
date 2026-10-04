from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from e5_guardrails.model import E5Encoder, tokenize


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    return torch.device(name)


def resolve_amp_dtype(precision: str, device: torch.device) -> torch.dtype:
    if precision == "auto":
        precision = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    dtypes = {"bf16": torch.bfloat16, "fp32": torch.float32}
    if precision not in dtypes:
        raise ValueError("training.precision must be auto, bf16 or fp32")
    return dtypes[precision]


def autocast(device: torch.device, dtype: torch.dtype):
    return torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32)


@torch.inference_mode()
def embed(
    encoder: E5Encoder,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    model_cfg: dict[str, Any],
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype = torch.float32,
) -> np.ndarray:
    """``h`` for every text, in input order. Leaves the encoder's train/eval mode as it was."""
    was_training = encoder.training
    encoder.eval()
    out = np.empty((len(texts), encoder.hidden_size), dtype=np.float32)
    order = np.argsort([-len(t) for t in texts], kind="stable")  # length-sorted batches
    for start in range(0, len(texts), batch_size):
        idx = order[start:start + batch_size]
        batch = tokenize(tokenizer, [texts[i] for i in idx], model_cfg["prefix"], model_cfg["max_length"]).to(device)
        with autocast(device, amp_dtype):
            h = encoder(batch["input_ids"], batch["attention_mask"])
        out[idx] = h.float().cpu().numpy()
    encoder.train(was_training)
    return out
