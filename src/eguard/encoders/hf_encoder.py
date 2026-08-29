"""Frozen HF-энкодер: токенизация -> forward -> пулинг -> np.ndarray."""
from __future__ import annotations

import time

import numpy as np
import torch

from ..config import EncoderSpec
from ..utils import get_logger
from .base import BaseEncoder, EncodeResult

logger = get_logger(__name__)


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pool(hidden: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    """hidden: [B, T, H], mask: [B, T] -> [B, H]."""
    if mode == "cls":
        return hidden[:, 0]
    mask_f = mask.unsqueeze(-1).to(hidden.dtype)
    if mode == "mean":
        return (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-9)
    if mode == "max":
        return hidden.masked_fill(mask_f == 0, torch.finfo(hidden.dtype).min).max(1).values
    raise ValueError(f"Unknown pooling: {mode}")


class HFEncoder(BaseEncoder):
    def __init__(self, spec: EncoderSpec, device: str = "auto", layer: int = -1, dtype: str = "auto"):
        from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer

        self.spec = spec
        self.key = spec.key
        self.layer = layer
        self.device = resolve_device(device)

        if dtype == "auto":
            torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        else:
            torch_dtype = getattr(torch, dtype)
        self.torch_dtype = torch_dtype

        kw = {"trust_remote_code": spec.trust_remote_code}
        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, **kw)

        if spec.head == "seqcls":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                spec.hf_id, torch_dtype=torch_dtype, **kw
            )
        else:
            self.model = AutoModel.from_pretrained(spec.hf_id, torch_dtype=torch_dtype, **kw)

        self.model.to(self.device).eval()
        for p in self.model.parameters():          # frozen: RQ1 без какого-либо дообучения
            p.requires_grad_(False)

        cfg = AutoConfig.from_pretrained(spec.hf_id, **kw)
        self.dim = getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model")
        # У некоторых моделей (Prompt Guard / mDeBERTa) окно жёстко ограничено 512.
        model_max = getattr(cfg, "max_position_embeddings", spec.max_length) or spec.max_length
        self.max_length = min(spec.max_length, model_max)
        if self.max_length < spec.max_length:
            logger.warning("%s: max_length урезан до %d (ограничение модели)", self.key, self.max_length)

        self._malicious_idx = self._find_malicious_index()

    def _find_malicious_index(self) -> int | None:
        id2label = getattr(self.model.config, "id2label", None) or {}
        for idx, name in id2label.items():
            if str(name).upper() in {"MALICIOUS", "UNSAFE", "LABEL_1", "INJECTION", "JAILBREAK"}:
                return int(idx)
        return 1 if len(id2label) == 2 else None

    def _prepare(self, texts: list[str]) -> list[str]:
        if not self.spec.prefix:
            return texts
        return [self.spec.prefix + t for t in texts]

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 32, show_progress: bool = True) -> EncodeResult:
        texts = self._prepare(texts)
        n = len(texts)

        # Батчи из текстов близкой длины: меньше паддинга, заметно быстрее на CPU.
        order = np.argsort([len(t) for t in texts])
        emb = np.zeros((n, self.dim), dtype=np.float32)
        scores = np.zeros(n, dtype=np.float32) if self.spec.head == "seqcls" else None

        t0 = time.perf_counter()
        n_batches = (n + batch_size - 1) // batch_size
        for bi in range(n_batches):
            idx = order[bi * batch_size : (bi + 1) * batch_size]
            batch = [texts[i] for i in idx]
            enc = self.tokenizer(
                batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            ).to(self.device)

            out = self.model(**enc, output_hidden_states=True)
            hidden = out.hidden_states[self.layer]
            vec = pool(hidden, enc["attention_mask"], self.spec.pooling)
            emb[idx] = vec.float().cpu().numpy()

            if scores is not None and self._malicious_idx is not None:
                probs = torch.softmax(out.logits.float(), dim=-1)[:, self._malicious_idx]
                scores[idx] = probs.cpu().numpy()

            if show_progress and (bi % 20 == 0 or bi == n_batches - 1):
                logger.info("%s: батч %d/%d", self.key, bi + 1, n_batches)

        elapsed = time.perf_counter() - t0
        meta = {
            "model": self.spec.hf_id,
            "pooling": self.spec.pooling,
            "layer": self.layer,
            "max_length": self.max_length,
            "prefix": self.spec.prefix,
            "dim": self.dim,
            "n": n,
            "device": str(self.device),
            "dtype": str(self.torch_dtype),
            "seconds": round(elapsed, 2),
            "texts_per_second": round(n / elapsed, 1) if elapsed else None,
        }
        logger.info("%s: %d текстов за %.1f c (%.1f/с)", self.key, n, elapsed, n / max(elapsed, 1e-9))
        return EncodeResult(embeddings=emb, scores=scores, meta=meta)
