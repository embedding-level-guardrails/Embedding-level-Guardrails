"""Обучаемый энкодер + опциональная классификационная голова.

Ровно тот же пулинг и тот же префикс, что в eguard.encoders.hf_encoder, — иначе
дообученный чекпоинт нельзя было бы сравнивать с frozen-базлайном из RQ1.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ..config import EncoderSpec
from ..encoders.hf_encoder import pool, resolve_device
from ..utils import get_logger

logger = get_logger(__name__)


class GuardEncoder(nn.Module):
    """Backbone + пулинг (+ линейная голова, если нужна classification-ветка)."""

    def __init__(self, spec: EncoderSpec, with_head: bool = False, dropout: float = 0.1):
        super().__init__()
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.spec = spec
        self.key = spec.key
        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
        self.backbone = AutoModel.from_pretrained(spec.hf_id)

        config = AutoConfig.from_pretrained(spec.hf_id)
        self.dim = getattr(config, "hidden_size", None) or getattr(config, "d_model")
        model_max = getattr(config, "max_position_embeddings", spec.max_length) or spec.max_length
        self.max_length = min(spec.max_length, model_max)

        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.dim, 2)) if with_head else None

    def prepare(self, texts: list[str]) -> list[str]:
        return [self.spec.prefix + t for t in texts] if self.spec.prefix else list(texts)

    def tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            self.prepare(texts), padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return pool(out.last_hidden_state, attention_mask, self.spec.pooling)

    def logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.head is None:
            raise RuntimeError("Модель собрана без классификационной головы (with_head=False)")
        return self.head(embeddings)

    @torch.no_grad()
    def encode_texts(self, texts: list[str], device: torch.device, batch_size: int = 64,
                     normalize: bool = True) -> torch.Tensor:
        """Инференс для валидации. Порядок текстов сохраняется."""
        self.eval()
        chunks = []
        for i in range(0, len(texts), batch_size):
            batch = self.tokenize(texts[i : i + batch_size]).to(device)
            emb = self(batch["input_ids"], batch["attention_mask"]).float()
            chunks.append(torch.nn.functional.normalize(emb, dim=-1) if normalize else emb)
        return torch.cat(chunks) if chunks else torch.empty(0, self.dim)

    def save(self, path: str | Path) -> Path:
        """Сохраняет так, чтобы чекпоинт читался обычным AutoModel из 01_embed.py."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        if self.head is not None:
            torch.save(self.head.state_dict(), path / "head.pt")
        logger.info("Чекпоинт сохранён: %s", path)
        return path


def build_model(spec: EncoderSpec, device: str = "auto", with_head: bool = False
                ) -> tuple[GuardEncoder, torch.device]:
    resolved = resolve_device(device)
    model = GuardEncoder(spec, with_head=with_head).to(resolved)
    return model, resolved


def load_checkpoint(path: str | Path, spec: EncoderSpec, device: str = "auto"
                    ) -> tuple[GuardEncoder, torch.device]:
    """Поднимает сохранённый чекпоинт: backbone/токенизатор с диска + голова, если есть.

    Пулинг и префикс берутся из spec, а не из чекпоинта, — они должны совпадать с
    теми, что использовались при обучении и в RQ1, иначе эмбеддинги несравнимы.
    """
    from dataclasses import replace

    path = Path(path)
    local = replace(spec, hf_id=str(path))
    head_path = path / "head.pt"
    resolved = resolve_device(device)

    model = GuardEncoder(local, with_head=head_path.exists()).to(resolved)
    if head_path.exists():
        model.head.load_state_dict(torch.load(head_path, map_location=resolved))
        logger.info("%s: голова загружена", path)
    model.eval()
    return model, resolved


def frozen_teacher(spec: EncoderSpec, device: torch.device) -> GuardEncoder:
    """Замороженная копия базовой модели — якорь для KL-регуляризатора (RQ4)."""
    teacher = GuardEncoder(spec, with_head=False).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher
