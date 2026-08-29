"""Фабрика энкодеров."""
from __future__ import annotations

from ..config import EncoderSpec
from .base import BaseEncoder, EncodeResult


def build_encoder(spec: EncoderSpec, device: str = "auto", layer: int = -1, dtype: str = "auto") -> BaseEncoder:
    if spec.hf_id == "dummy":
        from .dummy import DummyEncoder

        return DummyEncoder(spec)

    from .hf_encoder import HFEncoder  # импорт внутри: torch нужен только здесь

    return HFEncoder(spec, device=device, layer=layer, dtype=dtype)


__all__ = ["BaseEncoder", "EncodeResult", "build_encoder"]
