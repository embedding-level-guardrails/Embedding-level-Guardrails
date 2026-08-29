"""Общий интерфейс энкодера.

Любая модель (HF-энкодер, guardrail-классификатор, а позже — дообученный
контрастивно чекпоинт из RQ3) должна отдавать одно и то же:
матрицу эмбеддингов [n, dim] и, опционально, скор своей "родной" головы.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class EncodeResult:
    embeddings: np.ndarray            # [n, dim], float32, БЕЗ нормировки
    scores: np.ndarray | None = None  # [n] вероятность «плохого» класса, если у модели есть голова
    meta: dict | None = None


class BaseEncoder(ABC):
    key: str
    dim: int

    @abstractmethod
    def encode(self, texts: list[str], batch_size: int = 32, show_progress: bool = True) -> EncodeResult:
        ...
