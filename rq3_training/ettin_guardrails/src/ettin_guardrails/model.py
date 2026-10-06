from abc import ABC, abstractmethod
from pathlib import Path
from typing import Mapping, Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, PreTrainedTokenizerBase, AutoTokenizer

BACKBONE = "jhu-clsp/ettin-encoder-68m"


class ProjectionHead(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.mlp(batch)


class Embedder(nn.Module):

    def __init__(
            self,
            pooling_source: str = "hidden_state",
            _load_pretrained: bool = False,
    ):
        super().__init__()
        if pooling_source not in ("hidden_state", "mlm_output"):
            raise ValueError("pooling_source must be 'hidden_state' or 'mlm_output'")
        self.pooling_source = pooling_source
        self.mlm_head = None
        if pooling_source == "mlm_output":
            config = AutoConfig.from_pretrained(BACKBONE)
            if config.model_type != "modernbert":
                raise ValueError("mlm_output pooling requires a ModernBERT encoder")
            if _load_pretrained:
                mlm = AutoModelForMaskedLM.from_config(config)
            else:
                mlm = AutoModelForMaskedLM.from_pretrained(BACKBONE, config=config)
            self.encoder = mlm.base_model
            self.mlm_head = mlm.head
        elif _load_pretrained:
            config = AutoConfig.from_pretrained(BACKBONE)
            self.encoder = AutoModel.from_config(config)
        else:
            self.encoder = AutoModel.from_pretrained(BACKBONE)
        self.embedding_dim = self.encoder.config.hidden_size

    @classmethod
    def _load_model(cls, checkpoint: Mapping[str, Any]) -> "Embedder":
        model_type = checkpoint["model_name"]
        if model_type not in ("embedder",):
            raise ValueError(f"Expected a embedder checkpoint, got {model_type}")

        models_config = checkpoint["config"]["models"]
        embedder_config = dict(models_config.get("embedder", {}))
        model = cls(**embedder_config, _load_pretrained=True)
        model.load_state_dict(checkpoint["model"])
        return model

    @classmethod
    def load(cls, checkpoint_path: str | Path) -> tuple["Embedder", PreTrainedTokenizerBase, DictConfig]:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )
        config = OmegaConf.create(checkpoint["config"])
        embedder = cls._load_model(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
        return embedder, tokenizer, config

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_vectors = output.last_hidden_state
        if self.mlm_head is not None:
            token_vectors = self.mlm_head(token_vectors)
        return self._pool(token_vectors, attention_mask)

    def _pool(self, batch: torch.Tensor, attention_mask: torch.Tensor):
        mask = attention_mask.unsqueeze(-1).to(batch.dtype)
        return (mask * batch).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


class BaseClassifier(nn.Module, ABC):

    @abstractmethod
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _load_model_from_checkpoint(cls, checkpoint: Mapping[str, Any]) -> "BaseClassifier":
        raise NotImplementedError

    @classmethod
    def load(cls, checkpoint_path: str | Path) -> tuple["BaseClassifier", PreTrainedTokenizerBase, DictConfig]:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )
        config = OmegaConf.create(checkpoint["config"])
        classifier = cls._load_model_from_checkpoint(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
        return classifier, tokenizer, config


class Classifier(BaseClassifier):

    def __init__(
            self,
            pooling_source: str = "hidden_state",
            freeze_embedder: bool = False,
            head_hidden_dim: int | None = None,
            backbone_name: str | None = None,
            _load_pretrained: bool = False,
    ):
        super().__init__()
        if backbone_name is None or _load_pretrained:
            self.embedder = Embedder(
                pooling_source=pooling_source,
                _load_pretrained=_load_pretrained
            )
        else:
            self.embedder, _, _ = Embedder.load(backbone_name)
        self.freeze_embedder = freeze_embedder
        if self.freeze_embedder:
            self.embedder.requires_grad_(False)
            self.embedder.eval()
        if head_hidden_dim is None:
            self.mlp = nn.Linear(self.embedder.embedding_dim, 2)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(self.embedder.embedding_dim, head_hidden_dim),
                nn.ReLU(),
                nn.Linear(head_hidden_dim, 2),
            )

    @classmethod
    def _load_model_from_checkpoint(cls, checkpoint: Mapping[str, Any]) -> "Classifier":
        model_type = checkpoint["model_name"]
        if model_type not in ("classifier",):
            raise ValueError(f"Expected a classifier checkpoint, got {model_type}")
        models_config = checkpoint["config"]["models"]
        classifier_config = dict(models_config.get("classifier", {}))
        model = cls(**classifier_config, _load_pretrained=True)
        model.load_state_dict(checkpoint["model"])
        return model

    def train(self, mode: bool = True) -> "Classifier":
        super().train(mode)
        if self.freeze_embedder:
            self.embedder.eval()
        return self

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        if self.freeze_embedder:
            with torch.no_grad():
                embedding = self.embedder(input_ids, attention_mask)
        else:
            embedding = self.embedder(input_ids, attention_mask)
        return self.mlp(embedding)
