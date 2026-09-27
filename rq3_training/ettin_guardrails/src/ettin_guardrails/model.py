from abc import ABC, abstractmethod
from pathlib import Path
from typing import Mapping, Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedTokenizerBase, AutoTokenizer

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
            projection_hidden_dim: int | None = None,
            projection_output_dim: int | None = None,
            model_name: str = BACKBONE,
            _load_local: bool = False,
    ):
        super().__init__()
        if _load_local:
            config = AutoConfig.from_pretrained(model_name)
            self.encoder = AutoModel.from_config(config)
        else:
            self.encoder = AutoModel.from_pretrained(model_name)
        self.projection_head = None
        if projection_hidden_dim is not None and projection_output_dim is not None:
            self.projection_head = ProjectionHead(
                self.encoder.config.hidden_size,
                projection_hidden_dim,
                projection_output_dim
            )
            self.embedding_dim = projection_output_dim
        else:
            self.embedding_dim = self.encoder.config.hidden_size

    @classmethod
    def _load_model(cls, checkpoint: Mapping[str, Any]) -> "Embedder":
        model_type = checkpoint["model_name"]
        if model_type not in ("embedder", "ettin-cl"):
            raise ValueError(f"Expected a embedder checkpoint, got {model_type}")

        models_config = checkpoint["config"]["models"]
        embedder_config = dict(models_config.get("embedder", {}))
        model = cls(**embedder_config, _load_local=True)
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
        output = self._mean_pooling(output.last_hidden_state, attention_mask)
        if self.projection_head is not None:
            output = self.projection_head(output)
        return output

    def _mean_pooling(self, batch: torch.Tensor, attention_mask: torch.Tensor):
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
            head_hidden_dim: int | None,
            projection_hidden_dim: int | None = None,
            projection_output_dim: int | None = None,
            backbone_name: str | None = None,
            model_name: str = BACKBONE,
            _load_local: bool = False
    ):
        super().__init__()
        if backbone_name is not None:
            self.embedder, _, _ = Embedder.load(backbone_name)
        else:
            self.embedder = Embedder(
                projection_hidden_dim=projection_hidden_dim,
                projection_output_dim=projection_output_dim,
                model_name=model_name,
                _load_local=_load_local
            )
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
        if model_type not in ("classifier", "ettin-ce"):
            raise ValueError(f"Expected a classifier checkpoint, got {model_type}")
        models_config = checkpoint["config"]["models"]
        classifier_config = dict(models_config.get("classifier", {}))
        model = cls(**classifier_config, _load_local=True)
        model.load_state_dict(checkpoint["model"])
        return model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        embedding = self.embedder(input_ids, attention_mask)
        return self.mlp(embedding)


class LinearProbe(BaseClassifier):

    def __init__(
            self,
            model_name: str = BACKBONE,
            _load_local: bool = False,
    ):
        super().__init__()
        if _load_local:
            config = AutoConfig.from_pretrained(model_name)
            self.backbone = AutoModel.from_config(config)
        else:
            self.backbone = AutoModel.from_pretrained(model_name)
        # Buffers follow device/dtype changes and remain in checkpoints, but
        # are excluded from parameters() and autograd.
        for module in self.backbone.modules():
            for name, parameter in list(module.named_parameters(recurse=False)):
                delattr(module, name)
                module.register_buffer(name, parameter.detach())
        self.backbone.eval()
        self.mlp = nn.Linear(self.backbone.config.hidden_size, 2)

    def forward(
            self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = output.last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            embedding = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.mlp(embedding)

    def train(self, mode: bool = True) -> "LinearProbe":
        super().train(mode)
        self.backbone.eval()
        return self

    @classmethod
    def _load_model_from_checkpoint(cls, checkpoint: Mapping[str, Any]) -> "LinearProbe":
        model_type = checkpoint["model_name"]
        if model_type != "linear-probe":
            raise ValueError(f"Expected a linear-probe checkpoint, got {model_type}")

        models_config = checkpoint["config"]["models"]
        model = cls(model_name=models_config["backbone"], _load_local=True)
        model.load_state_dict(checkpoint["model"])
        return model