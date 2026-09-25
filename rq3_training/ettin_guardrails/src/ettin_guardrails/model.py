from typing import Mapping, Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

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
            model_name: str = "jhu-clsp/ettin-encoder-68m",
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
    def load(cls, checkpoint: Mapping[str, Any]) -> "Embedder":
        model_type = checkpoint["model_name"]
        if model_type not in ("embedder", "ettin-cl"):
            raise ValueError(f"Expected a embedder checkpoint, got {model_type}")

        models_config = checkpoint["config"]["models"]
        embedder_config = dict(models_config.get("embedder", {}))
        model = cls(**embedder_config, model_name=models_config["backbone"], _load_local=True)
        model.load_state_dict(checkpoint["model"])
        return model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        output = self._mean_pooling(output.last_hidden_state, attention_mask)
        if self.projection_head is not None:
            output = self.projection_head(output)
        return output


    def _mean_pooling(self, batch: torch.Tensor, attention_mask: torch.Tensor):
        mask = attention_mask.unsqueeze(-1).to(batch.dtype)
        return (mask * batch).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


class Classifier(nn.Module):

    def __init__(
            self,
            head_hidden_dim: int,
            projection_hidden_dim: int | None = None,
            projection_output_dim: int | None = None,
            model_name: str = "jhu-clsp/ettin-encoder-68m",
            _load_local: bool = False
    ):
        super().__init__()
        self.embedder = Embedder(
            projection_hidden_dim=projection_hidden_dim,
            projection_output_dim=projection_output_dim,
            model_name=model_name,
            _load_local=_load_local
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.embedder.embedding_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, 2),
        )

    @classmethod
    def load(cls, checkpoint: Mapping[str, Any]) -> "Classifier":
        model_type = checkpoint["model_name"]
        if model_type not in ("classifier", "joint-classifier", "ettin-ce"):
            raise ValueError(f"Expected a classifier checkpoint, got {model_type}")

        models_config = checkpoint["config"]["models"]
        classifier_config = dict(models_config.get("classifier", {}))
        if model_type == "joint-classifier":
            classifier_config.update(models_config.get("embedder", {}))
        model = cls(**classifier_config, model_name=models_config["backbone"], _load_local=True)
        model.load_state_dict(checkpoint["model"])
        return model


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        embedding = self.embedder(input_ids, attention_mask)
        return self.mlp(embedding), embedding
