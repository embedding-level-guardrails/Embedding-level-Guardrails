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
            _load_pretrained: bool = False,
            pooling_layer: int = -1,
            pooling_strategy: str = "mean",
    ):
        super().__init__()
        self.pooling_layer = pooling_layer
        self.pooling_strategy = pooling_strategy
        encoder_kwargs = {}
        if pooling_strategy == "main_token":
            # SDPA / Flash Attention do not return attention weights.
            encoder_kwargs["attn_implementation"] = "eager"
        if _load_pretrained:
            config = AutoConfig.from_pretrained(model_name)
            self.encoder = AutoModel.from_config(config, **encoder_kwargs)
        else:
            self.encoder = AutoModel.from_pretrained(model_name, **encoder_kwargs)
        if pooling_strategy == "main_token":
            num_layers = self.encoder.config.num_hidden_layers
            if pooling_layer == 0 or not -num_layers <= pooling_layer <= num_layers:
                raise ValueError(
                    "main_token requires pooling_layer in "
                    f"[-{num_layers}, -1] or [1, {num_layers}]; "
                    "layer 0 is the input embedding and has no attention."
                )
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
        attention = None
        if self.pooling_strategy == "main_token":
            output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=self.pooling_layer != -1,
                return_dict=True,
            )
            hidden_state = (
                output.last_hidden_state if self.pooling_layer == -1
                else output.hidden_states[self.pooling_layer]
            )
            # hidden_states[0] is the input embedding; attentions[0] is layer 1.
            attention_index = (
                self.pooling_layer - 1 if self.pooling_layer > 0
                else self.pooling_layer
            )
            if output.attentions is None or output.attentions[attention_index] is None:
                raise RuntimeError("main_token pooling requires encoder attention weights.")
            attention = output.attentions[attention_index]
        elif self.pooling_layer == -1:
            output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden_state = output.last_hidden_state
        else:
            output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_state = output.hidden_states[self.pooling_layer]
        output = self._pool(hidden_state, attention_mask, attention)
        if self.projection_head is not None:
            output = self.projection_head(output)
        return output

    def _pool(
            self, batch: torch.Tensor, attention_mask: torch.Tensor,
            attention: torch.Tensor | None = None,
    ):
        mask = attention_mask.unsqueeze(-1).to(batch.dtype)
        if self.pooling_strategy == "main_token":
            if attention is None:
                raise ValueError("main_token pooling requires attention weights.")
            valid = attention_mask.bool()
            # Attention axes: batch, head, querying token, receiving token.
            incoming = (
                attention.float().mean(dim=1) * valid.unsqueeze(-1)
            ).sum(dim=1)
            incoming = incoming.masked_fill(~valid, float("-inf"))
            indices = incoming.argmax(dim=-1)
            selected = batch[torch.arange(batch.size(0), device=batch.device), indices]
            return selected * valid.any(dim=1, keepdim=True).to(batch.dtype)
        if self.pooling_strategy == "first":
            return (mask * batch)[:, 0, :]
        if self.pooling_strategy == "random":
            indices = torch.multinomial(attention_mask.float(), num_samples=1).squeeze(1)
            return batch[torch.arange(batch.size(0), device=batch.device), indices]

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
            _load_pretrained: bool = False
    ):
        super().__init__()
        if backbone_name is None or _load_pretrained:
            self.embedder = Embedder(
                projection_hidden_dim=projection_hidden_dim,
                projection_output_dim=projection_output_dim,
                model_name=model_name,
                _load_pretrained=_load_pretrained
            )
        else:
            self.embedder, _, _ = Embedder.load(backbone_name)
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
        model = cls(**classifier_config, _load_pretrained=True)
        model.load_state_dict(checkpoint["model"])
        return model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        embedding = self.embedder(input_ids, attention_mask)
        return self.mlp(embedding)


class LinearProbe(BaseClassifier):

    def __init__(
            self,
            model_name: str = BACKBONE,
            _load_pretrained: bool = False,
    ):
        super().__init__()
        if _load_pretrained:
            config = AutoConfig.from_pretrained(BACKBONE)
            self.backbone = AutoModel.from_config(config)
        else:
            self.backbone = AutoModel.from_pretrained(BACKBONE)
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

        model = cls(_load_pretrained=True)
        model.load_state_dict(checkpoint["model"])
        return model