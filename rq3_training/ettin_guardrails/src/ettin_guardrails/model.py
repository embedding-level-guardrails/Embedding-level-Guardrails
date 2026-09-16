from typing import Literal, Any

import torch
from torch import nn
from transformers import AutoModel


class ProjectionHead(nn.Module):
    """
    Performs projection of input token representations through a multi-layer perceptron (MLP)
    after applying a mean pooling operation.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, batch: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(batch.dtype)
        pooled = (mask * batch).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.head(pooled)


class Embedder(nn.Module):

    def __init__(
            self,
            transformation_head: Literal["projection"] = "projection",
            *,
            hidden_dim: int | None = None,
            output_dim: int | None = None,
            model_name: str = "jhu-clsp/ettin-encoder-68m",
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        embedding_dim = self.encoder.config.hidden_size
        if transformation_head == "projection":
            hidden_dim = embedding_dim * 2 if hidden_dim is None else hidden_dim
            output_dim = embedding_dim if output_dim is None else output_dim
            self.transformation_head = ProjectionHead(
                input_dim=embedding_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
            )
        else:
            raise ValueError(
                f"Unknown transformation head {transformation_head!r}. "
                "Supported heads: 'projection'."
            )

    @classmethod
    def load_embedder(cls, checkpoint: dict[str, Any]):
        cfg = checkpoint["config"]
        model_config = dict(cfg["model"])
        # Preserve loading of checkpoints saved before the dimension parameters were split.
        if "head_hidden_dim" in model_config:
            model_config.setdefault("hidden_dim", model_config.pop("head_hidden_dim") or None)
        embedder = cls(**model_config)
        embedder.load_state_dict(checkpoint["model"], strict=True)
        return embedder

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.transformation_head(output.last_hidden_state, attention_mask)
        return embeddings
