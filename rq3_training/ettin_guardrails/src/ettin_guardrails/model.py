import torch
from torch import nn
from transformers import AutoModel

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
            hidden_dim: int | None,
            output_dim: int | None,
            model_name: str = "jhu-clsp/ettin-encoder-68m",
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.projection_head = None
        if hidden_dim is not None and output_dim is not None:
            self.projection_head = ProjectionHead(
                self.encoder.config.hidden_size,
                hidden_dim,
                output_dim
            )
            self.embedding_dim = output_dim
        else:
            self.embedding_dim = self.encoder.config.hidden_size

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
            projection_hidden_dim: int | None,
            projection_output_dim: int | None,
            model_name: str = "jhu-clsp/ettin-encoder-68m"
    ):
        super().__init__()
        self.embedder = Embedder(
            hidden_dim=projection_hidden_dim,
            output_dim=projection_output_dim,
            model_name=model_name
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.embedder.embedding_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, 2),
        )


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        embedding = self.embedder(input_ids, attention_mask)
        return self.mlp(embedding), embedding
