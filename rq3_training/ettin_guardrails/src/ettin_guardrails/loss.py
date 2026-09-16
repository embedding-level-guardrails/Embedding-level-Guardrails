import torch
import torch.nn.functional as F
from torch import Tensor


def info_nce_loss(embeddings: Tensor, category_ids: Tensor, temperature: float, eps: float) -> Tensor:
    batch_size = embeddings.shape[0]
    if batch_size < 2:
        return embeddings.sum() * 0

    normalized_embeddings = F.normalize(embeddings, p=2, dim=1, eps=eps)
    logits = (normalized_embeddings @ normalized_embeddings.T) / temperature

    diagonal = torch.eye(batch_size, device=category_ids.device, dtype=torch.bool)
    positives = category_ids[:, None].eq(category_ids[None, :]) & ~diagonal
    positive_count = positives.sum(dim=1)
    valid = (positive_count > 0) & (positive_count < batch_size - 1)

    log_denominator = logits.masked_fill(diagonal, -torch.inf).logsumexp(dim=1)
    mean_positive = logits.masked_fill(~positives, 0).sum(dim=1) / positive_count.clamp_min(1)

    losses = (log_denominator - mean_positive).masked_fill(~valid, 0)
    return losses.sum() / valid.sum().clamp_min(1)
