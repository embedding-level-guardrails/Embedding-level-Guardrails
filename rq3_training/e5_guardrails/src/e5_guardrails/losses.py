"""Supervised contrastive loss on binary safety labels (Khosla et al., 2020, L_out).

For anchor i with positives P(i) (other rows of the same safety class) and all
other rows A(i) = batch \\ {i} (positives + negatives of the other class):

    l_i = -1/|P(i)| * sum_{p in P(i)} log( exp(z_i.z_p / t) / sum_{a in A(i)} exp(z_i.z_a / t) )

z are L2-normalized, so z_i.z_p is cosine similarity. Self-comparisons are
excluded from both numerator and denominator. Anchors with no positive in the
batch are excluded from the mean and counted; the balanced sampler makes them
impossible in training. The contrastive set is exactly the batch (no memory
bank, no gradient accumulation).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def supcon_masks(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(positive, negative) boolean masks; the diagonal is in neither."""
    labels = labels.view(-1)
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    return same & ~eye, ~same


def supcon_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float) -> tuple[torch.Tensor, dict[str, int]]:
    z = F.normalize(z.float(), dim=-1)
    positive, _ = supcon_masks(labels)
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    logits = (z @ z.T / temperature).masked_fill(eye, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    n_pos = positive.sum(dim=1)
    has_pos = n_pos > 0
    stats = {"anchors": len(z), "anchors_without_positive": int((~has_pos).sum())}
    if not has_pos.any():
        return z.sum() * 0.0, stats
    pos_log_prob = log_prob.masked_fill(~positive, 0.0).sum(dim=1)
    loss = -(pos_log_prob[has_pos] / n_pos[has_pos]).mean()
    return loss, stats
