"""Batch sampler shared by every training mode. Torch-free.

The batch sequence depends only on (labels, partners, batch_size, seed), never
on the loss, so CE, SupCon and CE + SupCon see exactly the same texts in the
same order at a given seed.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

import numpy as np


class BalancedBatchSampler:
    """Index batches with ``batch_size / 2`` safe and ``batch_size / 2`` harmful rows.

    With ``batch_size >= 4`` every anchor has at least one same-class positive in
    its batch, which SupCon needs. Classes are drawn without replacement from a
    reshuffled pool (every row is seen before any repeats).

    ``partners`` (RQ2) maps a row to rows it is paired with. With probability
    ``pair_prob`` a drawn row pulls one partner into its batch by replacing a
    random unprotected slot of the partner's class, so the class balance is
    unchanged. Usable as a DataLoader ``batch_sampler``.
    """

    def __init__(
        self,
        labels: Sequence[int],
        batch_size: int,
        num_batches: int,
        seed: int,
        partners: Mapping[int, Sequence[int]] | None = None,
        pair_prob: float = 0.0,
    ) -> None:
        if batch_size < 4 or batch_size % 2:
            raise ValueError("batch_size must be even and >= 4 (SupCon needs >= 2 rows per class)")
        self.labels = np.asarray(labels)
        pools = [np.flatnonzero(self.labels == c) for c in (0, 1)]
        if any(len(pool) == 0 for pool in pools):
            raise ValueError("Both classes must be present")
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.partners = {int(k): [int(x) for x in v] for k, v in (partners or {}).items() if len(v)}
        self.pair_prob = pair_prob
        self._rng = np.random.default_rng(seed)
        self._streams = [self._cycle(pool) for pool in pools]

    def _cycle(self, pool: np.ndarray) -> Iterator[int]:
        while True:
            yield from self._rng.permutation(pool).tolist()

    def __len__(self) -> int:
        return self.num_batches

    def _add_partners(self, batch: list[int]) -> list[int]:
        protected: set[int] = set()
        for slot in range(len(batch)):
            anchor = batch[slot]
            if slot in protected or anchor not in self.partners or self._rng.random() >= self.pair_prob:
                continue
            partner = self.partners[anchor][self._rng.integers(len(self.partners[anchor]))]
            if partner in batch:
                protected.add(slot)
                continue
            free = [s for s in range(len(batch)) if s != slot and s not in protected
                    and self.labels[batch[s]] == self.labels[partner]]
            if not free:
                continue
            target = free[self._rng.integers(len(free))]
            batch[target] = partner
            protected.update((slot, target))
        return batch

    def __iter__(self) -> Iterator[list[int]]:
        half = self.batch_size // 2
        for _ in range(self.num_batches):
            batch = [next(stream) for stream in self._streams for _ in range(half)]
            if self.partners and self.pair_prob > 0:
                batch = self._add_partners(batch)
            yield batch
