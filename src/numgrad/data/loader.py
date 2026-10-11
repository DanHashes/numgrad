from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


class DataLoader:
    """Yields shuffled mini-batches of (X, y) each epoch (each fresh iteration)."""

    def __init__(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        batch_size: int,
        shuffle: bool = True,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        if len(self.X) != len(self.y):
            raise ValueError(
                f"X and y must have the same length, got {len(self.X)} vs {len(self.y)}"
            )
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = rng if rng is not None else np.random.default_rng()

    def __iter__(self) -> Iterator[tuple[Array, Array]]:
        n = len(self.X)
        indices = np.arange(n)
        if self.shuffle:
            self.rng.shuffle(indices)
        for start in range(0, n, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            yield self.X[batch_idx], self.y[batch_idx]

    def __len__(self) -> int:
        return math.ceil(len(self.X) / self.batch_size)
