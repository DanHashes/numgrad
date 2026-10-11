from __future__ import annotations

import numpy as np

from numgrad.nn.module import Module
from numgrad.tensor import Tensor


class Dropout(Module):
    """Inverted dropout: scales surviving activations by 1/keep_prob at train
    time, so eval time needs no rescaling and is a plain no-op.
    """

    def __init__(self, p: float = 0.5, rng: np.random.Generator | None = None) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout probability p must be in [0, 1)")
        self.p = p
        self._rng = rng if rng is not None else np.random.default_rng()

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        mask = (self._rng.uniform(size=x.shape) < keep_prob).astype(np.float64) / keep_prob
        return x * Tensor(mask)
