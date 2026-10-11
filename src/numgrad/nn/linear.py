from __future__ import annotations

import numpy as np

from numgrad.nn.module import Module
from numgrad.tensor import Tensor


class Linear(Module):
    """y = x @ W + b, with W initialized via Kaiming/He normal init.

    He et al. (2015): for a layer feeding into ReLU, drawing weights with
    Var(W) = 2 / fan_in keeps the variance of activations roughly constant
    across layers, countering the fact that ReLU zeroes out about half its
    inputs on average (a plain Xavier/Glorot init, derived assuming a linear
    or symmetric activation, would let variance shrink layer over layer).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        rng = rng if rng is not None else np.random.default_rng()
        std = np.sqrt(2.0 / in_features)
        self.weight = Tensor(
            rng.normal(0.0, std, size=(in_features, out_features)), requires_grad=True
        )
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias
