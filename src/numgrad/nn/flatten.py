from __future__ import annotations

from numgrad.nn.module import Module
from numgrad.tensor import Tensor


class Flatten(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.reshape(x.shape[0], -1)
