from __future__ import annotations

import numpy as np

from numgrad.nn.functional import conv2d
from numgrad.nn.module import Module
from numgrad.tensor import Tensor


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        rng = rng if rng is not None else np.random.default_rng()
        fan_in = in_channels * kernel_size * kernel_size
        std = np.sqrt(2.0 / fan_in)
        self.weight = Tensor(
            rng.normal(0.0, std, size=(out_channels, in_channels, kernel_size, kernel_size)),
            requires_grad=True,
        )
        self.bias = Tensor(np.zeros(out_channels), requires_grad=True)
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        return conv2d(x, self.weight, self.bias, self.stride, self.padding)
