from __future__ import annotations

from numgrad.nn.functional import max_pool2d
from numgrad.nn.module import Module
from numgrad.tensor import Tensor


class MaxPool2d(Module):
    def __init__(self, kernel_size: int, stride: int | None = None) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x: Tensor) -> Tensor:
        return max_pool2d(x, self.kernel_size, self.stride)
