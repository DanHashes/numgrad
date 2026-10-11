from __future__ import annotations

from numgrad.tensor import Tensor


class Optimizer:
    def __init__(self, parameters: list[Tensor], lr: float) -> None:
        self.parameters = list(parameters)
        self.lr = lr

    def zero_grad(self) -> None:
        for p in self.parameters:
            p.grad = None

    def step(self) -> None:
        raise NotImplementedError
