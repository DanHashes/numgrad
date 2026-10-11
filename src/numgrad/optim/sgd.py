from __future__ import annotations

import numpy as np
import numpy.typing as npt

from numgrad.optim.optimizer import Optimizer
from numgrad.tensor import Tensor

Array = npt.NDArray[np.float64]


class SGD(Optimizer):
    """Stochastic gradient descent with optional (classic, undampened) momentum:

    v_t = momentum * v_{t-1} + grad
    p_t = p_{t-1} - lr * v_t
    """

    def __init__(self, parameters: list[Tensor], lr: float, momentum: float = 0.0) -> None:
        super().__init__(parameters, lr)
        self.momentum = momentum
        self._velocity: list[Array] = [np.zeros_like(p.data) for p in self.parameters]

    def step(self) -> None:
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            if self.momentum > 0.0:
                self._velocity[i] = self.momentum * self._velocity[i] + p.grad
                update = self._velocity[i]
            else:
                update = p.grad
            p.data = p.data - self.lr * update
