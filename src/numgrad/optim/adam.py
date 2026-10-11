from __future__ import annotations

import numpy as np
import numpy.typing as npt

from numgrad.optim.optimizer import Optimizer
from numgrad.tensor import Tensor

Array = npt.NDArray[np.float64]


class Adam(Optimizer):
    """Adam (Kingma & Ba, 2015), with explicit bias correction.

    m and v are exponential moving averages of the gradient and its square,
    both initialized at zero — which biases both toward zero for small t, but
    v much more severely and for much longer, since beta2 (0.999) is far
    closer to 1 than beta1 (0.9). Dividing by (1 - beta^t) corrects each back
    up toward its true moment. Skipping that correction lets v's larger,
    longer-lived bias dominate m's: the update m/sqrt(v) ends up too large
    (by over 3x at t=1 with the defaults below) and stays inflated for many
    steps, rather than too small — exactly the instability bias correction
    exists to prevent.
    """

    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self._m: list[Array] = [np.zeros_like(p.data) for p in self.parameters]
        self._v: list[Array] = [np.zeros_like(p.data) for p in self.parameters]

    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            g = p.grad
            self._m[i] = self.beta1 * self._m[i] + (1 - self.beta1) * g
            self._v[i] = self.beta2 * self._v[i] + (1 - self.beta2) * (g**2)
            m_hat = self._m[i] / (1 - self.beta1**self.t)
            v_hat = self._v[i] / (1 - self.beta2**self.t)
            p.data = p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
