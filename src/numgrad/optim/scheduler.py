from __future__ import annotations

import math

from numgrad.optim.optimizer import Optimizer


class CosineAnnealingLR:
    """Anneals an optimizer's .lr from its initial value down to eta_min
    following a half-cosine curve over t_max steps, then holds at eta_min.
    Kept standalone (not baked into Optimizer) so any optimizer with a
    mutable .lr attribute can use it.
    """

    def __init__(self, optimizer: Optimizer, t_max: int, eta_min: float = 0.0) -> None:
        self.optimizer = optimizer
        self.t_max = t_max
        self.eta_min = eta_min
        self.base_lr = optimizer.lr
        self.t = 0

    def step(self) -> None:
        self.t += 1
        progress = min(self.t, self.t_max) / self.t_max
        self.optimizer.lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
            1 + math.cos(math.pi * progress)
        )
