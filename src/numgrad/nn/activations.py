from __future__ import annotations

import math

from numgrad.nn.module import Module
from numgrad.tensor import Tensor

_GELU_COEFF = math.sqrt(2.0 / math.pi)


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class GELU(Module):
    """GELU via the tanh approximation (Hendrycks & Gimpel, 2016):

    0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Built entirely from existing Tensor primitives (pow, mul, add, tanh) —
    no dedicated backward needed, same compositional approach as cross_entropy.
    """

    def forward(self, x: Tensor) -> Tensor:
        inner = _GELU_COEFF * (x + 0.044715 * x**3)
        return 0.5 * x * (1.0 + inner.tanh())
