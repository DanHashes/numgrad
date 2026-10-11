from __future__ import annotations

from numgrad.nn.module import Module
from numgrad.tensor import Tensor


class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        super().__init__()
        self.layers: list[Module] = list(layers)
        for i, layer in enumerate(self.layers):
            self._modules[str(i)] = layer

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
