from __future__ import annotations

from typing import Any

from numgrad.tensor import Tensor


class Module:
    """Base class for all layers and layer containers.

    Any Tensor with requires_grad=True assigned as a direct attribute is
    auto-registered as a parameter; any Module assigned as a direct attribute
    is auto-registered as a submodule, so .parameters() and .train()/.eval()
    recurse through arbitrarily nested structures.
    """

    def __init__(self) -> None:
        self._parameters: dict[str, Tensor] = {}
        self._modules: dict[str, Module] = {}
        self.training: bool = True

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Tensor) and value.requires_grad:
            self.__dict__.setdefault("_parameters", {})[name] = value
        elif isinstance(value, Module):
            self.__dict__.setdefault("_modules", {})[name] = value
        object.__setattr__(self, name, value)

    def parameters(self) -> list[Tensor]:
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def train(self, mode: bool = True) -> Module:
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self) -> Module:
        return self.train(False)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)
