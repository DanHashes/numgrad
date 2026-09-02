from __future__ import annotations

from collections.abc import Callable
from typing import Any, Union

import numpy as np
import numpy.typing as npt

from numgrad.tensor.grad_mode import is_grad_enabled

Array = npt.NDArray[np.float64]
TensorLike = Union["Tensor", float, int, npt.ArrayLike]
BackwardFn = Callable[[Array], list[tuple["Tensor", Array]]]


def _unbroadcast(grad: Array, target_shape: tuple[int, ...]) -> Array:
    """Sum-reduce a broadcasted gradient back down to target_shape."""
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(target_shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(target_shape)


Axis = int | tuple[int, ...] | None


def _normalize_axes(axis: Axis, ndim: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, int):
        return (axis % ndim,)
    return tuple(a % ndim for a in axis)


def _restore_reduced_dims(grad: Array, shape: tuple[int, ...], axis: Axis, keepdims: bool) -> Array:
    """Re-insert axes collapsed by a reduction (as size-1) so grad can broadcast against shape."""
    if not keepdims:
        for ax in sorted(_normalize_axes(axis, len(shape))):
            grad = np.expand_dims(grad, ax)
    return grad


def _ensure_tensor(x: TensorLike) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(x)


def _make(data: Array, parents: tuple[Tensor, ...], backward: BackwardFn | None) -> Tensor:
    """Build the output Tensor for an op, wiring up the graph only if needed."""
    requires_grad = is_grad_enabled() and any(p.requires_grad for p in parents)
    out = Tensor(data, requires_grad=requires_grad)
    if requires_grad:
        out._prev = parents
        out._backward = backward
    return out


class Tensor:
    def __init__(
        self,
        data: npt.ArrayLike,
        requires_grad: bool = False,
        _prev: tuple[Tensor, ...] = (),
        _op: str = "",
    ) -> None:
        self.data: Array = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad: Array | None = None
        self._prev = _prev
        # _backward(grad_output) -> contributions to push onto each parent that
        # needs one; kept as a pure function of the incoming gradient (not of
        # any node's .grad) so repeated backward() calls don't compound.
        self._backward: BackwardFn | None = None
        self._op = _op

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    def item(self) -> float:
        return float(self.data.item())

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def _accumulate_grad(self, grad: Array) -> None:
        if self.grad is None:
            self.grad = grad.copy()
        else:
            self.grad = self.grad + grad

    def backward(self, grad: npt.ArrayLike | None = None) -> None:
        if not self.requires_grad:
            raise RuntimeError("called backward() on a tensor that does not require grad")

        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be specified explicitly for non-scalar outputs")
            seed = np.ones_like(self.data)
        else:
            seed = np.asarray(grad, dtype=np.float64)
            if seed.shape != self.data.shape:
                raise ValueError(
                    f"grad shape {seed.shape} does not match tensor shape {self.data.shape}"
                )

        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(node: Tensor) -> None:
            if id(node) not in visited:
                visited.add(id(node))
                for parent in node._prev:
                    build(parent)
                topo.append(node)

        build(self)

        # Gradients relayed during THIS call only, keyed by id(); kept separate
        # from the persistent, cross-call node.grad so multiple backward()
        # calls on the same graph accumulate correctly instead of compounding.
        pending: dict[int, Array] = {id(self): seed}
        for node in reversed(topo):
            incoming = pending.pop(id(node), None)
            if incoming is None:
                continue
            node._accumulate_grad(incoming)
            if node._backward is not None:
                for parent, contribution in node._backward(incoming):
                    if id(parent) in pending:
                        pending[id(parent)] = pending[id(parent)] + contribution
                    else:
                        pending[id(parent)] = contribution

    # ---- arithmetic ops ----

    def __add__(self, other: TensorLike) -> Tensor:
        other_t = _ensure_tensor(other)
        data = self.data + other_t.data

        def _backward(grad_output: Array) -> list[tuple[Tensor, Array]]:
            contributions: list[tuple[Tensor, Array]] = []
            if self.requires_grad:
                contributions.append((self, _unbroadcast(grad_output, self.data.shape)))
            if other_t.requires_grad:
                contributions.append((other_t, _unbroadcast(grad_output, other_t.data.shape)))
            return contributions

        return _make(data, (self, other_t), _backward)

    __radd__ = __add__

    def __sub__(self, other: TensorLike) -> Tensor:
        other_t = _ensure_tensor(other)
        data = self.data - other_t.data

        def _backward(grad_output: Array) -> list[tuple[Tensor, Array]]:
            contributions: list[tuple[Tensor, Array]] = []
            if self.requires_grad:
                contributions.append((self, _unbroadcast(grad_output, self.data.shape)))
            if other_t.requires_grad:
                contributions.append((other_t, _unbroadcast(-grad_output, other_t.data.shape)))
            return contributions

        return _make(data, (self, other_t), _backward)

    def __rsub__(self, other: TensorLike) -> Tensor:
        return _ensure_tensor(other).__sub__(self)

    def __mul__(self, other: TensorLike) -> Tensor:
        other_t = _ensure_tensor(other)
        data = self.data * other_t.data

        def _backward(grad_output: Array) -> list[tuple[Tensor, Array]]:
            contributions: list[tuple[Tensor, Array]] = []
            if self.requires_grad:
                contributions.append(
                    (self, _unbroadcast(grad_output * other_t.data, self.data.shape))
                )
            if other_t.requires_grad:
                contributions.append(
                    (other_t, _unbroadcast(grad_output * self.data, other_t.data.shape))
                )
            return contributions

        return _make(data, (self, other_t), _backward)

    __rmul__ = __mul__

    def __truediv__(self, other: TensorLike) -> Tensor:
        other_t = _ensure_tensor(other)
        data = self.data / other_t.data

        def _backward(grad_output: Array) -> list[tuple[Tensor, Array]]:
            contributions: list[tuple[Tensor, Array]] = []
            if self.requires_grad:
                contributions.append(
                    (self, _unbroadcast(grad_output / other_t.data, self.data.shape))
                )
            if other_t.requires_grad:
                grad_other = -grad_output * self.data / (other_t.data**2)
                contributions.append((other_t, _unbroadcast(grad_other, other_t.data.shape)))
            return contributions

        return _make(data, (self, other_t), _backward)

    def __rtruediv__(self, other: TensorLike) -> Tensor:
        return _ensure_tensor(other).__truediv__(self)

    def __neg__(self) -> Tensor:
        data = -self.data

        def _backward(grad_output: Array) -> list[tuple[Tensor, Array]]:
            return [(self, -grad_output)] if self.requires_grad else []

        return _make(data, (self,), _backward)

    def __pow__(self, exponent: float) -> Tensor:
        if not isinstance(exponent, (int, float)):
            raise TypeError("Tensor.__pow__ only supports a scalar (int/float) exponent")
        data = self.data**exponent

        def _backward(grad_output: Array) -> list[tuple[Tensor, Array]]:
            if not self.requires_grad:
                return []
            local_grad = exponent * self.data ** (exponent - 1)
            return [(self, grad_output * local_grad)]

        return _make(data, (self,), _backward)

    def __matmul__(self, other: TensorLike) -> Tensor:
        other_t = _ensure_tensor(other)
        data = self.data @ other_t.data

        def _backward(grad_output: Array) -> list[tuple[Tensor, Array]]:
            contributions: list[tuple[Tensor, Array]] = []
            if self.requires_grad:
                other_T = np.swapaxes(other_t.data, -1, -2)
                grad_self = grad_output @ other_T
                contributions.append((self, _unbroadcast(grad_self, self.data.shape)))
            if other_t.requires_grad:
                self_T = np.swapaxes(self.data, -1, -2)
                grad_other = self_T @ grad_output
                contributions.append((other_t, _unbroadcast(grad_other, other_t.data.shape)))
            return contributions

        return _make(data, (self, other_t), _backward)

    # ---- reductions ----

    def sum(self, axis: Axis = None, keepdims: bool = False) -> Tensor:
        data = self.data.sum(axis=axis, keepdims=keepdims)
        shape = self.data.shape

        def _backward(grad_output: Array) -> list[tuple[Tensor, Array]]:
            if not self.requires_grad:
                return []
            restored = _restore_reduced_dims(grad_output, shape, axis, keepdims)
            return [(self, np.broadcast_to(restored, shape))]

        return _make(data, (self,), _backward)

    def mean(self, axis: Axis = None, keepdims: bool = False) -> Tensor:
        data = self.data.mean(axis=axis, keepdims=keepdims)
        shape = self.data.shape
        count = np.prod([shape[ax] for ax in _normalize_axes(axis, len(shape))])

        def _backward(grad_output: Array) -> list[tuple[Tensor, Array]]:
            if not self.requires_grad:
                return []
            restored = _restore_reduced_dims(grad_output, shape, axis, keepdims)
            return [(self, np.broadcast_to(restored, shape) / count)]

        return _make(data, (self,), _backward)

    def max(self, axis: Axis = None, keepdims: bool = False) -> Tensor:
        return self._minmax(np.max, axis, keepdims)

    def min(self, axis: Axis = None, keepdims: bool = False) -> Tensor:
        return self._minmax(np.min, axis, keepdims)

    def _minmax(self, reducer: Callable[..., Array], axis: Axis, keepdims: bool) -> Tensor:
        reduced = reducer(self.data, axis=axis, keepdims=True)
        # Ties split the gradient evenly across the tied elements.
        mask = (self.data == reduced).astype(np.float64)
        mask = mask / mask.sum(axis=axis, keepdims=True)
        shape = self.data.shape
        data = reduced if keepdims else np.squeeze(reduced, axis=axis)

        def _backward(grad_output: Array) -> list[tuple[Tensor, Array]]:
            if not self.requires_grad:
                return []
            restored = _restore_reduced_dims(grad_output, shape, axis, keepdims)
            return [(self, mask * restored)]

        return _make(data, (self,), _backward)

    # ---- shape ops ----

    def reshape(self, *shape: int | tuple[int, ...]) -> Tensor:
        new_shape = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        data = self.data.reshape(new_shape)
        orig_shape = self.data.shape

        def _backward(grad_output: Array) -> list[tuple[Tensor, Array]]:
            return [(self, grad_output.reshape(orig_shape))] if self.requires_grad else []

        return _make(data, (self,), _backward)

    def transpose(self, *axes: int) -> Tensor:
        axes_: tuple[int, ...] | None = axes if axes else None
        data = self.data.transpose(axes_)

        def _backward(grad_output: Array) -> list[tuple[Tensor, Array]]:
            if not self.requires_grad:
                return []
            if axes_ is None:
                grad = grad_output.transpose()
            else:
                grad = grad_output.transpose(tuple(np.argsort(axes_)))
            return [(self, grad)]

        return _make(data, (self,), _backward)

    @property
    def T(self) -> Tensor:
        return self.transpose()

    def __getitem__(self, idx: Any) -> Tensor:
        data = self.data[idx]
        shape = self.data.shape
        dtype = self.data.dtype

        def _backward(grad_output: Array) -> list[tuple[Tensor, Array]]:
            if not self.requires_grad:
                return []
            grad = np.zeros(shape, dtype=dtype)
            np.add.at(grad, idx, grad_output)
            return [(self, grad)]

        return _make(data, (self,), _backward)

    # ---- transcendental ops ----

    def exp(self) -> Tensor:
        data = np.exp(self.data)

        def _backward(grad_output: Array) -> list[tuple[Tensor, Array]]:
            return [(self, grad_output * data)] if self.requires_grad else []

        return _make(data, (self,), _backward)

    def log(self) -> Tensor:
        data = np.log(self.data)

        def _backward(grad_output: Array) -> list[tuple[Tensor, Array]]:
            return [(self, grad_output / self.data)] if self.requires_grad else []

        return _make(data, (self,), _backward)
