"""Targeted tests for real coverage gaps found via `pytest --cov` in Step 9:
multi-axis (tuple) reductions were never exercised, and a few small pieces of
public API (Tensor.ndim/.size/.__repr__) had no direct test. (A separate
category of gap -- `if not self.requires_grad: return []` guards inside
every single-parent op's backward closure -- turned out to be unreachable
dead code given how `_make` gates graph construction, and was deleted
rather than tested; see PROGRESS.md.)
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import gradcheck

from numgrad import Tensor
from numgrad.nn import Module
from numgrad.optim import SGD, Adam, Optimizer


def test_sum_over_multiple_axes_gradcheck() -> None:
    rng = np.random.default_rng(0)
    a = Tensor(rng.uniform(-2, 2, size=(2, 3, 4)), requires_grad=True)
    gradcheck(lambda a: a.sum(axis=(0, 2)), [a])


def test_mean_over_multiple_axes_gradcheck() -> None:
    rng = np.random.default_rng(0)
    a = Tensor(rng.uniform(-2, 2, size=(2, 3, 4)), requires_grad=True)
    gradcheck(lambda a: a.mean(axis=(0, 2)), [a])


def test_max_over_multiple_axes_gradcheck() -> None:
    rng = np.random.default_rng(0)
    a = Tensor(rng.uniform(-2, 2, size=(2, 3, 4)), requires_grad=True)
    gradcheck(lambda a: a.max(axis=(1, 2)), [a])


def test_sum_multi_axis_matches_plain_numpy() -> None:
    a = Tensor(np.arange(24, dtype=np.float64).reshape(2, 3, 4), requires_grad=True)
    out = a.sum(axis=(0, 2))
    np.testing.assert_allclose(out.data, np.arange(24).reshape(2, 3, 4).sum(axis=(0, 2)))
    out.backward(np.ones(3))
    assert a.grad is not None and a.grad.shape == (2, 3, 4)


def test_tensor_ndim_and_size_properties() -> None:
    a = Tensor(np.zeros((2, 3, 4)))
    assert a.ndim == 3
    assert a.size == 24


def test_pow_rejects_tensor_exponent() -> None:
    a = Tensor([2.0, 3.0])
    b = Tensor([2.0, 2.0])
    with pytest.raises(TypeError, match="scalar"):
        a**b  # type: ignore[operator]


def test_tensor_repr_includes_requires_grad() -> None:
    a = Tensor([1.0, 2.0], requires_grad=True)
    text = repr(a)
    assert "requires_grad=True" in text
    assert "Tensor(" in text


def test_module_base_forward_is_abstract() -> None:
    with pytest.raises(NotImplementedError):
        Module().forward(Tensor([1.0]))


def test_optimizer_base_step_is_abstract() -> None:
    p = Tensor(np.zeros(1))
    with pytest.raises(NotImplementedError):
        Optimizer([p], lr=0.1).step()


def test_adam_skips_parameters_with_no_gradient() -> None:
    p1 = Tensor(np.array([1.0]))
    p2 = Tensor(np.array([5.0]))
    p1.grad = np.array([1.0])
    # p2.grad left as None
    opt = Adam([p1, p2], lr=0.1)
    opt.step()
    assert p1.data[0] != 1.0  # updated
    assert p2.data[0] == 5.0  # untouched


def test_sgd_momentum_buffer_persists_across_steps() -> None:
    p = Tensor(np.array([0.0]))
    opt = SGD([p], lr=0.1, momentum=0.9)
    p.grad = np.array([1.0])
    opt.step()
    first = p.data[0]
    p.grad = np.array([1.0])
    opt.step()
    second_step_delta = p.data[0] - first
    # with momentum, the second step's update is larger than the first's
    # (velocity accumulates), not identical to a plain lr*grad step
    assert abs(second_step_delta) > 0.1
