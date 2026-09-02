from __future__ import annotations

import numpy as np
import pytest
from helpers import gradcheck

from numgrad import Tensor, no_grad

SEEDS = [0, 1, 2]

BROADCAST_SHAPES: list[tuple[tuple[int, ...], tuple[int, ...]]] = [
    ((3, 4), (3, 4)),
    ((3, 4), (4,)),  # vector + matrix
    ((3, 4), (1, 4)),  # row + matrix
    ((3, 4), ()),  # scalar + matrix
    ((2, 3, 4), (1, 3, 4)),  # mismatched batch dim
]

BINARY_OPS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
}


def rand_tensor(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    low: float = -2.0,
    high: float = 2.0,
) -> Tensor:
    return Tensor(rng.uniform(low, high, size=shape), requires_grad=True)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("shape_a,shape_b", BROADCAST_SHAPES)
@pytest.mark.parametrize("op_name,op", list(BINARY_OPS.items()))
def test_binary_op_gradcheck(
    seed: int,
    shape_a: tuple[int, ...],
    shape_b: tuple[int, ...],
    op_name: str,
    op: object,
) -> None:
    rng = np.random.default_rng(seed)
    a = rand_tensor(rng, shape_a)
    b = rand_tensor(rng, shape_b)
    gradcheck(op, [a, b])  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("shape_a,shape_b", BROADCAST_SHAPES)
def test_div_gradcheck(
    seed: int, shape_a: tuple[int, ...], shape_b: tuple[int, ...]
) -> None:
    rng = np.random.default_rng(seed)
    a = rand_tensor(rng, shape_a)
    b = rand_tensor(rng, shape_b, low=0.5, high=2.0)  # keep denominator away from 0
    gradcheck(lambda a, b: a / b, [a, b])


@pytest.mark.parametrize("seed", SEEDS)
def test_neg_gradcheck(seed: int) -> None:
    rng = np.random.default_rng(seed)
    a = rand_tensor(rng, (3, 4))
    gradcheck(lambda a: -a, [a])


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("exponent", [2.0, 3.0, 0.5])
def test_pow_gradcheck(seed: int, exponent: float) -> None:
    rng = np.random.default_rng(seed)
    a = rand_tensor(rng, (3, 4), low=0.5, high=2.0)  # keep base positive
    gradcheck(lambda a: a**exponent, [a])


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(
    "shape_a,shape_b",
    [
        ((3, 4), (4, 5)),
        ((2, 3, 4), (2, 4, 5)),  # batched
        ((1, 3, 4), (2, 4, 5)),  # mismatched/broadcast batch dim
    ],
)
def test_matmul_gradcheck(
    seed: int, shape_a: tuple[int, ...], shape_b: tuple[int, ...]
) -> None:
    rng = np.random.default_rng(seed)
    a = rand_tensor(rng, shape_a)
    b = rand_tensor(rng, shape_b)
    gradcheck(lambda a, b: a @ b, [a, b])


@pytest.mark.parametrize("seed", SEEDS)
def test_scalar_operand_gradcheck(seed: int) -> None:
    rng = np.random.default_rng(seed)
    a = rand_tensor(rng, (3, 4), low=0.5, high=2.0)
    gradcheck(lambda a: a + 2.0, [a])
    gradcheck(lambda a: 2.0 + a, [a])
    gradcheck(lambda a: a - 2.0, [a])
    gradcheck(lambda a: 2.0 - a, [a])
    gradcheck(lambda a: a * 2.0, [a])
    gradcheck(lambda a: 2.0 * a, [a])
    gradcheck(lambda a: a / 2.0, [a])
    gradcheck(lambda a: 2.0 / a, [a])


def test_backward_requires_grad() -> None:
    a = Tensor([1.0, 2.0], requires_grad=False)
    with pytest.raises(RuntimeError):
        a.backward()


def test_backward_requires_explicit_grad_for_non_scalar() -> None:
    a = Tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(RuntimeError):
        a.backward()


def test_backward_grad_shape_mismatch() -> None:
    a = Tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(ValueError):
        a.backward(np.ones(3))


def test_backward_scalar_default_seed() -> None:
    a = Tensor(3.0, requires_grad=True)
    b = Tensor(4.0, requires_grad=True)
    c = a * b
    c.backward()
    assert a.grad is not None and np.isclose(a.grad, 4.0)
    assert b.grad is not None and np.isclose(b.grad, 3.0)


def test_grad_accumulates_across_multiple_backward_calls() -> None:
    a = Tensor(2.0, requires_grad=True)
    b = a * a
    b.backward()
    b.backward()
    assert a.grad is not None and np.isclose(a.grad, 8.0)  # 2*2 + 2*2


def test_diamond_graph_accumulates_correctly() -> None:
    # c = a + a; d = c * c => d = 4a^2, dd/da = 8a
    a = Tensor(3.0, requires_grad=True)
    c = a + a
    d = c * c
    d.backward()
    assert a.grad is not None and np.isclose(a.grad, 8 * 3.0)


def test_no_grad_disables_graph_building() -> None:
    a = Tensor([1.0, 2.0], requires_grad=True)
    b = Tensor([3.0, 4.0], requires_grad=True)
    with no_grad():
        c = a + b
    assert c.requires_grad is False
    with pytest.raises(RuntimeError):
        c.backward(np.ones(2))


def test_no_grad_restores_previous_state() -> None:
    from numgrad.tensor.grad_mode import is_grad_enabled

    assert is_grad_enabled() is True
    with no_grad():
        assert is_grad_enabled() is False
    assert is_grad_enabled() is True


def test_intermediate_node_receives_grad() -> None:
    a = Tensor(2.0, requires_grad=True)
    b = Tensor(3.0, requires_grad=True)
    c = a + b  # intermediate, non-leaf
    d = c * c
    d.backward()
    assert c.grad is not None and np.isclose(c.grad, 2 * (2.0 + 3.0))
