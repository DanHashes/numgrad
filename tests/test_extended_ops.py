from __future__ import annotations

import numpy as np
import pytest
from helpers import gradcheck

from numgrad import Tensor, cross_entropy, log_softmax
from numgrad.tensor.functional import clip_grad_norm

SEEDS = [0, 1, 2]


def rand_tensor(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    low: float = -2.0,
    high: float = 2.0,
) -> Tensor:
    return Tensor(rng.uniform(low, high, size=shape), requires_grad=True)


# ---- reductions ----


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("axis,keepdims", [(None, False), (0, False), (1, False), (1, True)])
def test_sum_gradcheck(seed: int, axis: int | None, keepdims: bool) -> None:
    rng = np.random.default_rng(seed)
    a = rand_tensor(rng, (3, 4))
    gradcheck(lambda a: a.sum(axis=axis, keepdims=keepdims), [a])


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("axis,keepdims", [(None, False), (0, False), (1, True)])
def test_mean_gradcheck(seed: int, axis: int | None, keepdims: bool) -> None:
    rng = np.random.default_rng(seed)
    a = rand_tensor(rng, (3, 4))
    gradcheck(lambda a: a.mean(axis=axis, keepdims=keepdims), [a])


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("axis,keepdims", [(None, False), (0, False), (1, True)])
def test_max_gradcheck(seed: int, axis: int | None, keepdims: bool) -> None:
    rng = np.random.default_rng(seed)
    # Add small per-element jitter so ties are vanishingly unlikely; the tie
    # case is covered explicitly below.
    a = rand_tensor(rng, (3, 4))
    gradcheck(lambda a: a.max(axis=axis, keepdims=keepdims), [a])


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("axis,keepdims", [(None, False), (0, False), (1, True)])
def test_min_gradcheck(seed: int, axis: int | None, keepdims: bool) -> None:
    rng = np.random.default_rng(seed)
    a = rand_tensor(rng, (3, 4))
    gradcheck(lambda a: a.min(axis=axis, keepdims=keepdims), [a])


def test_max_tie_splits_gradient_evenly() -> None:
    a = Tensor([1.0, 3.0, 3.0, 2.0], requires_grad=True)
    out = a.max()
    out.backward()
    assert a.grad is not None
    np.testing.assert_allclose(a.grad, [0.0, 0.5, 0.5, 0.0])


def test_min_tie_splits_gradient_evenly() -> None:
    a = Tensor([[1.0, 1.0, 5.0]], requires_grad=True)
    out = a.min(axis=1)
    out.backward(np.ones(1))
    assert a.grad is not None
    np.testing.assert_allclose(a.grad, [[0.5, 0.5, 0.0]])


# ---- shape ops ----


@pytest.mark.parametrize("seed", SEEDS)
def test_reshape_gradcheck(seed: int) -> None:
    rng = np.random.default_rng(seed)
    a = rand_tensor(rng, (3, 4))
    gradcheck(lambda a: a.reshape(4, 3), [a])
    gradcheck(lambda a: a.reshape((2, 6)), [a])


@pytest.mark.parametrize("seed", SEEDS)
def test_transpose_gradcheck(seed: int) -> None:
    rng = np.random.default_rng(seed)
    a = rand_tensor(rng, (3, 4))
    gradcheck(lambda a: a.transpose(), [a])
    gradcheck(lambda a: a.T, [a])

    b = rand_tensor(rng, (2, 3, 4))
    gradcheck(lambda b: b.transpose(2, 0, 1), [b])


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(
    "idx",
    [
        1,
        slice(1, 3),
        (slice(None), 2),
        (np.array([0, 1, 1]), np.array([0, 1, 2])),  # fancy indexing w/ a repeat
    ],
    ids=["int", "slice", "row-slice-col", "fancy-with-repeat"],
)
def test_getitem_gradcheck(seed: int, idx: object) -> None:
    rng = np.random.default_rng(seed)
    a = rand_tensor(rng, (3, 4))
    gradcheck(lambda a: a[idx], [a])


# ---- transcendental ops ----


@pytest.mark.parametrize("seed", SEEDS)
def test_exp_gradcheck(seed: int) -> None:
    rng = np.random.default_rng(seed)
    a = rand_tensor(rng, (3, 4), low=-1.0, high=1.0)
    gradcheck(lambda a: a.exp(), [a])


@pytest.mark.parametrize("seed", SEEDS)
def test_log_gradcheck(seed: int) -> None:
    rng = np.random.default_rng(seed)
    a = rand_tensor(rng, (3, 4), low=0.5, high=2.0)
    gradcheck(lambda a: a.log(), [a])


# ---- log_softmax / cross_entropy ----


@pytest.mark.parametrize("seed", SEEDS)
def test_log_softmax_gradcheck(seed: int) -> None:
    rng = np.random.default_rng(seed)
    a = rand_tensor(rng, (4, 5))
    gradcheck(lambda a: log_softmax(a, axis=-1), [a])


def test_log_softmax_rows_sum_to_one_in_prob_space() -> None:
    logits = Tensor(np.random.default_rng(0).normal(size=(4, 5)))
    probs = np.exp(log_softmax(logits).data)
    np.testing.assert_allclose(probs.sum(axis=-1), np.ones(4), rtol=1e-6)


@pytest.mark.parametrize("seed", SEEDS)
def test_cross_entropy_gradcheck(seed: int) -> None:
    rng = np.random.default_rng(seed)
    a = rand_tensor(rng, (4, 5))
    targets = rng.integers(0, 5, size=4)
    gradcheck(lambda a: cross_entropy(a, targets), [a])


def test_naive_log_softmax_overflows_on_large_logits() -> None:
    """Demonstrates the failure mode the log-sum-exp trick avoids."""
    x = np.array([[1e4, 1e4 + 1.0, 1e4 + 2.0]])

    def naive_log_softmax(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x)  # overflows to inf for x this large
        probs = exp_x / exp_x.sum(axis=-1, keepdims=True)
        return np.log(probs)

    naive_result = naive_log_softmax(x)
    assert np.any(~np.isfinite(naive_result)), (
        "expected the naive implementation to break on large logits"
    )


def test_stable_log_softmax_handles_large_logits() -> None:
    x = Tensor(np.array([[1e4, 1e4 + 1.0, 1e4 + 2.0]]), requires_grad=True)
    out = log_softmax(x)
    assert np.all(np.isfinite(out.data))
    out.sum().backward()
    assert x.grad is not None and np.all(np.isfinite(x.grad))


def test_stable_cross_entropy_handles_large_logits() -> None:
    x = Tensor(np.array([[1e4, 1e4 + 1.0, 1e4 + 2.0], [-1e4, 0.0, 1e4]]), requires_grad=True)
    targets = np.array([2, 0])
    loss = cross_entropy(x, targets)
    assert np.isfinite(loss.item())
    loss.backward()
    assert x.grad is not None and np.all(np.isfinite(x.grad))


# ---- clip_grad_norm ----


def test_clip_grad_norm_rescales_when_over_threshold() -> None:
    p1 = Tensor(np.zeros(2))
    p1.grad = np.array([3.0, 4.0])  # norm = 5
    total_norm = clip_grad_norm([p1], max_norm=1.0)
    assert total_norm == pytest.approx(5.0)
    assert p1.grad is not None
    np.testing.assert_allclose(np.linalg.norm(p1.grad), 1.0, rtol=1e-5)


def test_clip_grad_norm_leaves_grad_under_threshold_untouched() -> None:
    p1 = Tensor(np.zeros(2))
    p1.grad = np.array([0.3, 0.4])  # norm = 0.5
    total_norm = clip_grad_norm([p1], max_norm=1.0)
    assert total_norm == pytest.approx(0.5)
    np.testing.assert_allclose(p1.grad, [0.3, 0.4])


def test_clip_grad_norm_is_global_across_parameters() -> None:
    p1 = Tensor(np.zeros(1))
    p2 = Tensor(np.zeros(1))
    p1.grad = np.array([3.0])
    p2.grad = np.array([4.0])
    total_norm = clip_grad_norm([p1, p2], max_norm=1.0)
    assert total_norm == pytest.approx(5.0)
    assert p1.grad is not None and p2.grad is not None
    combined = np.sqrt(p1.grad.item() ** 2 + p2.grad.item() ** 2)
    assert combined == pytest.approx(1.0)


def test_clip_grad_norm_ignores_none_grads() -> None:
    p1 = Tensor(np.zeros(1))  # grad still None
    assert clip_grad_norm([p1], max_norm=1.0) == 0.0
