from __future__ import annotations

import math

import numpy as np
import pytest

from numgrad import Tensor
from numgrad.nn import Linear
from numgrad.optim import SGD, Adam, CosineAnnealingLR


def _linear_regression_data(
    seed: int, n: int = 200, d: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    true_w = rng.normal(size=(d, 1))
    true_b = rng.normal(size=(1,))
    y = X @ true_w + true_b
    return X, y


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sgd_converges_on_linear_regression(seed: int) -> None:
    X, y = _linear_regression_data(seed)
    rng = np.random.default_rng(seed)
    layer = Linear(X.shape[1], 1, rng=rng)
    opt = SGD(layer.parameters(), lr=0.1)
    X_t, y_t = Tensor(X), Tensor(y)

    first_loss = None
    loss = None
    for _ in range(300):
        opt.zero_grad()
        pred = layer(X_t)
        loss = ((pred - y_t) ** 2).mean()
        if first_loss is None:
            first_loss = loss.item()
        loss.backward()
        opt.step()

    assert loss is not None and first_loss is not None
    assert loss.item() < 1e-4
    assert loss.item() < first_loss


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sgd_with_momentum_converges_on_linear_regression(seed: int) -> None:
    X, y = _linear_regression_data(seed)
    rng = np.random.default_rng(seed)
    layer = Linear(X.shape[1], 1, rng=rng)
    opt = SGD(layer.parameters(), lr=0.05, momentum=0.9)
    X_t, y_t = Tensor(X), Tensor(y)

    loss = None
    for _ in range(300):
        opt.zero_grad()
        pred = layer(X_t)
        loss = ((pred - y_t) ** 2).mean()
        loss.backward()
        opt.step()

    assert loss is not None and loss.item() < 1e-3


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_adam_converges_on_linear_regression(seed: int) -> None:
    X, y = _linear_regression_data(seed)
    rng = np.random.default_rng(seed)
    layer = Linear(X.shape[1], 1, rng=rng)
    opt = Adam(layer.parameters(), lr=0.05)
    X_t, y_t = Tensor(X), Tensor(y)

    first_loss = None
    loss = None
    for _ in range(300):
        opt.zero_grad()
        pred = layer(X_t)
        loss = ((pred - y_t) ** 2).mean()
        if first_loss is None:
            first_loss = loss.item()
        loss.backward()
        opt.step()

    assert loss is not None and first_loss is not None
    assert loss.item() < 1e-6
    assert loss.item() < first_loss


def test_optimizer_zero_grad_clears_gradients() -> None:
    p = Tensor(np.zeros(2))
    p.grad = np.array([1.0, 2.0])
    opt = SGD([p], lr=0.1)
    opt.zero_grad()
    assert p.grad is None


def test_sgd_skips_parameters_with_no_gradient() -> None:
    p1 = Tensor(np.array([1.0]))
    p2 = Tensor(np.array([5.0]))
    p1.grad = np.array([1.0])
    # p2.grad left as None
    opt = SGD([p1, p2], lr=0.1)
    opt.step()
    assert p1.data[0] == pytest.approx(0.9)
    assert p2.data[0] == pytest.approx(5.0)  # untouched


# ---- Adam bias correction ----


def test_adam_bias_correction_first_two_steps_match_hand_computation() -> None:
    p = Tensor(np.array([0.0]))
    opt = Adam([p], lr=0.1, betas=(0.9, 0.999), eps=1e-8)

    p.grad = np.array([1.0])
    opt.step()
    # m1 = 0.1*1 = 0.1, v1 = 0.001*1 = 0.001
    # m_hat1 = 0.1 / (1 - 0.9) = 1.0, v_hat1 = 0.001 / (1 - 0.999) = 1.0
    expected_p1 = 0.0 - 0.1 * 1.0 / (np.sqrt(1.0) + 1e-8)
    assert p.data[0] == pytest.approx(expected_p1, rel=1e-6)

    p.grad = np.array([1.0])
    opt.step()
    # m2 = 0.9*0.1 + 0.1*1 = 0.19, v2 = 0.999*0.001 + 0.001*1 = 0.001999
    # m_hat2 = 0.19 / (1 - 0.81) = 1.0, v_hat2 = 0.001999 / (1 - 0.998001) = 1.0
    expected_update2 = 0.1 * 1.0 / (np.sqrt(1.0) + 1e-8)
    expected_p2 = expected_p1 - expected_update2
    assert p.data[0] == pytest.approx(expected_p2, rel=1e-6)


def test_adam_bias_correction_cancels_startup_bias_for_constant_gradient() -> None:
    """With m0=v0=0, m_t = (1-beta1^t)*g by induction for constant g, so
    m_hat_t = g (and v_hat_t = g^2) for every t -- bias correction exactly
    cancels the EMA startup transient, giving an identical update at every
    step. This is precisely the failure mode bias correction exists to fix:
    without it, the raw (uncorrected) m/v would be far too small early on.
    """
    p = Tensor(np.array([0.0]))
    opt = Adam([p], lr=0.1)
    g = 2.0
    updates = []
    prev = p.data[0]
    for _ in range(6):
        p.grad = np.array([g])
        opt.step()
        updates.append(prev - p.data[0])
        prev = p.data[0]

    for u in updates:
        assert u == pytest.approx(updates[0], rel=1e-6)
    # lr * g / (|g| + eps) ~= lr * sign(g)
    assert updates[0] == pytest.approx(0.1, rel=1e-3)


def test_adam_without_bias_correction_would_overstate_early_updates() -> None:
    # Demonstrates the actual failure mode: using the raw (uncorrected) m, v
    # instead of m_hat, v_hat at t=1 makes the first update over 3x too
    # large here, because v's bias dominates m's (beta2=0.999 is much closer
    # to 1 than beta1=0.9, so v is relatively more underestimated than m is,
    # and dividing by its too-small sqrt(v) inflates the step).
    beta1, beta2, eps, g = 0.9, 0.999, 1e-8, 2.0
    m1 = (1 - beta1) * g
    v1 = (1 - beta2) * g**2
    uncorrected_update = m1 / (np.sqrt(v1) + eps)
    corrected_update = (m1 / (1 - beta1)) / (np.sqrt(v1 / (1 - beta2)) + eps)
    assert abs(uncorrected_update) > 2.0 * abs(corrected_update)


# ---- CosineAnnealingLR ----


def test_cosine_annealing_lr_matches_closed_form() -> None:
    p = Tensor(np.array([0.0]))
    opt = SGD([p], lr=1.0)
    sched = CosineAnnealingLR(opt, t_max=10, eta_min=0.0)
    for t in range(1, 11):
        sched.step()
        expected = 0.5 * (1.0 + math.cos(math.pi * t / 10))
        assert opt.lr == pytest.approx(expected)


def test_cosine_annealing_lr_reaches_eta_min_at_t_max() -> None:
    p = Tensor(np.array([0.0]))
    opt = SGD([p], lr=1.0)
    sched = CosineAnnealingLR(opt, t_max=5, eta_min=0.01)
    for _ in range(5):
        sched.step()
    assert opt.lr == pytest.approx(0.01)


def test_cosine_annealing_lr_holds_at_eta_min_past_t_max() -> None:
    p = Tensor(np.array([0.0]))
    opt = SGD([p], lr=1.0)
    sched = CosineAnnealingLR(opt, t_max=3, eta_min=0.0)
    for _ in range(3):
        sched.step()
    sched.step()  # past t_max
    assert opt.lr == pytest.approx(0.0)
