from __future__ import annotations

import numpy as np
import pytest
from helpers import gradcheck

from numgrad import Tensor
from numgrad.nn import GELU, Dropout, Linear, Module, ReLU, Sequential, Sigmoid, Tanh

SEEDS = [0, 1, 2]


def rand_tensor(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    low: float = -2.0,
    high: float = 2.0,
) -> Tensor:
    return Tensor(rng.uniform(low, high, size=shape), requires_grad=True)


# ---- activation gradient checks ----


@pytest.mark.parametrize("seed", SEEDS)
def test_relu_gradcheck(seed: int) -> None:
    rng = np.random.default_rng(seed)
    # keep away from the kink at 0, where the numeric/analytic subgradients differ
    a = rand_tensor(rng, (3, 4), low=0.5, high=2.0)
    b = rand_tensor(rng, (3, 4), low=-2.0, high=-0.5)
    gradcheck(lambda a: ReLU()(a), [a])
    gradcheck(lambda b: ReLU()(b), [b])


@pytest.mark.parametrize("seed", SEEDS)
def test_sigmoid_gradcheck(seed: int) -> None:
    rng = np.random.default_rng(seed)
    a = rand_tensor(rng, (3, 4))
    gradcheck(lambda a: Sigmoid()(a), [a])


@pytest.mark.parametrize("seed", SEEDS)
def test_tanh_gradcheck(seed: int) -> None:
    rng = np.random.default_rng(seed)
    a = rand_tensor(rng, (3, 4))
    gradcheck(lambda a: Tanh()(a), [a])


@pytest.mark.parametrize("seed", SEEDS)
def test_gelu_gradcheck(seed: int) -> None:
    rng = np.random.default_rng(seed)
    a = rand_tensor(rng, (3, 4))
    gradcheck(lambda a: GELU()(a), [a])


def test_sigmoid_stable_for_large_magnitude_inputs() -> None:
    x = Tensor(np.array([-1e4, 1e4]), requires_grad=True)
    out = Sigmoid()(x)
    assert np.all(np.isfinite(out.data))
    np.testing.assert_allclose(out.data, [0.0, 1.0], atol=1e-10)
    out.backward(np.ones(2))
    assert x.grad is not None and np.all(np.isfinite(x.grad))


# ---- Linear ----


def test_linear_output_shape_and_param_shapes() -> None:
    rng = np.random.default_rng(0)
    layer = Linear(4, 3, rng=rng)
    assert layer.weight.shape == (4, 3)
    assert layer.bias.shape == (3,)
    x = Tensor(rng.normal(size=(5, 4)))
    out = layer(x)
    assert out.shape == (5, 3)


def test_linear_bias_initialized_to_zero() -> None:
    layer = Linear(4, 3, rng=np.random.default_rng(0))
    np.testing.assert_allclose(layer.bias.data, np.zeros(3))


def test_linear_weight_std_matches_kaiming_formula() -> None:
    # Statistical check with a large fan_in/fan_out so the sample std is a
    # reliable estimate of the true std = sqrt(2 / fan_in).
    rng = np.random.default_rng(0)
    layer = Linear(10_000, 50, rng=rng)
    expected_std = np.sqrt(2.0 / 10_000)
    assert layer.weight.data.std() == pytest.approx(expected_std, rel=0.05)


@pytest.mark.parametrize("seed", SEEDS)
def test_linear_gradcheck(seed: int) -> None:
    rng = np.random.default_rng(seed)
    layer = Linear(4, 3, rng=rng)
    x = rand_tensor(rng, (5, 4))
    gradcheck(lambda x: layer(x), [x])
    gradcheck(lambda w: x @ w + layer.bias, [layer.weight])
    gradcheck(lambda b: x @ layer.weight + b, [layer.bias])


# ---- end-to-end Sequential gradcheck (catches graph-wiring bugs) ----


@pytest.mark.parametrize("seed", SEEDS)
def test_sequential_mlp_gradcheck(seed: int) -> None:
    rng = np.random.default_rng(seed)
    model = Sequential(Linear(4, 6, rng=rng), ReLU(), Linear(6, 3, rng=rng))
    # keep inputs away from the ReLU kink at 0, where the numeric/analytic
    # subgradients can legitimately disagree
    x = rand_tensor(rng, (5, 4), low=0.5, high=1.5)
    gradcheck(lambda x: model(x), [x])

    # model.parameters() returns the actual registered Tensor objects (not
    # copies), so gradcheck can perturb each one's .data in place and
    # model(x) will pick up the perturbation directly — this exercises
    # gradients flowing through the whole stack into every parameter, not
    # just into the input.
    for p in model.parameters():
        gradcheck(lambda p: model(x), [p])


@pytest.mark.parametrize("seed", SEEDS)
def test_deep_sequential_gradcheck(seed: int) -> None:
    rng = np.random.default_rng(seed)
    model = Sequential(
        Linear(3, 5, rng=rng),
        Tanh(),
        Linear(5, 4, rng=rng),
        Sigmoid(),
        Linear(4, 2, rng=rng),
    )
    x = rand_tensor(rng, (4, 3))
    gradcheck(lambda x: model(x), [x])


# ---- Dropout ----


def test_dropout_eval_mode_is_identity() -> None:
    d = Dropout(p=0.5, rng=np.random.default_rng(0))
    d.eval()
    x = Tensor(np.ones(1000), requires_grad=True)
    out = d(x)
    np.testing.assert_array_equal(out.data, x.data)


def test_dropout_train_mode_zeros_elements_statistically() -> None:
    d = Dropout(p=0.3, rng=np.random.default_rng(0))
    d.train()
    x = Tensor(np.ones(10_000))
    out = d(x)
    zero_fraction = float((out.data == 0).mean())
    assert zero_fraction == pytest.approx(0.3, abs=0.02)
    # inverted dropout: surviving elements are scaled by 1 / keep_prob
    surviving = out.data[out.data != 0]
    np.testing.assert_allclose(surviving, 1.0 / 0.7)


def test_dropout_p_zero_is_identity_even_in_train_mode() -> None:
    d = Dropout(p=0.0)
    d.train()
    x = Tensor(np.arange(10, dtype=np.float64))
    out = d(x)
    np.testing.assert_array_equal(out.data, x.data)


def test_dropout_invalid_p_raises() -> None:
    with pytest.raises(ValueError):
        Dropout(p=1.0)
    with pytest.raises(ValueError):
        Dropout(p=-0.1)


def test_dropout_backward_matches_forward_mask() -> None:
    rng = np.random.default_rng(0)
    d = Dropout(p=0.5, rng=rng)
    d.train()
    x = Tensor(np.ones(20), requires_grad=True)
    out = d(x)
    out.backward(np.ones(20))
    assert x.grad is not None
    # Wherever the forward output was zeroed, the gradient must be zero too,
    # and wherever it survived, gradient == forward scale (1/keep_prob).
    np.testing.assert_allclose(x.grad, out.data)


# ---- Module / parameters() / zero_grad() / train() / eval() ----


def test_module_parameters_nested_sequential_count_and_shapes() -> None:
    rng = np.random.default_rng(0)
    model = Sequential(
        Sequential(Linear(4, 6, rng=rng), ReLU()),
        Sequential(Linear(6, 2, rng=rng)),
    )
    params = model.parameters()
    assert len(params) == 4  # 2 Linear layers x (weight, bias)
    shapes = {tuple(p.shape) for p in params}
    assert shapes == {(4, 6), (6,), (6, 2), (2,)}


def test_module_zero_grad_clears_all_nested_parameters() -> None:
    rng = np.random.default_rng(0)
    model = Sequential(Linear(3, 4, rng=rng), ReLU(), Linear(4, 2, rng=rng))
    x = Tensor(rng.normal(size=(2, 3)))
    model(x).sum().backward()
    assert all(p.grad is not None for p in model.parameters())
    model.zero_grad()
    assert all(p.grad is None for p in model.parameters())


def test_module_train_eval_propagates_to_nested_submodules() -> None:
    rng = np.random.default_rng(0)
    d1 = Dropout(p=0.5)
    d2 = Dropout(p=0.5)
    model = Sequential(Linear(3, 3, rng=rng), d1, Sequential(d2))

    model.eval()
    assert d1.training is False
    assert d2.training is False

    model.train()
    assert d1.training is True
    assert d2.training is True


def test_module_without_params_has_empty_parameter_list() -> None:
    assert ReLU().parameters() == []


def test_custom_module_subclass_registers_tensor_and_submodule_attributes() -> None:
    class Tiny(Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = Tensor(np.ones(3), requires_grad=True)
            self.not_a_param = Tensor(np.ones(3), requires_grad=False)
            self.inner = Linear(3, 2, rng=np.random.default_rng(0))

        def forward(self, x: Tensor) -> Tensor:
            return self.inner(x * self.w)

    m = Tiny()
    params = m.parameters()
    assert len(params) == 3  # w, inner.weight, inner.bias
    assert any(p is m.w for p in params)
    assert not any(p is m.not_a_param for p in params)
