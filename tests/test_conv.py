from __future__ import annotations

import numpy as np
import pytest
from helpers import gradcheck

from numgrad import Tensor
from numgrad.nn import Conv2d, Flatten, MaxPool2d
from numgrad.nn.functional import col2im, conv2d, im2col, max_pool2d

SEEDS = [0, 1, 2]


def rand_tensor(
    rng: np.random.Generator, shape: tuple[int, ...], low: float = -1.0, high: float = 1.0
) -> Tensor:
    return Tensor(rng.uniform(low, high, size=shape), requires_grad=True)


# ---- im2col / col2im shape sanity ----


@pytest.mark.parametrize(
    "shape,kh,kw,stride,padding,expected_out",
    [
        ((2, 3, 8, 8), 3, 3, 1, 1, (8, 8)),
        ((1, 1, 5, 5), 3, 3, 2, 0, (2, 2)),
        ((2, 4, 6, 6), 2, 2, 2, 0, (3, 3)),
    ],
)
def test_im2col_shape(
    shape: tuple[int, int, int, int],
    kh: int,
    kw: int,
    stride: int,
    padding: int,
    expected_out: tuple[int, int],
) -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=shape)
    cols = im2col(x, kh, kw, stride, padding)
    n, c = shape[0], shape[1]
    out_h, out_w = expected_out
    assert cols.shape == (n, c * kh * kw, out_h * out_w)


def test_col2im_is_left_inverse_on_non_overlapping_patches() -> None:
    # With stride == kernel size, patches don't overlap, so im2col -> col2im
    # should exactly reconstruct the original (no padding to crop either).
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 3, 6, 6))
    cols = im2col(x, 2, 2, 2, 0)
    reconstructed = col2im(cols, x.shape, 2, 2, 2, 0)
    np.testing.assert_allclose(reconstructed, x)


def test_conv2d_output_matches_hand_computed_sums() -> None:
    # 1x1x4x4 input, a single 2x2 all-ones kernel, stride 2, no padding:
    # each output element is just the sum of its 2x2 block.
    x = np.arange(16, dtype=np.float64).reshape(1, 1, 4, 4)
    weight = Tensor(np.ones((1, 1, 2, 2)))
    bias = Tensor(np.zeros(1))
    out = conv2d(Tensor(x), weight, bias, stride=2, padding=0)
    expected = np.array([[[[10.0, 18.0], [42.0, 50.0]]]])
    np.testing.assert_allclose(out.data, expected)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(
    "in_shape,out_channels,kernel_size,stride,padding",
    [
        ((2, 2, 5, 5), 3, 3, 1, 1),
        ((1, 3, 6, 6), 2, 3, 2, 0),
        ((2, 1, 4, 4), 4, 2, 2, 1),
    ],
)
def test_conv2d_gradcheck(
    seed: int,
    in_shape: tuple[int, int, int, int],
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
) -> None:
    rng = np.random.default_rng(seed)
    in_channels = in_shape[1]
    conv = Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, rng=rng)
    x = rand_tensor(rng, in_shape)

    gradcheck(lambda x: conv(x), [x])
    gradcheck(lambda w: conv2d(x, w, conv.bias, stride, padding), [conv.weight])
    gradcheck(lambda b: conv2d(x, conv.weight, b, stride, padding), [conv.bias])


@pytest.mark.parametrize("seed", SEEDS)
def test_maxpool2d_gradcheck(seed: int) -> None:
    rng = np.random.default_rng(seed)
    # jitter inputs so exact ties between pooled elements are vanishingly
    # unlikely (ties are a separate, deterministic test below)
    x = rand_tensor(rng, (2, 3, 6, 6))
    gradcheck(lambda x: max_pool2d(x, kernel_size=2, stride=2), [x])


def test_maxpool2d_matches_hand_computed_max() -> None:
    x = np.array(
        [
            [
                [
                    [1.0, 2.0, 5.0, 6.0],
                    [3.0, 4.0, 7.0, 8.0],
                    [9.0, 10.0, 13.0, 14.0],
                    [11.0, 12.0, 15.0, 16.0],
                ]
            ]
        ]
    )
    out = max_pool2d(Tensor(x), kernel_size=2, stride=2)
    expected = np.array([[[[4.0, 8.0], [12.0, 16.0]]]])
    np.testing.assert_allclose(out.data, expected)


def test_maxpool2d_routes_gradient_only_to_argmax() -> None:
    x = Tensor(
        np.array([[[[1.0, 2.0], [3.0, 4.0]]]]), requires_grad=True
    )  # single 2x2 window, max is 4.0 at (1,1)
    out = max_pool2d(x, kernel_size=2, stride=2)
    out.backward(np.ones((1, 1, 1, 1)))
    assert x.grad is not None
    np.testing.assert_allclose(x.grad, [[[[0.0, 0.0], [0.0, 1.0]]]])


def test_maxpool2d_tie_goes_to_first_occurrence() -> None:
    x = Tensor(np.array([[[[5.0, 5.0], [1.0, 2.0]]]]), requires_grad=True)
    out = max_pool2d(x, kernel_size=2, stride=2)
    out.backward(np.ones((1, 1, 1, 1)))
    assert x.grad is not None
    np.testing.assert_allclose(x.grad, [[[[1.0, 0.0], [0.0, 0.0]]]])


def test_flatten_shape_and_gradcheck() -> None:
    rng = np.random.default_rng(0)
    x = rand_tensor(rng, (3, 2, 4, 4))
    flat = Flatten()
    out = flat(x)
    assert out.shape == (3, 32)
    gradcheck(lambda x: flat(x), [x])


# ---- end-to-end CNN block gradcheck ----


@pytest.mark.parametrize("seed", SEEDS)
def test_conv_pool_flatten_linear_end_to_end_gradcheck(seed: int) -> None:
    from numgrad.nn import Linear, Sequential

    rng = np.random.default_rng(seed)
    model = Sequential(
        Conv2d(1, 2, 3, stride=1, padding=1, rng=rng),
        MaxPool2d(2, 2),
        Flatten(),
        Linear(2 * 4 * 4, 3, rng=rng),
    )
    x = rand_tensor(rng, (2, 1, 8, 8), low=0.2, high=1.0)
    gradcheck(lambda x: model(x), [x])
    for p in model.parameters():
        gradcheck(lambda p: model(x), [p])
