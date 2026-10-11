"""Naive 4-nested-loop convolution vs. im2col + matmul.

The naive version loops over (batch, output channel, output row, output col)
in Python and sums the kernel window by hand for each one. The im2col version
reduces the whole convolution to a single matmul, pushing the per-pixel work
down into NumPy's compiled, contiguous-memory, BLAS-backed inner loops. This
is the headline benchmark for the project's "why vectorize" story.

Run directly: python benchmarks/bench_conv.py
"""

from __future__ import annotations

import time

import numpy as np

from numgrad import Tensor
from numgrad.nn.functional import conv2d

CONFIGS = [
    # (N, C_in, H, W, C_out, kernel_size)
    (1, 1, 16, 16, 1, 3),
    (2, 3, 16, 16, 4, 3),
    (2, 3, 32, 32, 8, 3),
    (4, 8, 32, 32, 16, 3),
]
STRIDE = 1
PADDING = 1
REPEATS = 3


def naive_conv2d(
    x: np.ndarray, weight: np.ndarray, bias: np.ndarray, stride: int, padding: int
) -> np.ndarray:
    n, c_in, h, w = x.shape
    c_out, _, kh, kw = weight.shape
    out_h = (h + 2 * padding - kh) // stride + 1
    out_w = (w + 2 * padding - kw) // stride + 1
    x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    out = np.zeros((n, c_out, out_h, out_w))

    for ni in range(n):
        for co in range(c_out):
            for oh in range(out_h):
                for ow in range(out_w):
                    h0, w0 = oh * stride, ow * stride
                    patch = x_padded[ni, :, h0 : h0 + kh, w0 : w0 + kw]
                    out[ni, co, oh, ow] = np.sum(patch * weight[co]) + bias[co]
    return out


def vectorized_conv2d(
    x: np.ndarray, weight: np.ndarray, bias: np.ndarray, stride: int, padding: int
) -> np.ndarray:
    return conv2d(Tensor(x), Tensor(weight), Tensor(bias), stride, padding).data


def time_call(fn, *args, repeats: int = REPEATS) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - start)
    return best


def main() -> None:
    rng = np.random.default_rng(0)
    header = (
        f"{'config (N,Cin,H,W,Cout,k)':>28} | {'naive (s)':>12} | "
        f"{'im2col (s)':>12} | {'speedup':>10}"
    )
    print(header)
    print("-" * len(header))
    for n, c_in, h, w, c_out, k in CONFIGS:
        x = rng.uniform(-1.0, 1.0, size=(n, c_in, h, w))
        weight = rng.uniform(-1.0, 1.0, size=(c_out, c_in, k, k))
        bias = rng.uniform(-1.0, 1.0, size=(c_out,))

        naive_time = time_call(naive_conv2d, x, weight, bias, STRIDE, PADDING)
        vectorized_time = time_call(vectorized_conv2d, x, weight, bias, STRIDE, PADDING)
        speedup = naive_time / vectorized_time

        config_str = f"({n},{c_in},{h},{w},{c_out},{k})"
        print(f"{config_str:>28} | {naive_time:12.6f} | {vectorized_time:12.6f} | {speedup:9.1f}x")


if __name__ == "__main__":
    main()
