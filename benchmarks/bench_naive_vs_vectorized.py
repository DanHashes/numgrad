"""Naive pure-Python vs. vectorized (NumPy-backed Tensor) elementwise multiply-and-sum.

This is the first data point in the project's "why vectorize" story: NumPy
pushes the elementwise multiply and reduction down into contiguous-memory,
compiled C loops (with BLAS/SIMD where applicable), while pure Python pays
per-element interpreter overhead (bytecode dispatch, boxed float objects,
attribute/index lookups) on every single multiply-add.

Run directly: python benchmarks/bench_naive_vs_vectorized.py
"""

from __future__ import annotations

import time

import numpy as np

from numgrad import Tensor

SIZES = [(32, 32), (128, 128), (512, 512), (1024, 1024)]
REPEATS = 5


def naive_multiply_and_sum(a: list[list[float]], b: list[list[float]]) -> float:
    total = 0.0
    for i in range(len(a)):
        row_a = a[i]
        row_b = b[i]
        for j in range(len(row_a)):
            total += row_a[j] * row_b[j]
    return total


def vectorized_multiply_and_sum(a: np.ndarray, b: np.ndarray) -> float:
    return (Tensor(a) * Tensor(b)).sum().item()


def time_call(fn, *args, repeats: int = REPEATS) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - start)
    return best


def main() -> None:
    rng = np.random.default_rng(0)
    print(f"{'shape':>14} | {'naive (s)':>12} | {'vectorized (s)':>15} | {'speedup':>10}")
    print("-" * 60)
    for shape in SIZES:
        a_np = rng.uniform(-1.0, 1.0, size=shape)
        b_np = rng.uniform(-1.0, 1.0, size=shape)
        a_list = a_np.tolist()
        b_list = b_np.tolist()

        naive_time = time_call(naive_multiply_and_sum, a_list, b_list)
        vectorized_time = time_call(vectorized_multiply_and_sum, a_np, b_np)
        speedup = naive_time / vectorized_time

        print(
            f"{str(shape):>14} | {naive_time:12.6f} | {vectorized_time:15.6f} | {speedup:9.1f}x"
        )


if __name__ == "__main__":
    main()
