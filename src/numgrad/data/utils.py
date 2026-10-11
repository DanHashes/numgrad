from __future__ import annotations

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


def train_test_split(
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    test_size: float = 0.2,
    rng: np.random.Generator | None = None,
) -> tuple[Array, Array, Array, Array]:
    X = np.asarray(X)
    y = np.asarray(y)
    rng = rng if rng is not None else np.random.default_rng()

    n = len(X)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_test = int(round(n * test_size))
    test_idx, train_idx = indices[:n_test], indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def one_hot(labels: npt.ArrayLike, num_classes: int) -> Array:
    labels_arr = np.asarray(labels, dtype=np.int64)
    out = np.zeros((labels_arr.shape[0], num_classes))
    out[np.arange(labels_arr.shape[0]), labels_arr] = 1.0
    return out
