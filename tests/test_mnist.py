from __future__ import annotations

import gzip
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from numgrad.data import DataLoader, make_synthetic_mnist, train_test_split
from numgrad.data.mnist import (
    _parse_idx_images,
    _parse_idx_labels,
    flatten_images,
    load_mnist,
)

# ---- IDX parser (no network needed: builds tiny fake IDX files by hand) ----


def _write_idx_images(path: Path, images: np.ndarray) -> None:
    n, rows, cols = images.shape
    header = struct.pack(">IIII", 2051, n, rows, cols)
    with gzip.open(path, "wb") as f:
        f.write(header + images.astype(np.uint8).tobytes())


def _write_idx_labels(path: Path, labels: np.ndarray) -> None:
    header = struct.pack(">II", 2049, len(labels))
    with gzip.open(path, "wb") as f:
        f.write(header + labels.astype(np.uint8).tobytes())


def test_parse_idx_images_roundtrip(tmp_path: Path) -> None:
    images = np.array(
        [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[9, 8, 7], [6, 5, 4], [3, 2, 1]]], dtype=np.uint8
    )
    path = tmp_path / "images.gz"
    _write_idx_images(path, images)
    parsed = _parse_idx_images(path)
    np.testing.assert_array_equal(parsed, images)


def test_parse_idx_labels_roundtrip(tmp_path: Path) -> None:
    labels = np.array([3, 7, 0, 9], dtype=np.uint8)
    path = tmp_path / "labels.gz"
    _write_idx_labels(path, labels)
    parsed = _parse_idx_labels(path)
    np.testing.assert_array_equal(parsed, labels)


def test_parse_idx_images_rejects_bad_magic_number(tmp_path: Path) -> None:
    path = tmp_path / "bad.gz"
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">IIII", 1234, 1, 2, 2) + bytes(4))
    with pytest.raises(ValueError, match="magic number"):
        _parse_idx_images(path)


# ---- synthetic fallback (CI-safe, network-free) ----


def test_synthetic_mnist_shapes_dtypes_and_ranges() -> None:
    X_train, y_train, X_test, y_test = make_synthetic_mnist(
        n_train=40, n_test=10, rng=np.random.default_rng(0)
    )
    assert X_train.shape == (40, 1, 28, 28)
    assert X_test.shape == (10, 1, 28, 28)
    assert y_train.shape == (40,)
    assert y_test.shape == (10,)
    assert X_train.dtype == np.float64
    assert y_train.dtype == np.int64
    assert X_train.min() >= 0.0
    assert X_train.max() <= 1.0
    assert set(np.unique(y_train).tolist()) <= set(range(10))


def test_synthetic_mnist_reproducible_with_seed() -> None:
    a = make_synthetic_mnist(n_train=20, n_test=5, rng=np.random.default_rng(1))
    b = make_synthetic_mnist(n_train=20, n_test=5, rng=np.random.default_rng(1))
    for arr_a, arr_b in zip(a, b, strict=True):
        np.testing.assert_array_equal(arr_a, arr_b)


def test_synthetic_mnist_differs_across_seeds() -> None:
    a = make_synthetic_mnist(n_train=20, n_test=5, rng=np.random.default_rng(1))
    b = make_synthetic_mnist(n_train=20, n_test=5, rng=np.random.default_rng(2))
    assert not np.array_equal(a[0], b[0])


def test_synthetic_mnist_has_learnable_per_class_structure() -> None:
    # A naive nearest-centroid classifier should do far better than chance
    # (10%), confirming the data has real per-class signal, not just noise.
    X_train, y_train, X_test, y_test = make_synthetic_mnist(
        n_train=300, n_test=100, num_classes=10, rng=np.random.default_rng(0)
    )
    centroids = np.stack([X_train[y_train == c].mean(axis=0) for c in range(10)])
    dists = ((X_test[:, None] - centroids[None, :]) ** 2).sum(axis=(2, 3, 4))
    preds = np.argmin(dists, axis=1)
    accuracy = float((preds == y_test).mean())
    assert accuracy > 0.5


def test_flatten_images_shape() -> None:
    X = np.zeros((5, 1, 28, 28))
    flat = flatten_images(X)
    assert flat.shape == (5, 784)
    np.testing.assert_array_equal(flat.reshape(5, 1, 28, 28), X)


def test_dataloader_works_with_synthetic_mnist() -> None:
    rng = np.random.default_rng(0)
    X_train, y_train, _, _ = make_synthetic_mnist(n_train=64, n_test=16, rng=rng)
    dl = DataLoader(X_train, y_train, batch_size=16, shuffle=True, rng=np.random.default_rng(1))
    batches = list(dl)
    assert len(batches) == 4
    for bx, by in batches:
        assert bx.shape[1:] == (1, 28, 28)
        assert len(bx) == len(by)


def test_train_test_split_works_with_synthetic_mnist() -> None:
    rng = np.random.default_rng(0)
    X_train, y_train, _, _ = make_synthetic_mnist(n_train=64, n_test=16, rng=rng)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_train, y_train, test_size=0.25, rng=np.random.default_rng(2)
    )
    assert len(X_te) == 16
    assert len(X_tr) == 48
    assert X_tr.shape[1:] == (1, 28, 28)


# ---- real MNIST download: network access, skipped in CI by default ----


@pytest.mark.skipif(
    os.environ.get("NUMGRAD_RUN_NETWORK_TESTS") != "1",
    reason="downloads real MNIST from the network; set NUMGRAD_RUN_NETWORK_TESTS=1 to run manually",
)
def test_real_mnist_shapes_and_label_counts() -> None:
    X_train, y_train, X_test, y_test = load_mnist()
    assert X_train.shape == (60000, 1, 28, 28)
    assert y_train.shape == (60000,)
    assert X_test.shape == (10000, 1, 28, 28)
    assert y_test.shape == (10000,)
    assert X_train.dtype == np.float64
    assert X_train.min() >= 0.0 and X_train.max() <= 1.0
    assert set(np.unique(y_train).tolist()) == set(range(10))
    assert set(np.unique(y_test).tolist()) == set(range(10))
