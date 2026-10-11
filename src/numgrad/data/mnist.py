"""MNIST loading: a real IDX downloader/parser (stdlib + NumPy only, no
torchvision/sklearn) plus a synthetic, network-free fallback shaped
identically to real MNIST so CI never depends on network access.
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

# The original http://yann.lecun.com/exdb/mnist/ has been unreliable for
# programmatic downloads for years; this is the mirror torchvision itself
# has defaulted to since 2021.
_DEFAULT_BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

_IMAGE_MAGIC = 2051
_LABEL_MAGIC = 2049


def _ensure_cached(name: str, cache_dir: Path, base_url: str) -> Path:
    dest = cache_dir / _FILES[name]
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(base_url + _FILES[name], dest)  # noqa: S310
    return dest


def _parse_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != _IMAGE_MAGIC:
            raise ValueError(f"unexpected magic number {magic} for images file {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols)


def _parse_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != _LABEL_MAGIC:
            raise ValueError(f"unexpected magic number {magic} for labels file {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def _normalize_images(images: np.ndarray) -> Array:
    n, h, w = images.shape
    return (images.astype(np.float64) / 255.0).reshape(n, 1, h, w)


def load_mnist(
    cache_dir: str | Path = "~/.numgrad/mnist",
    base_url: str = _DEFAULT_BASE_URL,
) -> tuple[Array, np.ndarray, Array, np.ndarray]:
    """Download (if not already cached) and parse real MNIST.

    Returns (X_train, y_train, X_test, y_test): images as float64 in [0, 1]
    with shape (N, 1, 28, 28); labels as int64, shape (N,).
    """
    cache_path = Path(cache_dir).expanduser()
    train_images_path = _ensure_cached("train_images", cache_path, base_url)
    train_labels_path = _ensure_cached("train_labels", cache_path, base_url)
    test_images_path = _ensure_cached("test_images", cache_path, base_url)
    test_labels_path = _ensure_cached("test_labels", cache_path, base_url)

    X_train = _normalize_images(_parse_idx_images(train_images_path))
    y_train = _parse_idx_labels(train_labels_path).astype(np.int64)
    X_test = _normalize_images(_parse_idx_images(test_images_path))
    y_test = _parse_idx_labels(test_labels_path).astype(np.int64)
    return X_train, y_train, X_test, y_test


def flatten_images(X: Array) -> Array:
    """(N, 1, H, W) -> (N, H*W), for the MLP path."""
    return X.reshape(X.shape[0], -1)


def make_synthetic_mnist(
    n_train: int = 200,
    n_test: int = 50,
    num_classes: int = 10,
    image_size: int = 28,
    noise_std: float = 0.15,
    rng: np.random.Generator | None = None,
) -> tuple[Array, np.ndarray, Array, np.ndarray]:
    """A small, network-free stand-in for MNIST.

    Each class gets one fixed random "template" image; every sample of that
    class is the template plus Gaussian noise, clipped to [0, 1] — so the
    data is structured (there's a real, learnable per-class signal) rather
    than pure noise, while being network-free and shaped identically to real
    MNIST: (N, 1, 28, 28) float64 in [0, 1], int64 labels in [0, num_classes).
    Used so CI (and quick local smoke tests) never depend on network access.
    """
    rng = rng if rng is not None else np.random.default_rng()
    templates = rng.uniform(0.0, 1.0, size=(num_classes, 1, image_size, image_size))

    def _make_split(n: int) -> tuple[Array, np.ndarray]:
        labels = rng.integers(0, num_classes, size=n)
        images = templates[labels] + rng.normal(
            0.0, noise_std, size=(n, 1, image_size, image_size)
        )
        images = np.clip(images, 0.0, 1.0)
        return images, labels.astype(np.int64)

    X_train, y_train = _make_split(n_train)
    X_test, y_test = _make_split(n_test)
    return X_train, y_train, X_test, y_test
