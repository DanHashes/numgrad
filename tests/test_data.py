from __future__ import annotations

import numpy as np
import pytest

from numgrad.data import DataLoader, one_hot, train_test_split


def test_dataloader_batch_count_and_shapes() -> None:
    X = np.arange(100 * 4).reshape(100, 4)
    y = np.arange(100)
    dl = DataLoader(X, y, batch_size=32, shuffle=False)
    assert len(dl) == 4  # ceil(100/32)

    batches = list(dl)
    assert len(batches) == 4
    sizes = [len(bx) for bx, _ in batches]
    assert sizes == [32, 32, 32, 4]
    for bx, by in batches:
        assert bx.shape[1] == 4
        assert len(bx) == len(by)


def test_dataloader_no_shuffle_preserves_order() -> None:
    X = np.arange(10).reshape(10, 1)
    y = np.arange(10)
    dl = DataLoader(X, y, batch_size=10, shuffle=False)
    (bx, by), = list(dl)
    np.testing.assert_array_equal(bx.flatten(), np.arange(10))
    np.testing.assert_array_equal(by, np.arange(10))


def test_dataloader_shuffling_is_reproducible_with_same_seed() -> None:
    X = np.arange(20).reshape(20, 1)
    y = np.arange(20)
    dl1 = DataLoader(X, y, batch_size=20, shuffle=True, rng=np.random.default_rng(42))
    dl2 = DataLoader(X, y, batch_size=20, shuffle=True, rng=np.random.default_rng(42))

    (bx1, _), = list(dl1)
    (bx2, _), = list(dl2)
    np.testing.assert_array_equal(bx1, bx2)


def test_dataloader_shuffling_differs_across_epochs() -> None:
    X = np.arange(50).reshape(50, 1)
    y = np.arange(50)
    dl = DataLoader(X, y, batch_size=50, shuffle=True, rng=np.random.default_rng(0))
    (epoch1,), = [[bx for bx, _ in dl]]
    (epoch2,) = [bx for bx, _ in dl]
    assert not np.array_equal(epoch1, epoch2)


def test_dataloader_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        DataLoader(np.zeros((5, 2)), np.zeros(4), batch_size=2)


def test_train_test_split_sizes_and_no_overlap() -> None:
    X = np.arange(100).reshape(100, 1)
    y = np.arange(100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, rng=np.random.default_rng(0)
    )
    assert len(X_test) == 30
    assert len(X_train) == 70
    train_set = set(X_train.flatten().tolist())
    test_set = set(X_test.flatten().tolist())
    assert train_set.isdisjoint(test_set)
    assert train_set | test_set == set(range(100))


def test_train_test_split_keeps_x_y_aligned() -> None:
    X = np.arange(50).reshape(50, 1)
    y = np.arange(50) * 10
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, rng=np.random.default_rng(0)
    )
    np.testing.assert_array_equal(X_train.flatten() * 10, y_train)
    np.testing.assert_array_equal(X_test.flatten() * 10, y_test)


def test_train_test_split_reproducible_with_seed() -> None:
    X = np.arange(40).reshape(40, 1)
    y = np.arange(40)
    split1 = train_test_split(X, y, test_size=0.25, rng=np.random.default_rng(7))
    split2 = train_test_split(X, y, test_size=0.25, rng=np.random.default_rng(7))
    for a, b in zip(split1, split2, strict=True):
        np.testing.assert_array_equal(a, b)


def test_one_hot_basic_encoding() -> None:
    labels = np.array([0, 2, 1])
    encoded = one_hot(labels, num_classes=3)
    expected = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    np.testing.assert_array_equal(encoded, expected)


def test_one_hot_rows_sum_to_one() -> None:
    labels = np.array([0, 1, 2, 3, 4])
    encoded = one_hot(labels, num_classes=5)
    np.testing.assert_array_equal(encoded.sum(axis=1), np.ones(5))
