"""Smoke-tests scripts/train.py end to end, using the synthetic dataset so it
stays fast and network-free in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from train import (  # noqa: E402
    build_cnn,
    build_mlp,
    load_checkpoint,
    parse_args,
    save_checkpoint,
    train,
)


def _args(**overrides: object) -> object:
    argv_backup = sys.argv
    sys.argv = ["train.py"]
    try:
        args = parse_args()
    finally:
        sys.argv = argv_backup
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.mark.parametrize("model_type", ["mlp", "cnn"])
def test_train_runs_end_to_end_on_synthetic_data(model_type: str, tmp_path: Path) -> None:
    import train as train_module

    train_module.CHECKPOINT_DIR = tmp_path / "checkpoints"
    train_module.ASSETS_DIR = tmp_path / "assets"

    args = _args(model=model_type, epochs=2, synthetic=True, batch_size=64)
    result = train(args)

    assert "history" in result
    history = result["history"]
    assert len(history["train_loss"]) == 2
    assert all(np.isfinite(v) for v in history["train_loss"])
    assert all(np.isfinite(v) for v in history["val_loss"])
    assert 0.0 <= result["test_acc"] <= 1.0

    ckpt_path = train_module.CHECKPOINT_DIR / f"{model_type}.npz"
    assert ckpt_path.exists()
    plot_path = train_module.ASSETS_DIR / f"training_curves_{model_type}.png"
    assert plot_path.exists()


def test_checkpoint_roundtrip_reproduces_predictions(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    model = build_mlp(rng, dropout=0.0)
    model.eval()

    from numgrad import Tensor

    x = Tensor(rng.normal(size=(4, 1, 28, 28)))
    original_output = model(x).data.copy()

    ckpt_path = tmp_path / "model.npz"
    save_checkpoint(model, ckpt_path)
    reloaded = load_checkpoint("mlp", ckpt_path)
    reloaded_output = reloaded(x).data

    np.testing.assert_allclose(original_output, reloaded_output)


def test_build_mlp_and_cnn_produce_expected_output_shape() -> None:
    from numgrad import Tensor

    rng = np.random.default_rng(0)
    x = Tensor(rng.normal(size=(3, 1, 28, 28)))

    mlp = build_mlp(rng, dropout=0.1)
    assert mlp(x).shape == (3, 10)

    cnn = build_cnn(rng, dropout=0.1)
    assert cnn(x).shape == (3, 10)
