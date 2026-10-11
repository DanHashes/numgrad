"""Train an MLP or CNN on MNIST (or the network-free synthetic fallback) using
nothing but numgrad.

Usage:
    python scripts/train.py --model mlp --epochs 12 --optimizer adam --lr 1e-3
    python scripts/train.py --model cnn --epochs 5 --optimizer adam --lr 1e-3
    python scripts/train.py --model mlp --synthetic --epochs 3   # quick, no download
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from numgrad import Tensor, clip_grad_norm, cross_entropy, no_grad
from numgrad.data import DataLoader, load_mnist, make_synthetic_mnist, train_test_split
from numgrad.nn import Conv2d, Dropout, Flatten, Linear, MaxPool2d, Module, ReLU, Sequential
from numgrad.optim import SGD, Adam, Optimizer

Array = npt.NDArray[np.float64]

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
ASSETS_DIR = REPO_ROOT / "docs" / "assets"


def build_mlp(rng: np.random.Generator, dropout: float) -> Sequential:
    return Sequential(
        Flatten(),
        Linear(784, 256, rng=rng),
        ReLU(),
        Dropout(dropout, rng=rng),
        Linear(256, 128, rng=rng),
        ReLU(),
        Linear(128, 10, rng=rng),
    )


def build_cnn(rng: np.random.Generator, dropout: float) -> Sequential:
    return Sequential(
        Conv2d(1, 8, 3, stride=1, padding=1, rng=rng),
        ReLU(),
        MaxPool2d(2, 2),
        Conv2d(8, 16, 3, stride=1, padding=1, rng=rng),
        ReLU(),
        MaxPool2d(2, 2),
        Flatten(),
        Linear(16 * 7 * 7, 64, rng=rng),
        ReLU(),
        Dropout(dropout, rng=rng),
        Linear(64, 10, rng=rng),
    )


def build_model(model_type: str, rng: np.random.Generator, dropout: float) -> Sequential:
    if model_type == "mlp":
        return build_mlp(rng, dropout)
    if model_type == "cnn":
        return build_cnn(rng, dropout)
    raise ValueError(f"unknown model type {model_type!r}")


def save_checkpoint(model: Module, path: Path) -> None:
    params = {f"param_{i}": p.data for i, p in enumerate(model.parameters())}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **params)


def load_checkpoint(model_type: str, path: Path, dropout: float = 0.0) -> Sequential:
    """Reconstruct a model of the given architecture and load trained weights
    from a checkpoint saved by save_checkpoint().
    """
    model = build_model(model_type, np.random.default_rng(0), dropout)
    data = np.load(path)
    for i, p in enumerate(model.parameters()):
        p.data = data[f"param_{i}"]
    model.eval()
    return model


def evaluate(
    model: Module, X: Array, y: npt.NDArray[np.int64], batch_size: int
) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_correct = 0
    n = len(X)
    with no_grad():
        for start in range(0, n, batch_size):
            xb = Tensor(X[start : start + batch_size])
            yb = y[start : start + batch_size]
            logits = model(xb)
            loss = cross_entropy(logits, yb)
            total_loss += loss.item() * len(yb)
            total_correct += int((np.argmax(logits.data, axis=-1) == yb).sum())
    model.train(was_training)
    return total_loss / n, total_correct / n


def plot_training_curves(history: dict[str, list[float]], path: Path, model_name: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="val")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title(f"{model_name.upper()} loss")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"], label="val")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.set_title(f"{model_name.upper()} accuracy")
    ax2.legend()

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def build_optimizer(model: Module, args: argparse.Namespace) -> Optimizer:
    if args.optimizer == "sgd":
        return SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    return Adam(model.parameters(), lr=args.lr)


def train(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)

    if args.synthetic:
        X_train_full, y_train_full, X_test, y_test = make_synthetic_mnist(
            n_train=2000, n_test=400, rng=rng
        )
    else:
        X_train_full, y_train_full, X_test, y_test = load_mnist()

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, rng=rng
    )

    model = build_model(args.model, rng, args.dropout)
    optimizer = build_optimizer(model, args)
    loader = DataLoader(X_train, y_train, batch_size=args.batch_size, shuffle=True, rng=rng)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        running_loss = 0.0
        running_correct = 0
        n_seen = 0

        for xb_np, yb_np in loader:
            optimizer.zero_grad()
            xb = Tensor(xb_np)
            logits = model(xb)
            loss = cross_entropy(logits, yb_np)
            loss.backward()
            clip_grad_norm(model.parameters(), args.clip_grad_norm)
            optimizer.step()

            running_loss += loss.item() * len(yb_np)
            running_correct += int((np.argmax(logits.data, axis=-1) == yb_np).sum())
            n_seen += len(yb_np)

        train_loss = running_loss / n_seen
        train_acc = running_correct / n_seen
        val_loss, val_acc = evaluate(model, X_val, y_val, args.batch_size)
        epoch_time = time.perf_counter() - epoch_start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"epoch {epoch:2d}/{args.epochs} | "
            f"train_loss {train_loss:.4f} train_acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} val_acc {val_acc:.4f} | "
            f"{epoch_time:.1f}s"
        )

    test_loss, test_acc = evaluate(model, X_test, y_test, args.batch_size)
    print(f"final: test_loss {test_loss:.4f} test_acc {test_acc:.4f}")

    ckpt_path = CHECKPOINT_DIR / f"{args.model}.npz"
    save_checkpoint(model, ckpt_path)
    print(f"saved checkpoint to {ckpt_path}")

    plot_path = ASSETS_DIR / f"training_curves_{args.model}.png"
    plot_training_curves(history, plot_path, args.model)
    print(f"saved training curves to {plot_path}")

    return {"history": history, "test_loss": test_loss, "test_acc": test_acc}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MLP or CNN on MNIST with numgrad.")
    parser.add_argument("--model", choices=["mlp", "cnn"], default="mlp")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--clip-grad-norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="use the network-free synthetic dataset instead of real MNIST",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
