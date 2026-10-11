# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-10-10

Initial release: a complete, from-scratch reverse-mode autodiff engine and
neural network framework on top of NumPy, trained end to end on MNIST.

### Added

- `Tensor`: a NumPy-backed autodiff core with a computation graph, topological
  `.backward()`, and a `no_grad()` context manager.
- Core ops with broadcasting-aware gradients: `+ - * / neg pow @`, reductions
  (`sum`/`mean`/`max`/`min`), shape ops (`reshape`/`transpose`/slicing), and
  `exp`/`log`/`relu`/`sigmoid`/`tanh`.
- Numerically stable `log_softmax`/`cross_entropy` (log-sum-exp trick), plus
  `clip_grad_norm`.
- `nn`: `Module`, `Linear` (Kaiming/He init), `ReLU`/`Sigmoid`/`Tanh`/`GELU`,
  `Dropout` (inverted), `Sequential`, `Conv2d`/`MaxPool2d` (via `im2col`/
  `col2im`), `Flatten`.
- `optim`: `SGD` (with momentum), `Adam` (explicit bias correction),
  `CosineAnnealingLR`.
- `data`: `DataLoader`, `train_test_split`, `one_hot`, a real MNIST IDX
  downloader/parser, and a synthetic network-free fallback for CI.
- `scripts/train.py`: CLI training script for an MLP or CNN on MNIST, with
  checkpointing and training-curve plots.
- Finite-difference gradient checking (`tests/helpers.py::gradcheck`) used
  across the entire op/layer surface.
- Benchmarks: naive-vs-vectorized elementwise ops (`benchmarks/
  bench_naive_vs_vectorized.py`) and naive-vs-im2col convolution
  (`benchmarks/bench_conv.py`).
- `docs/math.md` (derivations) and `docs/profiling.md` (CPU/memory profiling
  findings and a real fix: `einsum` → batched `matmul` in `Conv2d`'s
  backward, ~26% faster per epoch).
- CI: separate `lint`/`typecheck`/`test` (Python 3.10–3.12 matrix) GitHub
  Actions jobs, network-free by default.

### Results

- MLP (784→256→128→10): **98.13%** MNIST test accuracy.
- CNN (conv→pool→conv→pool→64→10): **98.77%** MNIST test accuracy.
- 284 tests passing, 97% line coverage (100% outside the network-dependent
  MNIST download path).
