# Progress

- [x] Step 1 — Repository scaffolding
- [x] Step 2 — Core `Tensor` and the autodiff engine
- [ ] Step 3 — Extended ops and numerical stability
- [ ] Step 4 — Neural network layers (fully connected)
- [ ] Step 5 — Convolution and the vectorization case study
- [ ] Step 6 — Optimizers and training utilities
- [ ] Step 7 — MNIST data pipeline
- [ ] Step 8 — Capstone: train on MNIST
- [ ] Step 9 — Test hardening, profiling, and CI finalization
- [ ] Step 10 — Documentation and release

## Step summaries

### Step 1 — Repository scaffolding

Set up the `src/` layout package skeleton (`numgrad` with `tensor/`, `nn/`, `optim/`, `data/` subpackages), `pyproject.toml` (hatchling build backend, PEP 621 metadata, Python 3.10+ floor, `numpy` runtime dependency, dev extras for `pytest`/`pytest-cov`/`ruff`/`mypy`/`matplotlib`), ruff and mypy (`strict = true`) configuration, MIT license, `.gitignore`, and a minimal GitHub Actions CI workflow that installs the package and runs `pytest`. Chose `hatchling` as the build backend for its simple `src/` layout support without extra config.

### Step 2 — Core `Tensor` and the autodiff engine

Built the `Tensor` class (`src/numgrad/tensor/tensor.py`) wrapping a `float64` NumPy array with `.grad`, `.requires_grad`, and graph bookkeeping (`_prev` parents, a `_backward` function), plus operator overloads for `+ - * / neg pow @` and a `no_grad()` context manager (`grad_mode.py`) that skips graph-building entirely when active. Every op routes through a shared `_make()` helper that only attaches parents/backward when at least one input requires grad, and every broadcasting op routes through a single `_unbroadcast(grad, target_shape)` helper rather than duplicating shape-reduction logic per operator.

Design decision worth flagging: the first implementation had `_backward` closures read the output node's `.grad` directly, which is the natural micrograd-style approach — but it silently produces wrong (compounding) gradients if `.backward()` is called more than once on the same graph, because the output's `.grad` from the previous call bleeds into the next call's seed. Fixed by making `_backward(grad_output)` a pure function of the incoming gradient that returns `(parent, contribution)` pairs; `backward()` now relays gradients call-locally through a temporary dict and only writes the final per-call total into each node's persistent `.grad` once. A test (`test_grad_accumulates_across_multiple_backward_calls`) catches a regression here.

Gradient checking: `tests/helpers.py` implements a finite-difference `gradcheck(f, inputs)` using the standard relative-error formula, seeding `backward()` with a ones-tensor (so it doesn't need `Tensor.sum()`, which doesn't exist until Step 3). All 94 tests pass, covering every op across multiple random seeds and broadcasting shapes (scalar+matrix, vector+matrix, mismatched batch dims for both elementwise ops and `matmul`).
