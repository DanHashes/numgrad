# Progress

- [x] Step 1 — Repository scaffolding
- [ ] Step 2 — Core `Tensor` and the autodiff engine
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
