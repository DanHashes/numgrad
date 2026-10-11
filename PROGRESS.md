# Progress

- [x] Step 1 — Repository scaffolding
- [x] Step 2 — Core `Tensor` and the autodiff engine
- [x] Step 3 — Extended ops and numerical stability
- [x] Step 4 — Neural network layers (fully connected)
- [x] Step 5 — Convolution and the vectorization case study
- [x] Step 6 — Optimizers and training utilities
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

### Step 3 — Extended ops and numerical stability

Added reductions (`sum`, `mean`, `max`, `min`, all with `axis`/`keepdims`), shape ops (`reshape`, `transpose`, `.T`, `__getitem__` slicing via `np.add.at` for correct gradient scatter), and transcendental ops (`exp`, `log`) to `Tensor`. `max`/`min` route gradient only to the argmax/argmin element, splitting evenly across ties (tested explicitly). All reductions share one `_restore_reduced_dims` helper that re-inserts collapsed axes before broadcasting the incoming gradient back to the input shape, mirroring the `_unbroadcast` pattern from Step 2.

Built `log_softmax`/`cross_entropy` (`src/numgrad/tensor/functional.py`) using the log-sum-exp trick (subtract the row max before exponentiating). Notable finding: `log_softmax` doesn't need any special-cased "detach the max" handling — differentiating straight through the max subtraction via the ordinary `Tensor.max()` backward still gives the exact analytic gradient `delta_ij - softmax(x)_j`, because the max's own gradient terms cancel out algebraically in the full expression, regardless of how ties in the max are routed. `cross_entropy` is built compositionally from `log_softmax` + fancy-indexing + `mean()`, with no hand-derived backward of its own — correctness follows from the primitives. A stability test demonstrates a naive `log(softmax(x))` implementation overflowing to `nan`/`inf` on logits around `1e4`, contrasted against the stable version staying finite (forward and backward).

Added `clip_grad_norm(parameters, max_norm)` for later training use — rescales all parameters' `.grad` in place by a single global L2 norm ratio when it exceeds the threshold.

`benchmarks/bench_naive_vs_vectorized.py` times a pure-Python nested-loop elementwise-multiply-and-sum against the vectorized `Tensor` version across sizes up to 1024×1024, printing a speedup table (observed roughly 4×–40× in this run — the first concrete data point for the "why vectorize" story that Step 5's convolution benchmark will build on).

All 173 tests pass; `ruff check .` and `mypy src` (strict) both clean.

### Step 4 — Neural network layers (fully connected)

Added `src/numgrad/nn/`: a `Module` base class whose `__setattr__` auto-registers any `Tensor` with `requires_grad=True` as a parameter and any `Module` as a submodule, so `.parameters()`, `.zero_grad()`, and `.train()`/`.eval()` all recurse correctly through arbitrarily nested structures without any manual bookkeeping. `Linear` initializes its weight with Kaiming/He normal init (`std = sqrt(2/fan_in)`, implemented directly rather than hardcoded) — chosen over Xavier because it's derived specifically to counteract ReLU zeroing out about half its inputs on average, which is the activation this project's MLP/CNN actually use. `ReLU`/`Sigmoid`/`Tanh` are thin `Module` wrappers around new `Tensor` primitives (`relu`/`sigmoid`/`tanh`); `GELU` and `Dropout` (inverted, train-time-only scaling) are implemented as `Module`s. `Sequential` chains layers and registers each into `_modules` by index so parameter discovery and forward order match.

Caught a second real numerical bug the same way Step 2's compounding-gradient bug was caught — by testing, not inspection: the first `sigmoid()` used `np.where(cond, branch_a, branch_b)` to pick the overflow-safe formula per element, but NumPy evaluates *both* branches eagerly before selecting, so the unused branch still overflowed (`exp(-x)` for very negative `x`, `exp(x)` for very positive `x`) — invisible in the final result only because `inf`'s limiting value happened to be correct, but producing the exact same `RuntimeWarning: overflow encountered in exp` that Step 3's naive-softmax test exists to catch. A test with `x = [-1e4, 1e4]` surfaced the warning. Fixed by masking the *input* array before exponentiating (`exp(-x[x>=0])`, `exp(x[x<0])`) so only the safe branch is ever computed per element — this is the actual technique behind `scipy.special.expit`.

End-to-end gradient checks run through a full `Sequential(Linear, ReLU, Linear)` stack — both with respect to the input and to each layer's parameters directly, by perturbing the exact `Tensor` objects `model.parameters()` returns in place (no need to reconstruct or swap anything, since the model holds live references) — which catches graph-wiring bugs that per-op tests alone would miss. All 208 tests pass; `ruff check .` and `mypy src` (strict) both clean.

### Step 5 — Convolution and the vectorization case study

Added `src/numgrad/nn/functional.py` with `im2col`/`col2im` (built on `numpy.lib.stride_tricks.sliding_window_view` for a copy-free windowed view, plus one transpose+reshape to lay patches out as matmul-ready columns) and `conv2d`/`max_pool2d` as hand-backward Tensor ops (importing the tensor package's internal `_make` helper directly, same low-level mechanism every other custom-backward op in the project uses). `Conv2d` reduces its forward pass to one matmul of the reshaped kernel against `im2col(x)`; its backward applies ordinary matmul gradient rules for `d_weight`/`d_bias` and uses `col2im` (im2col's inverse, scatter-adding overlapping patch gradients back onto the padded input then cropping) for `d_x`. `MaxPool2d` reuses the same im2col windowing but maxes within each channel independently (no cross-channel matmul), routing gradient to the first-occurrence argmax per window — a deliberate difference from `Tensor.max()`'s tie-splitting in Step 3, matching how pooling is conventionally implemented. `Conv2d`/`MaxPool2d`/`Flatten` are thin `Module` wrappers; `Conv2d` uses the same Kaiming/He init as `Linear`.

Correctness: gradient-checked `Conv2d`/`MaxPool2d` (input, weight, and bias separately) across multiple shapes/strides/padding, a hand-computed-sum sanity check for a trivial all-ones kernel, an explicit tie-breaking test for `MaxPool2d`, a `col2im`-is-left-inverse-of-`im2col` check on non-overlapping patches, and an end-to-end `Conv2d → MaxPool2d → Flatten → Linear` gradcheck.

Headline benchmark: `benchmarks/bench_conv.py` times a pure-Python 4-nested-loop convolution (batch, output channel, output row, output col, with the kernel-window sum done per output pixel) against the im2col+matmul version. Observed speedups ranged from **16× up to ~500×** across input sizes up to `(4,8,32,32)` with a `16`-channel `3x3` kernel — squarely in the "one to two orders of magnitude" range the plan anticipated, and the clearest evidence yet for why vectorization matters: NumPy pushes the same arithmetic into contiguous-memory, BLAS-backed C loops instead of paying Python's per-element interpreter overhead on every multiply-add.

All 232 tests pass; `ruff check .` and `mypy src` (strict) both clean.

### Step 6 — Optimizers and training utilities

Added `src/numgrad/optim/`: an `Optimizer` base (`parameters`, `lr`, `zero_grad()`), `SGD` (classic undampened momentum: `v = momentum*v + grad`, `p -= lr*v`), `Adam` (explicit bias-corrected moment estimates, following Kingma & Ba 2015 directly), and `CosineAnnealingLR` as a standalone scheduler that mutates an optimizer's `.lr` attribute rather than being baked into the optimizer. Added `src/numgrad/data/`: `DataLoader` (shuffled mini-batches per iteration, reproducible via an injected `np.random.Generator`), `train_test_split`, and `one_hot`.

Debugging note worth flagging: while writing the Adam bias-correction test, I initially asserted the wrong direction — guessed that skipping correction would make early updates too *small*. Checking the actual arithmetic (`beta1=0.9, beta2=0.999, g=2.0`) showed the opposite: the uncorrected first update is over 3x *larger* than the corrected one. The reason is `v`'s bias dominates `m`'s, since `beta2` is far closer to 1 than `beta1`, so `v` stays underestimated for much longer — dividing by its too-small `sqrt(v)` inflates the step rather than shrinking it. Fixed the test assertion and the explanatory comment to match; this is exactly the kind of thing gradient/numerical checking is for — catching a plausible-sounding but wrong mental model before it ships. A separate test demonstrates a cleaner, provable property: with `m0=v0=0` and a *constant* gradient, bias correction exactly cancels the EMA startup transient (`m_hat_t = g` for every `t`, by induction), giving an identical update at every single step from `t=1` onward.

Convergence tests train a `Linear` layer on synthetic linear regression (known ground-truth weights) with both `SGD` and `Adam`, driving loss to near-zero within 300 steps; `DataLoader`/`train_test_split` are checked for correct batch counts/shapes, reproducible shuffling with a fixed seed, and disjoint/complete train-test partitions. All 259 tests pass; `ruff check .` and `mypy src` (strict) both clean.
