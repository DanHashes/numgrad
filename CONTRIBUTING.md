# Contributing

This started as a from-scratch learning project, so the bar for contributions
is: does it keep working the way everything else here works — pure NumPy, no
autodiff/ML dependencies, every op gradient-checked.

## Setup

```bash
git clone https://github.com/DanHashes/numgrad.git
cd numgrad
pip install -e ".[dev]"
```

## Before opening a PR

```bash
ruff check .        # lint
mypy src            # strict type checking on the library (not scripts/)
pytest               # full test suite (network-free by default)
pytest --cov=src/numgrad --cov-report=term-missing   # coverage, if you touched core logic
```

All three must pass; CI runs them as separate gating jobs (`lint`,
`typecheck`, `test`) across Python 3.10–3.12.

## Adding a new `Tensor` op

Every differentiable op follows the same shape:

```python
def my_op(self, ...) -> Tensor:
    data = ...  # forward, plain NumPy on self.data
    def _backward(grad_output: Array) -> list[tuple[Tensor, Array]]:
        return [(self, ...)]  # local gradient contribution(s)
    return _make(data, (self, ...other parent tensors...), _backward)
```

- `_backward` must be a **pure function of `grad_output`** — never read a
  node's `.grad` from inside it. That's not a style preference: an earlier
  version did read `.grad` directly and it silently produced wrong,
  compounding gradients on a second `.backward()` call (see `PROGRESS.md`,
  Step 2). `_make()` already gates whether `_backward` is even attached, so
  don't add an `if not self.requires_grad: return []` guard inside a
  single-parent op's closure either — it's unreachable dead code there (see
  Step 9); it's only meaningful for multi-parent ops where one operand can
  legitimately not require grad while another does.
- If the op broadcasts, route the gradient through `_unbroadcast(grad,
  target_shape)` rather than writing bespoke shape-reduction logic.
- Add a `tests/helpers.py::gradcheck` test (see any existing op's test for
  the pattern) across a couple of shapes and random seeds. If the op has a
  numerically sensitive form (anything involving `exp`/`log`/division),
  also add a stability test with large-magnitude inputs, following the
  `log_softmax` naive-vs-stable pattern in `tests/test_extended_ops.py`.

## Commit messages

No fixed convention required for new contributions, but PRs that touch core
numerics should explain *why*, not just *what* — the existing commit history
(and `PROGRESS.md`) tries to record the reasoning and any bugs caught along
the way; new changes are a lot more useful to future readers if they do the
same.
