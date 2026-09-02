"""Shared test utilities: finite-difference gradient checking.

Reused across test_tensor_ops.py and later steps' test files as new ops/layers
are added, so the checking methodology stays consistent throughout the project.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from numgrad import Tensor

Array = npt.NDArray[np.float64]


def gradcheck(
    f: Callable[..., Tensor],
    inputs: list[Tensor],
    eps: float = 1e-5,
    tol: float = 1e-4,
) -> None:
    """Assert analytic gradients from backward() match finite-difference estimates.

    f(*inputs) may return a Tensor of any shape; internally this checks the
    gradient of sum(f(*inputs)) by seeding backward() with a ones-tensor,
    which sidesteps needing a real Tensor.sum() reduction to exist yet.
    """
    for t in inputs:
        t.grad = None

    out = f(*inputs)
    out.backward(np.ones_like(out.data))
    analytic_grads = [
        (t.grad.copy() if t.grad is not None else np.zeros_like(t.data)) for t in inputs
    ]

    for t, analytic_grad in zip(inputs, analytic_grads, strict=True):
        numeric_grad = np.zeros_like(t.data)
        it = np.nditer(t.data, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            orig = t.data[idx]

            t.data[idx] = orig + eps
            plus = float(f(*inputs).data.sum())

            t.data[idx] = orig - eps
            minus = float(f(*inputs).data.sum())

            t.data[idx] = orig
            numeric_grad[idx] = (plus - minus) / (2 * eps)

        rel_error = np.abs(analytic_grad - numeric_grad) / (
            np.abs(analytic_grad) + np.abs(numeric_grad) + eps
        )
        max_rel_error = float(np.max(rel_error))
        assert max_rel_error < tol, (
            f"gradcheck failed for input shape {t.data.shape}: "
            f"max relative error {max_rel_error:.2e} (tol={tol:.0e})\n"
            f"analytic:\n{analytic_grad}\nnumeric:\n{numeric_grad}"
        )
