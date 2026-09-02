from __future__ import annotations

import numpy as np
import numpy.typing as npt

from numgrad.tensor.tensor import Array, Tensor


def log_softmax(logits: Tensor, axis: int = -1) -> Tensor:
    """log(softmax(logits)), computed with the log-sum-exp stability trick.

    Subtracting the row max before exponentiating keeps exp() from overflowing
    on large logits. Differentiating straight through the max subtraction
    (rather than treating it as a detached constant) still gives the exact
    analytic gradient delta_ij - softmax(x)_j: the max's own gradient terms
    cancel out algebraically in the full expression, regardless of how ties
    in the max are routed.
    """
    shifted = logits - logits.max(axis=axis, keepdims=True)
    log_sum_exp = shifted.exp().sum(axis=axis, keepdims=True).log()
    return shifted - log_sum_exp


def cross_entropy(logits: Tensor, targets: npt.ArrayLike) -> Tensor:
    """Mean cross-entropy loss for integer class-index targets, shape (N,)."""
    targets_arr = np.asarray(targets, dtype=np.int64)
    n = logits.shape[0]
    log_probs = log_softmax(logits, axis=-1)
    picked = log_probs[np.arange(n), targets_arr]
    return -picked.mean()


def clip_grad_norm(parameters: list[Tensor], max_norm: float, eps: float = 1e-6) -> float:
    """Rescale a set of parameters' .grad in place if their global L2 norm exceeds max_norm.

    Returns the (pre-clipping) total norm.
    """
    grads: list[Array] = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0

    total_norm = float(np.sqrt(sum(np.sum(g**2) for g in grads)))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad = p.grad * scale
    return total_norm
