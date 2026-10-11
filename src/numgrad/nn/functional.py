from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from numgrad.tensor.tensor import Array, Tensor, _make


def im2col(x: Array, kernel_h: int, kernel_w: int, stride: int, padding: int) -> Array:
    """(N, C, H, W) -> (N, C*kernel_h*kernel_w, out_h*out_w).

    Takes a strided sliding-window VIEW (no copy) over the padded input, then
    does a single transpose+reshape to lay patches out as matmul-ready
    columns, so convolution reduces to one big matmul instead of four nested
    Python loops.
    """
    n, c, h, w = x.shape
    out_h = (h + 2 * padding - kernel_h) // stride + 1
    out_w = (w + 2 * padding - kernel_w) // stride + 1
    x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

    windows = sliding_window_view(x_padded, (kernel_h, kernel_w), axis=(2, 3))
    windows = windows[:, :, ::stride, ::stride, :, :]  # (N, C, out_h, out_w, kh, kw)
    cols = windows.transpose(0, 1, 4, 5, 2, 3)  # (N, C, kh, kw, out_h, out_w)
    return np.ascontiguousarray(cols).reshape(n, c * kernel_h * kernel_w, out_h * out_w)


def col2im(
    cols: Array,
    x_shape: tuple[int, int, int, int],
    kernel_h: int,
    kernel_w: int,
    stride: int,
    padding: int,
) -> Array:
    """Inverse of im2col: scatter-add overlapping patch gradients back onto
    the (padded) input, then crop the padding off. Loops over the kernel_h *
    kernel_w positions (small and fixed) rather than every output pixel.
    """
    n, c, h, w = x_shape
    out_h = (h + 2 * padding - kernel_h) // stride + 1
    out_w = (w + 2 * padding - kernel_w) // stride + 1
    x_padded = np.zeros((n, c, h + 2 * padding, w + 2 * padding), dtype=cols.dtype)
    cols_reshaped = cols.reshape(n, c, kernel_h, kernel_w, out_h, out_w)

    for i in range(kernel_h):
        for j in range(kernel_w):
            h_slice = slice(i, i + stride * out_h, stride)
            w_slice = slice(j, j + stride * out_w, stride)
            x_padded[:, :, h_slice, w_slice] += cols_reshaped[:, :, i, j, :, :]

    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


def conv2d(x: Tensor, weight: Tensor, bias: Tensor, stride: int, padding: int) -> Tensor:
    """2D convolution via im2col + matmul.

    weight: (C_out, C_in, kH, kW). Forward reduces to one matmul of the
    reshaped kernel against the im2col'd input; backward applies ordinary
    matmul gradient rules to get d_weight, and col2im (im2col's inverse) to
    scatter the column-space gradient back onto the input.

    d_weight/d_cols are expressed as batched matmuls (`@`), not `np.einsum`:
    profiling the Step 8 training loop (docs/profiling.md) found einsum was
    the single largest cost in the whole run, ~3-4.5x slower here than the
    equivalent matmul despite computing the same contraction — einsum's
    general-purpose evaluator doesn't route through BLAS the way `@` does.
    """
    n, _, h, w = x.data.shape
    c_out, _, kh, kw = weight.data.shape
    out_h = (h + 2 * padding - kh) // stride + 1
    out_w = (w + 2 * padding - kw) // stride + 1

    cols = im2col(x.data, kh, kw, stride, padding)  # (N, C_in*kh*kw, out_h*out_w)
    weight_flat = weight.data.reshape(c_out, -1)  # (C_out, K)
    out_data = weight_flat[None, :, :] @ cols  # (N, C_out, out_h*out_w)
    out_data = out_data + bias.data.reshape(1, c_out, 1)
    out_data = out_data.reshape(n, c_out, out_h, out_w)

    def _backward(grad_output: Array) -> list[tuple[Tensor, Array]]:
        contributions: list[tuple[Tensor, Array]] = []
        grad_flat = grad_output.reshape(n, c_out, out_h * out_w)
        if bias.requires_grad:
            contributions.append((bias, grad_flat.sum(axis=(0, 2))))
        if weight.requires_grad:
            # sum_n grad_flat[n] @ cols[n].T -> (C_out, K)
            d_weight = (grad_flat @ cols.transpose(0, 2, 1)).sum(axis=0)
            contributions.append((weight, d_weight.reshape(weight.data.shape)))
        if x.requires_grad:
            # weight_flat.T @ grad_flat[n] -> (K, out_h*out_w), per batch
            d_cols = weight_flat.T[None, :, :] @ grad_flat
            d_x = col2im(d_cols, x.data.shape, kh, kw, stride, padding)
            contributions.append((x, d_x))
        return contributions

    return _make(out_data, (x, weight, bias), _backward)


def max_pool2d(x: Tensor, kernel_size: int, stride: int) -> Tensor:
    """Max pooling via the same im2col windowing, maxing within each channel's
    own kernel_size x kernel_size window (no cross-channel mixing, unlike
    conv2d's matmul). Ties go to the first occurrence (np.argmax's default),
    matching standard pooling implementations — a deliberate difference from
    Tensor.max()'s generic reduction, which splits ties evenly.
    """
    n, c, h, w = x.data.shape
    kh = kw = kernel_size
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1

    cols = im2col(x.data, kh, kw, stride, padding=0).reshape(n, c, kh * kw, out_h * out_w)
    out_data = cols.max(axis=2).reshape(n, c, out_h, out_w)
    argmax_idx = cols.argmax(axis=2)  # (N, C, out_h*out_w)

    def _backward(grad_output: Array) -> list[tuple[Tensor, Array]]:
        grad_flat = grad_output.reshape(n, c, out_h * out_w)
        d_cols = np.zeros((n, c, kh * kw, out_h * out_w), dtype=x.data.dtype)
        np.put_along_axis(d_cols, argmax_idx[:, :, None, :], grad_flat[:, :, None, :], axis=2)
        d_x = col2im(
            d_cols.reshape(n, c * kh * kw, out_h * out_w), x.data.shape, kh, kw, stride, padding=0
        )
        return [(x, d_x)]

    return _make(out_data, (x,), _backward)
