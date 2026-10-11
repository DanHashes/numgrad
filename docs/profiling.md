# Profiling notes

Methodology: `cProfile` around one full epoch of `scripts/train.py --model cnn`
on real MNIST (batch size 128, ~422 training batches + validation), and
`tracemalloc` around a single forward+backward pass on one batch for both
model types. The CNN was chosen for the CPU-time profile because it's the
model doing the most interesting arithmetic; the memory comparison is more
informative done against the MLP as a baseline.

## CPU time: was convolution actually the bottleneck?

Short answer: yes, but not for the reason I expected. Going in, the obvious
guess was "the im2col + matmul forward pass will dominate." The actual
profile (sorted by cumulative time, top of the list) said otherwise:

```
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
  1266   15.273    0.012   15.273    0.012 {built-in method numpy._core._multiarray_umath.c_einsum}
  1096    4.359    0.004    8.516    0.008 nn/functional.py:57(conv2d)              <- forward
  1266    4.884    0.004    5.068    0.004 nn/functional.py:28(col2im)
```

`np.einsum` alone — 15.3s out of a 51.1s epoch (~30%) — was the single
largest cost in the entire training loop, bigger than `conv2d`'s own forward
pass (8.5s cumulative). It was being called from `conv2d`'s *backward*: both
`d_weight = einsum("nop,nkp->ok", grad_flat, cols)` and
`d_cols = einsum("nop,ok->nkp", grad_flat, weight_flat)` are the kind of
contraction `einsum` is supposed to be good at, but NumPy's general-purpose
`einsum` evaluator doesn't route this particular pattern through BLAS the
way `@` (`matmul`) does — it falls back to a much more naive C loop. So the
real finding isn't "convolution is slow," it's narrower and more useful:
**the backward pass's tensor contractions were slower than the forward
convolution that motivated im2col in the first place**, and for a reason
(generic `einsum` vs. BLAS-backed `matmul`) that's easy to miss by just
reading the math — the two expressions are mathematically identical.

### The fix

Both contractions are expressible as batched matmuls instead:

```python
# d_weight = sum_n grad_flat[n] @ cols[n].T  ->  (C_out, K)
d_weight = (grad_flat @ cols.transpose(0, 2, 1)).sum(axis=0)

# d_cols = weight_flat.T @ grad_flat[n], per batch  ->  (N, K, out_h*out_w)
d_cols = weight_flat.T[None, :, :] @ grad_flat
```

Verified numerically identical to the `einsum` versions (`np.allclose`,
every existing `Conv2d` gradcheck still passes) and markedly faster in
isolation:

| contraction | einsum | matmul | speedup |
|---|---|---|---|
| `d_weight` (N=128,Cout=16,K=72,P=196) | 7.56ms | 2.59ms | 2.9x |
| `d_cols` (same shapes) | 27.0ms | 5.98ms | 4.5x |

End to end, one real CNN epoch on MNIST:

| | before | after |
|---|---|---|
| epoch wall time | 47.1s | **35.0s** |
| profiled total (cProfile overhead included) | 51.1s | 38.9s |

**~26% faster per epoch** from changing two lines, with `einsum` no longer
appearing anywhere in the post-fix profile's hot path. This is now the
implementation in `src/numgrad/nn/functional.py::conv2d`.

### What's left in the hot path after the fix

With `einsum` gone, the next-largest items are what you'd actually expect
from an im2col-based conv net: `conv2d` forward (8.4s), `col2im` (4.8s, the
backward-pass counterpart of `im2col`'s windowing), `im2col` itself (6.4s),
and `np.ascontiguousarray` (3.8s, the copy im2col needs to materialize a
strided view into a real buffer for matmul). None of these looked like an
"easy win" the way the einsum call was — they're the actual, unavoidable
cost of the im2col approach, not a mechanical inefficiency.

## Memory: `tracemalloc` on one forward+backward pass, batch size 128

| model | peak memory |
|---|---|
| MLP | 6.78 MB |
| CNN | 76.69 MB |

The CNN uses **~11x more memory** than the MLP for an identical batch size.
Breaking down the top allocation sites for the CNN:

1. `tensor.py: Tensor._accumulate_grad`'s `grad.copy()` — 22.1 MB across 63
   still-live gradient buffers (one per graph node that received a
   gradient). Expected: proportional to the number of op nodes and their
   activation sizes, and really a tally of the *design* ("every node gets a
   persistent `.grad`"), not an avoidable cost specific to this run.
2. `functional.py: im2col`'s `np.ascontiguousarray(cols)` — 20.7 MB across 9
   still-live column buffers. This is the real signature of the im2col
   approach: each conv/pool layer's input gets replicated `kernel_h *
   kernel_w` times over (once per offset in the sliding window) to make the
   matmul possible, and because `conv2d`'s backward closure captures `cols`
   by reference, that expanded buffer stays alive for as long as the
   computation graph does — not just for the duration of the forward call.

This is a genuine, inherent space/time tradeoff of im2col, not a bug: the
alternative (a direct/looped convolution, or recomputing `im2col` during
backward instead of retaining it) would use less memory but either be much
slower (back to nested Python loops) or repeat work already shown to be a
bottleneck above. For this project — training on a CPU, where a few tens of
MB per batch is immaterial — keeping the retained-`cols` version is the
right call, so no change was made here; it's recorded as a known
characteristic rather than a fix.

## Coverage note (see PROGRESS.md Step 9 for the full story)

`pytest --cov` surfaced one systematic, real pattern worth a mention here
too since it came from the same hardening pass: every single-input Tensor
op's backward closure had an `if not self.requires_grad: return []` guard
that coverage marked as never hit. It turned out to be genuinely
unreachable — `_make()` only ever attaches a `_backward` function when the
op's sole parent already has `requires_grad=True`, so the guard could never
see `False`. Deleted rather than tested (ten occurrences across
`tensor.py` and one in `nn/functional.py`'s `max_pool2d`); the equivalent
guards on *multi*-parent ops (`add`, `mul`, `conv2d`, ...) are real and
stayed, since any individual operand there can legitimately have
`requires_grad=False` while the op's output still does.
