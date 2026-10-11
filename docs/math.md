# The math behind numgrad

This is the derivation sheet behind `src/numgrad`: backprop through a matmul
and through broadcasting, the log-sum-exp stability trick, and the
`im2col`/`col2im` convolution gradient. Each section matches a specific
function in the code, cited by path.

## 1. Backprop through matrix multiplication

`src/numgrad/tensor/tensor.py::Tensor.__matmul__`

Forward: $C = A B$, where $A$ is $(m \times k)$, $B$ is $(k \times n)$, $C$ is
$(m \times n)$.

Given the upstream gradient $\bar{C} = \partial L/\partial C$ (same shape as
$C$), we want $\bar{A} = \partial L/\partial A$ and $\bar{B} = \partial
L/\partial B$.

Element-wise, $C_{ij} = \sum_k A_{ik} B_{kj}$, so:

$$\frac{\partial L}{\partial A_{ik}} = \sum_j \frac{\partial L}{\partial C_{ij}} \frac{\partial C_{ij}}{\partial A_{ik}} = \sum_j \bar{C}_{ij} B_{kj}$$

which is exactly $(\bar{C} B^T)_{ik}$. Symmetrically:

$$\frac{\partial L}{\partial B_{kj}} = \sum_i \bar{C}_{ij} A_{ik} = (A^T \bar{C})_{kj}$$

So:

$$\bar{A} = \bar{C} B^T \qquad \bar{B} = A^T \bar{C}$$

**Shape sanity check** (this is the fast way to re-derive this from memory
without redoing the algebra): $\bar{A}$ must have the same shape as $A$,
$(m \times k)$. $\bar{C}$ is $(m \times n)$ and $B^T$ is $(n \times k)$, so
$\bar{C} B^T$ is $(m \times k)$ — matches. Same check for $\bar{B} = A^T
\bar{C}$: $A^T$ is $(k \times m)$, $\bar{C}$ is $(m \times n)$, product is
$(k \times n)$ — matches $B$'s shape.

The code generalizes this to batched matmul (shape `(..., m, k) @ (..., k,
n)`) by swapping only the last two axes (`np.swapaxes(x, -1, -2)`, not a
full transpose) before applying the same two formulas, then routing the
result through `_unbroadcast` in case the batch dimensions themselves were
broadcast (see §2).

## 2. Backprop through broadcasting

`src/numgrad/tensor/tensor.py::_unbroadcast`

NumPy's broadcasting rule: to combine two arrays of different shapes, each
is conceptually aligned from the right, and any axis of size 1 (or missing
entirely) is treated as repeated to match the other operand. The question
for backprop: if the forward pass broadcast `a` (shape `(3,)`) up to shape
`(32, 3)` to add it to a batch, what's the correct gradient *back* onto the
original `(3,)`-shaped `a`?

Broadcasting a value is mathematically the same as a `sum`/reduction run
**backwards**: if $y = x$ repeated along some axis, then $\partial L/\partial
x$ must *sum* the incoming gradient over exactly the axis (or axes) that got
repeated — this falls straight out of the multivariate chain rule, since
each repeated copy of $x$ independently contributes to the loss and those
contributions add.

`_unbroadcast(grad, target_shape)` applies this in two passes:

1. **Extra leading axes** (broadcasting prepends axes, e.g. `(3,)` → `(32,
   3)` adds a new axis 0): sum over axis 0 repeatedly until `grad.ndim ==
   len(target_shape)`.
2. **Existing axes that were size 1** (e.g. `(1, 3)` → `(32, 3)`): for each
   axis where `target_shape[axis] == 1` but `grad.shape[axis] != 1`, sum
   over that axis with `keepdims=True`.

This is the one piece of logic every binary/broadcasting op in the codebase
(`+`, `-`, `*`, `/`, `@`) routes through — writing it once here, rather than
re-deriving "which axes got broadcast" inside every operator, is what keeps
each operator's backward short and (per Step 2's postmortem) is also what
makes the one subtle bug class in this area easy to find and fix in a
single place instead of a dozen.

## 3. The log-sum-exp trick, and why differentiating through the max is still correct

`src/numgrad/tensor/functional.py::log_softmax`

**The numerical problem.** $\text{softmax}(x)_i = e^{x_i} / \sum_j e^{x_j}$.
For logits with any sizable magnitude (a few hundred, let alone the `1e4`
used in this project's stability test), $e^{x_i}$ overflows a float64 long
before the ratio itself would be a perfectly reasonable number in $[0, 1]$.

**The fix.** Subtract the row max $m = \max_j x_j$ before exponentiating —
mathematically free, since it cancels in the ratio:

$$\text{softmax}(x)_i = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}$$

Now every exponent is $\le 0$, so $e^{x_i - m} \in (0, 1]$: no overflow.
Taking `log` of that directly (rather than `log(softmax(x))` computed
naively) gives the stable `log_softmax`:

$$\log\text{softmax}(x)_i = (x_i - m) - \log\sum_j e^{x_j - m}$$

which is exactly what `log_softmax()` computes, using ordinary `Tensor` ops
(`max`, subtraction, `exp`, `sum`, `log`).

**The subtlety.** $m$ is itself a function of $x$ (via `Tensor.max()`, which
has its own backward). Does differentiating straight through that
subtraction — rather than treating $m$ as a detached constant, which is what
most hand-written implementations do "to be safe" — still give the right
gradient? Taking the derivative of the full expression above with respect to
$x_i$, writing $S = \sum_j e^{x_j - m}$:

$$\frac{\partial}{\partial x_i}\Big[(x_i - m) - \log S\Big] = \delta_i - \frac{\partial m}{\partial x_i} - \frac{1}{S}\sum_j e^{x_j-m}\left(\delta_{ij} - \frac{\partial m}{\partial x_i}\right)$$

Expanding the sum ($\sum_j e^{x_j-m}\delta_{ij} = e^{x_i-m}$ and $\sum_j
e^{x_j-m} = S$):

$$= \delta_i - \frac{\partial m}{\partial x_i} - \frac{e^{x_i-m}}{S} + \frac{\partial m}{\partial x_i} = \delta_i - \text{softmax}(x)_i$$

The two $\partial m/\partial x_i$ terms **cancel exactly**, leaving the
textbook-correct gradient $\delta_i - \text{softmax}(x)_i$ — regardless of
what $\partial m/\partial x_i$ actually is. That's why `Tensor.max()`'s
particular tie-breaking convention (splitting gradient evenly across tied
maxima, a Step 3 design choice) has zero effect on `log_softmax`'s
correctness: the cancellation holds for *any* valid subgradient of max.
Confirmed in practice by `cross_entropy`'s gradient check, which never
special-cases the max subtraction at all — correctness follows purely from
composing already-correct primitives.

## 4. `im2col` / `col2im`: convolution as one matmul, and its gradient

`src/numgrad/nn/functional.py::im2col`, `col2im`, `conv2d`

**Forward.** A direct 2D convolution with kernel $W \in
\mathbb{R}^{C_{out}\times C_{in}\times k_h\times k_w}$ computes, for each
output pixel, a dot product between the kernel and the corresponding input
patch. `im2col(x)` lays out every such patch as a column of a matrix $X_{col}
\in \mathbb{R}^{(C_{in}k_hk_w)\times(H_{out}W_{out})}$ (per batch element),
and reshapes $W$ to $W_{flat} \in \mathbb{R}^{C_{out}\times(C_{in}k_hk_w)}$.
Convolution becomes a single matmul:

$$Y_{flat} = W_{flat}\, X_{col} \qquad (C_{out}\times(C_{in}k_hk_w)) \cdot ((C_{in}k_hk_w)\times(H_{out}W_{out})) = C_{out}\times(H_{out}W_{out})$$

reshaped back to $(C_{out}, H_{out}, W_{out})$.

**Backward w.r.t. the kernel.** This is now literally the matmul gradient
from §1, batched over $N$ and summed (since the same $W_{flat}$ is reused
for every sample in the batch, every sample's contribution to
$\partial L/\partial W_{flat}$ adds):

$$\bar{W}_{flat} = \sum_{n=1}^{N} \bar{Y}_{flat}^{(n)} \, (X_{col}^{(n)})^T$$

implemented as a batched matmul followed by a sum over the batch axis
(`(grad_flat @ cols.transpose(0,2,1)).sum(axis=0)` — see `docs/profiling.md`
for why this beats the mathematically-equivalent `np.einsum` version by
3-4.5x in practice).

**Backward w.r.t. the input: `col2im`.** By the same matmul rule,
$\bar{X}_{col}^{(n)} = W_{flat}^T\, \bar{Y}_{flat}^{(n)}$ gives the gradient
*in column space* — but we need it back in image space, as $\bar{x}$. This
is where `col2im` comes in: it's `im2col`'s exact inverse operation for
gradients. Since `im2col` built each column by *copying* (not moving) a
patch of the input — and patches overlap whenever `stride < kernel_size` —
each input pixel can appear in multiple columns. The multivariate chain rule
says their gradient contributions must be **summed**, not overwritten:

$$\bar{x}[n, c, i, j] = \sum_{\substack{(p, q, oh, ow) \\ \text{s.t. } i = oh\cdot s + p,\ j = ow\cdot s + q}} \bar{X}_{col}[n, (c,p,q), (oh,ow)]$$

`col2im` implements exactly this scatter-add, but organized as a loop over
the (small, fixed) $k_h \times k_w$ kernel offsets rather than over every
output pixel — for each offset `(i, j)`, every output position's gradient
for that offset lands at a *regularly strided* slice of the padded input, so
the whole offset can be added in one vectorized NumPy operation instead of
one Python-level addition per output pixel. This is also why `col2im` is
NOT simply `im2col` run backward through the same code path: `im2col`
reads with a strided *view* (no copy needed, since reads can alias), while
`col2im` must accumulate into overlapping regions, which requires real
scatter-add semantics — a view can't do that safely.

`MaxPool2d`'s backward is simpler by comparison: no channel-mixing matmul,
so there's no analogous kernel gradient, just the same `col2im` scatter
applied to a column buffer that's zero everywhere except at each window's
argmax position (see `Tensor.max()`'s similar, but not identical, masking
in Step 3 — pooling routes to the first tied maximum, matching standard
implementations, where `Tensor.max()` deliberately splits ties evenly).
