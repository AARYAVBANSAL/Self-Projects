# GPT-2 Small — Math Notes

Companion notes for `gpt2_small.py`, covering the linear algebra and calculus
behind each component.

## 1. Embeddings
Token embedding: a lookup table `E ∈ R^(V×d)` mapping token id → vector.
Positional embedding: `P ∈ R^(T×d)` mapping position → vector.
Input to the first block: `x = E[tokens] + P[positions]`, shape `(B, T, d)`.

## 2. Scaled dot-product self-attention
For queries `Q`, keys `K`, values `V` (each `(T, d_k)` per head):

    Attention(Q, K, V) = softmax( Q Kᵀ / sqrt(d_k) + M ) V

- `Q Kᵀ` is a `(T, T)` matrix of similarity scores between every pair of
  positions (dot product = projection of one vector onto another).
- Dividing by `sqrt(d_k)` keeps the softmax input's variance ~1 regardless
  of head dimension, preventing vanishing gradients from a saturated
  softmax (a large dot-product magnitude pushes softmax toward one-hot,
  killing gradient flow).
- `M` is the causal mask: `M[i,j] = -∞` for `j > i`, `0` otherwise, so
  softmax assigns zero probability to future positions.
- Multi-head attention runs `h` of these in parallel on `d/h`-dim
  subspaces, then concatenates and projects back to `d`, letting the model
  attend to different representation subspaces simultaneously.

## 3. Layer normalization
For a vector `x ∈ R^d`:

    LN(x) = γ * (x - μ) / sqrt(σ² + ε) + β

where `μ, σ²` are the mean/variance across the feature dimension. This
re-centers and re-scales activations per-token, stabilizing the
distribution of inputs to each sub-layer (independent of batch statistics,
unlike BatchNorm) which is important for training deep transformer stacks.

## 4. MLP block
`h = GELU(x W₁ + b₁) W₂ + b₂`, expanding to `4d` and back. GELU is a smooth
approximation to `x · Φ(x)` (`Φ` = standard normal CDF), giving a
differentiable, non-monotonic nonlinearity that empirically outperforms
ReLU for transformers (avoids ReLU's hard zero-gradient region near 0).

## 5. Residual connections
Each sub-layer computes `x = x + f(LN(x))`. The identity shortcut means the
gradient of the loss w.r.t. an early layer's output has a direct
`∂L/∂x_out · 1` term (from the `+x`) plus the sub-layer's Jacobian term:

    ∂L/∂x_in = ∂L/∂x_out * (I + ∂f/∂x_in)

The `I` term prevents vanishing gradients in very deep networks (this is
exactly the same idea as ResNets).

## 6. Loss and backpropagation
Given logits `z ∈ R^V` and true next-token id `t`, the per-token loss is
cross-entropy with softmax:

    L = -log( exp(z_t) / Σ_v exp(z_v) ) = -z_t + logsumexp(z)

Its gradient w.r.t. logits has the clean closed form `∂L/∂z = softmax(z) - onehot(t)`,
which is why cross-entropy + softmax is used almost universally for
classification/language-modeling heads — it gives large gradients when the
model is confidently wrong and small gradients when it's already correct.

Backprop then applies the chain rule through: `lm_head → ln_f → blocks (in
reverse) → embeddings`, accumulating `∂L/∂θ` for every parameter `θ` via
reverse-mode automatic differentiation (what `loss.backward()` does).

## 7. Optimization
AdamW maintains running estimates of the first and second moments of the
gradient:

    m_t = β₁ m_{t-1} + (1-β₁) g_t
    v_t = β₂ v_{t-1} + (1-β₂) g_t²
    θ_t = θ_{t-1} - lr * m̂_t / (sqrt(v̂_t) + ε) - lr * λ * θ_{t-1}

The last term is *decoupled* weight decay (`λ`), applied directly to the
parameters rather than folded into the gradient like classic L2
regularization — this is the key difference between Adam and AdamW and
tends to generalize better for transformers.

## 8. Byte-Pair Encoding
BPE is a greedy compression scheme, not calculus-based, but it's worth
noting why it works well for language modeling: it interpolates between
character-level (small vocab, long sequences) and word-level (huge vocab,
poor handling of rare/unseen words) tokenization by merging the most
frequent adjacent symbol pairs first, so common sub-words become single
tokens while rare words fall back to smaller pieces.
