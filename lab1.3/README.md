# Module 1.3 — Attention (Standalone)

> Part 1 — Building Blocks · [DiT from Scratch](../README.md)

**Goal**: implement scaled dot-product attention and multi-head attention from scratch, prove it matches `torch.nn.MultiheadAttention` numerically, then train an attention-only model on the reverse task to see attention actually learn.

**Why this matters for DiT**: every DiT block is a self-attention layer plus an MLP. The kernel you write here — `softmax(QKᵀ/√d) · V`, split across heads — is the exact operation DiT uses on its patch tokens. Getting it right once in isolation, with no transformer residuals or AdaLN in the way, means you never have to re-derive it later.

**Deliverables**:
- `attention.py` with `scaled_dot_product_attention` and `MultiHeadAttention` — parameter layout matches PyTorch's built-in so weights can be copied 1-to-1.
- `verify.py` confirming bit-for-bit parity (forward, causal-masked, and backward) against `torch.nn.MultiheadAttention`.
- `train.py` training an attention-only model on the reverse task to ≥99% per-token accuracy.
- `visualize.py` producing an attention heatmap showing the learned anti-diagonal pattern.

## What the model learns

**The reverse task.** Given a random integer sequence of length `L`, output the reversed sequence. That's it — no language, no real data, just `[a, b, c, d] → [d, c, b, a]`.

Why reverse and not copy: copy is solved by `attn = I` (each position attends to itself), which is trivially achievable by the positional embedding alone and makes attention look boring. Reverse requires `attn[i, j] = 1 if j == L-1-i else 0` — an anti-diagonal — which forces the query at position i to use positional information to pick out the key at position `L-1-i`. Mechanically interesting, visually striking.

```
 input tokens                    logits over vocab
(B, L)  →  [embed + pos]  →  [MultiHeadAttention]  →  [Linear head]  →  (B, L, vocab)
```

The model has exactly **one** attention layer. No MLP. No LayerNorm. No residual. If it reaches high accuracy, attention alone is doing the work.

## Files

| File | What it is |
| --- | --- |
| `attention.py` | `scaled_dot_product_attention`, `MultiHeadAttention` (matches PyTorch's layout), `causal_mask`, and `AttentionModel` (embed → attention → head) |
| `verify.py` | Copies weights from `torch.nn.MultiheadAttention` into ours and checks outputs, attention weights, and gradients match |
| `train.py` | Trains `AttentionModel` on the reverse task with random batches on the fly |
| `visualize.py` | Loads a checkpoint and saves a per-head attention heatmap for sample inputs |
| `demo/app.py` | Gradio webapp: interactive reverse prediction with live heatmaps, plus an attention calculator and parity checker |

## Instructions

### 1. Set up

Python 3.9+. From `lab1.3/`:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Training auto-selects device: CUDA → Apple MPS → CPU. No dataset download — the reverse task generates data on the fly.

### 2. Verify the implementation

Before training, prove the kernel is correct:

```bash
python verify.py
```

Expected output:

```
unmasked self-attention
  ✓ outputs match                    max |Δ| = 0.00e+00
  ✓ attention weights match          max |Δ| = 0.00e+00

causal self-attention
  ✓ outputs match (causal)           max |Δ| = 0.00e+00
  ✓ attention weights match (causal) max |Δ| = 0.00e+00
  ✓ causal attn upper-tri is zero    max |Δ| = 0.00e+00

backward pass parity (gradients w.r.t. input)
  ✓ input gradients match            max |Δ| = ~1e-7

ALL CHECKS PASSED ✓
```

This works because `MultiHeadAttention` in `attention.py` uses the same parameter layout as PyTorch's built-in (`in_proj_weight`, `in_proj_bias`, `out_proj.weight`, `out_proj.bias`), so weights copy 1-to-1. If outputs match at the same weights, the math is right.

### 3. How scaled dot-product attention works

Open `attention.py` and read the `scaled_dot_product_attention` docstring. The whole operation is three lines:

```python
scores = Q @ Kᵀ / sqrt(d_k)     # (L_q, L_k) — each row is "query i's similarity to all keys"
attn   = softmax(scores, -1)     # row-normalize: each query distributes attention probability 1.0
out    = attn @ V                # each output row is a convex combination of value rows
```

**Why `sqrt(d_k)`?** The variance of `q · k` grows linearly in `d_k`. Without scaling, softmax saturates near a one-hot vector and its gradient vanishes. Dividing by `sqrt(d_k)` keeps the pre-softmax logits roughly unit-variance at initialization.

**Why multi-head?** A single head has one `(d_k × d_k)` similarity pattern for the whole sequence. Splitting into `H` heads gives `H` independent patterns of width `d_k / H` — each can specialize. Total parameter count is unchanged: you're reshaping the same `(D, D)` projection.

**What the mask does.** `attn_mask` either blocks (True/-inf) or reweights (additive float) specific query-key pairs *before* softmax. In lab 1.4 (nanoGPT) you'll use a causal mask so position `i` can't see positions `> i`. The reverse task uses no mask — each output position needs to see the whole input to find its mirror partner.

### 4. Train on the reverse task

```bash
python train.py
```

Default: Adam, lr=3e-3, 2000 steps, seq_len=10, vocab=20, embed_dim=64, 4 heads. Per-log output:

```
step   100 |   2.8s | train_loss 2.4891  train_acc 0.191 | val_loss 2.3312  val_acc 0.225
...
step  2000 |  28.6s | train_loss 0.0273  train_acc 0.995 | val_loss 0.0251  val_acc 0.996
```

A random baseline is `loss = log(vocab) ≈ 3.0`, `acc = 1/vocab = 0.05`. Watch the loss drop as the model discovers the anti-diagonal. Reaching ≥99% val accuracy is typical within 2000 steps on a laptop.

### 5. Visualize the learned attention

```bash
python visualize.py --ckpt attention.pt --save attention_weights.png
```

Each row of the saved figure is one example; each column is one attention head. The cells at position `(i, L-1-i)` — the anti-diagonal — should be bright; everything else near zero. That's the learned routing: "to produce output at position i, attend to input at position `L-1-i`."

To average across heads (easier to read, hides specialization):

```bash
python visualize.py --average --save attention_avg.png
```

Different heads often learn slightly different anti-diagonals — some sharper, some more diffuse, some slightly offset. With only one attention layer solving a simple task, many heads become redundant; in a transformer with MLPs and stacked layers, each head earns a distinct role.

### 6. Interactive demo

After training, launch the webapp:

```bash
cd demo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py   # opens http://127.0.0.1:7860
```

Three tabs:

- **Reverse** — type (or randomize) an input sequence; see the prediction and per-head attention heatmaps live. Correct predictions show a sharp anti-diagonal.
- **Attention 101** — edit Q, K, V matrices by hand (3 tokens × 2 dims) and watch the scores, softmax weights, and output update. Nudge one query toward a key and see the weight concentrate.
- **Parity** — runs `verify.py`'s checks in the browser against the live PyTorch install.

## Discussion

**Why the parity test is the real milestone.** Attention has enough moving pieces (Q/K/V split, head reshape, scaling, softmax, output projection) that any one of them can be silently wrong and still produce plausible-looking outputs. Matching PyTorch bit-for-bit — forward *and* backward — at the same weights is the strongest possible statement that the math is right.

**Why this task generalizes.** Attention is the mechanism that lets any output position gather information from any input position *based on content, not just position*. The reverse task exercises the positional half (routing based on `pos(query) + pos(key)`); content-based routing happens naturally once you stack attention on real data (lab 1.4, nanoGPT) or cross-attention on conditioning (lab 3.3, text-to-image).

**What changes in a real transformer block.** You add: a residual connection around attention, a LayerNorm before it (pre-norm transformers) or after it (post-norm), an MLP sublayer with its own residual and norm. Each is a small modification — none of them changes the attention kernel itself. That's why getting it right here, in isolation, pays off for every subsequent lab.

**Why DiT needs the exact same kernel.** DiT operates on flattened image patches as tokens. The attention it runs is scaled dot-product multi-head self-attention over those patch tokens — identical to what you implemented here. What DiT adds is AdaLN-Zero conditioning (lab 1.1's MLP-from-time-embedding pattern modulating the norm layers), not anything about attention itself.
