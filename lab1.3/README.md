# Module 1.3 — Attention (Standalone)

> Part 1 — Building Blocks · [DiT from Scratch](../README.md)

**Goal**: implement scaled dot-product attention and multi-head attention from scratch, prove it matches `torch.nn.MultiheadAttention` numerically, then train an attention-only model on the reverse task to see attention actually learn.

**Why this matters for DiT**: every DiT block is a self-attention layer plus an MLP. The kernel you write here — `softmax(QKᵀ/√d) · V`, split across heads — is the exact operation DiT uses on its patch tokens. Getting it right once in isolation, with no transformer residuals or AdaLN in the way, means you never have to re-derive it later.

**Deliverables**:
- `attention.py` with `scaled_dot_product_attention` and `MultiHeadAttention` — parameter layout matches PyTorch's built-in so weights can be copied 1-to-1.
- `verify.py` confirming bit-for-bit parity (forward, causal-masked, and backward) against `torch.nn.MultiheadAttention`.
- `train.py` training an attention-only model on the reverse task to ≥99% per-token accuracy.
- `visualize.py` producing an attention heatmap showing the learned anti-diagonal pattern.

## What attention was invented for

**The machine translation landscape circa 2014.** The canonical benchmarks were the **WMT shared tasks** (Workshop on Machine Translation) — an annual competition with standard parallel corpora and test sets, primarily **English↔French** and **English↔German**, later expanded to Chinese, Russian, Czech, and others. Systems were ranked by **BLEU**, a metric comparing predicted n-gram overlap against human reference translations. WMT test sets and BLEU numbers were the de facto leaderboard the field optimized against.

**What RNN seq2seq solved (2014).** Sutskever et al. ("Sequence to Sequence Learning with Neural Networks") and Cho et al. ("Learning Phrase Representations using RNN Encoder-Decoder") showed you could do translation with one neural network trained end-to-end: a recurrent encoder reads the source sentence token-by-token into a final hidden vector, a recurrent decoder starts from that vector and generates the target sentence token-by-token. Train on parallel corpora with cross-entropy. Sutskever's system hit competitive BLEU on WMT English→French. The paradigm — **neural machine translation** — was born here.

**What RNN seq2seq did badly.** BLEU dropped sharply as source sentences grew longer. The whole source — 5 words or 50 — had to pass through a single fixed-size hidden vector. For long sentences, earlier tokens got overwritten as the encoder kept reading. Sutskever's workaround was to **reverse the source sentence** before encoding, putting recent source tokens near the decoder's starting state — a hack, not a fix. (The mechanical reasons RNNs struggle on long sequences are in the next section.)

**How attention fixed it (Bahdanau 2014).** Instead of a single handoff vector, keep **all** the encoder hidden states — one per source token. At each decoding step, compute a similarity score between the decoder's current state and every source state, softmax those into weights, and take a weighted average of the source states. Feed that weighted average into the decoder alongside its normal input. The decoder now reads a *different* mix of source tokens at every output step, dynamically chosen based on what it is producing right now — the "attention" metaphor. Practical effect: BLEU on long sentences stopped degrading, and the learned attention weights, when plotted, showed clean **soft alignments** between source and target words (French *"la maison bleue"* attending to English *"the blue house"* with the expected adjective-noun reordering). First time neural MT had interpretable internals, and quality no longer fell apart on long sentences.

**The Transformer (Vaswani 2017).** If attention alone can route information between positions, why keep the RNN at all? Stack pure attention layers with positional embeddings and residuals. Call it the **Transformer**. Win WMT English-German and English-French outright, and train substantially faster because attention parallelizes across positions in a way RNNs fundamentally cannot. Attention is no longer a bridge between two RNNs — it is the **only** mechanism that moves information between positions. That is the architecture this lab builds.

**What attention does well, in one list.** Handles arbitrarily long dependencies without degradation (every position can reach every other in one hop). Parallelizes across positions on GPUs (one matmul, not N sequential steps). Produces interpretable routing patterns you can visualize. Generalizes to any modality that can be tokenized — translation was just the first domain. After 2017, the same kernel went on to power language modeling (GPT, BERT), vision (ViT), protein folding (AlphaFold), audio (Whisper), and image diffusion (DiT, the endpoint of this curriculum). The kernel stayed the same; only the inputs changed.

## Limitations of previous RNN-based approaches

**What an RNN is, in one line.** Process a sequence one token at a time, carrying a hidden state forward that summarizes everything seen so far: `h_t = f(h_{t-1}, x_t)`. Output at each step comes from `h_t`. Think of it as a stream processor with a fixed-size state variable.

**Three things RNNs do badly that attention fixes:**

1. **Sequential compute.** You cannot compute `h_5` until you have `h_4`. A length-1000 sequence means 1000 sequential steps — GPUs hate this, parallelism across positions is nearly impossible. Attention computes all positions in parallel via one big matmul.
2. **Single-vector bottleneck.** The entire past is compressed into one fixed-size `h_t`, so long-range info gets overwritten as new tokens arrive. Attention keeps every past position available and lets any position reach back to any other directly.
3. **Vanishing / exploding gradients over long distances.** Gradients flowing backward through many RNN steps shrink (or explode) multiplicatively; dependencies more than ~50 tokens apart are hard to learn. In attention, any two positions are one matmul away — the gradient path between them is effectively distance-1.

LSTM/GRU gates, teacher forcing, and backprop-through-time are irrelevant to the DiT path. Where RNN intuition *does* pay off later: state-space models like **Mamba** and **S4** revive RNN-like ideas with modern tricks to fix the three problems above. Not on this curriculum's path — a bookmark, not a prerequisite.

## What "routing information" means

"Information" here means the numerical content at each position of the activation tensor — at the start, position `i`'s vector carries info about the token at position `i`. **Routing** means deciding which positions' content flows into which other positions. Different layers do this differently:

| Layer | How positions mix | Routing is… |
|---|---|---|
| Convolution | pixel mixes with its fixed local neighborhood | **static** — same kernel everywhere, baked in at train time |
| Fully-connected across positions | every position to every position via one weight matrix | **static** — fixed weights, same for every input |
| Attention | every position can read from every other position | **dynamic** — weights are *computed from the inputs themselves*, every forward pass |

That last row is the qualitative leap. In a CNN, "combine pixel (3,5) with pixel (4,5)" is a fact about the architecture. In attention, "combine position 3 with position 7" is a decision the network makes on the fly based on what's currently at positions 3 and 7. Two different inputs produce two different attention matrices. This is called **content-dependent** (or **data-dependent**) routing, and it is why the same attention kernel works on text, image patches, audio frames, and anything else you can tokenize — the routing is not hardcoded for a modality.

In this lab the tokens are meaningless integers, so attention routes on **position** rather than token content: Q and K are built from `pos_embed`, producing similarity peaks at `j = L-1-i`. In a language model it routes on both. The mechanism is identical.

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
