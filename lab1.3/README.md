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

**What RNN seq2seq did badly.** BLEU dropped sharply as source sentences grew longer. The whole source — 5 words or 50 — had to pass through a single fixed-size hidden vector. For long sentences, earlier tokens got overwritten as the encoder kept reading.

**How attention fixed it (Bahdanau 2014).** Instead of a single handoff vector, keep **all** the encoder hidden states — one per source token. At each decoding step, compute a similarity score between the decoder's current state and every source state, softmax those into weights, and take a weighted average of the source states. Feed that weighted average into the decoder alongside its normal input. The decoder now reads a *different* mix of source tokens at every output step, dynamically chosen based on what it is producing right now — the "attention" metaphor. Practical effect: BLEU on long sentences stopped degrading, and the learned attention weights, when plotted, showed clean **soft alignments** between source and target words (French *"la maison bleue"* attending to English *"the blue house"* with the expected adjective-noun reordering). First time neural MT had interpretable internals, and quality no longer fell apart on long sentences.

**The Transformer (Vaswani 2017).** If attention alone can route information between positions, why keep the RNN at all? Stack pure attention layers with positional embeddings and residuals. Call it the **Transformer**. Win WMT English-German and English-French outright, and train substantially faster because attention parallelizes across positions in a way RNNs fundamentally cannot. Attention is no longer a bridge between two RNNs — it is the **only** mechanism that moves information between positions. That is the architecture this lab builds.

**What attention does well, in one list.** Handles arbitrarily long dependencies without degradation (every position can reach every other in one hop). Parallelizes across positions on GPUs (one matmul, not N sequential steps). Produces interpretable routing patterns you can visualize. Generalizes to any modality that can be tokenized — translation was just the first domain. After 2017, the same kernel went on to power language modeling (GPT, BERT), vision (ViT), protein folding (AlphaFold), audio (Whisper), and image diffusion (DiT, the endpoint of this curriculum). The kernel stayed the same; only the inputs changed.

## Limitations of previous RNN-based approaches

**What an RNN is, in one line.** Process a sequence one token at a time, carrying a hidden state forward that summarizes everything seen so far: `h_t = f(h_{t-1}, x_t)`. Output at each step comes from `h_t`. Think of it as a stream processor with a fixed-size state variable.

**Three things RNNs do badly that attention fixes:**

1. **Sequential compute.** You cannot compute `h_5` until you have `h_4`. A length-1000 sequence means 1000 sequential steps — GPUs hate this, parallelism across positions is nearly impossible. Attention computes all positions in parallel via one big matmul.
2. **Single-vector bottleneck.** The entire past is compressed into one fixed-size `h_t`, so long-range info gets overwritten as new tokens arrive. Attention keeps every past position available and lets any position reach back to any other directly.
3. **Vanishing / exploding gradients over long distances.** Gradients flowing backward through many RNN steps shrink (or explode) multiplicatively; dependencies more than ~50 tokens apart are hard to learn. In attention, any two positions are one matmul away — the gradient path between them is effectively distance-1.

## The intuition behind attention

**Attention as a soft dictionary lookup.** Every position gets three projections of itself: a **query** ("what am I looking for?"), a **key** ("this is what I offer to anyone looking for me"), and a **value** ("this is the content I'll hand over if chosen"). A hard dictionary returns one value for an exact-match key. Attention softens this: compare a query against *every* key, softmax the similarity scores into weights, and return the weighted average of values. The output at each position is a blend of everyone's values, with weights decided by query-key similarities. Because the weights are recomputed from the inputs every forward pass, the routing is **data-dependent** — two different inputs produce two different attention matrices. Unlike a fixed weight matrix, attention does not learn "always combine position 3 with position 7"; it learns *how to decide* which positions to combine based on what is currently there.

**A concrete example (translation).** Translating "the blue house" → "la maison bleue", suppose the decoder is about to emit the target word at output position 2 — which should be "bleue". This is **cross-attention**: the decoder's current state is projected into a **query** encoding "what source information do I need right now." Each of the three source words has emitted a **key** — a compact representation of what it can offer. The query dots against all three keys; training has arranged for it to score highest against "blue"'s key. Softmax over the three scores gives weights like `[0.02, 0.95, 0.03]` over ("the", "blue", "house"). Each source word's **value** is a vector carrying its meaning. The weighted sum is mostly the value of "blue" — and that is the source information the decoder combines with its own state to emit "bleue" via softmax over the French vocabulary. Repeat at every decode step; the full matrix of weights across source and target positions is a **soft alignment** map — Bahdanau's 2014 result, the first time neural translation's internals became interpretable.

**Why position matters — and how it gets in.** Attention by itself is **permutation-equivariant**: permute the input tokens and the outputs permute the same way. It has no built-in notion of "this token is at position 3." For some tasks (classifying a bag of features) that is fine; for anything where order encodes meaning — language, reversing a sequence — it is fatal. The fix is **positional embeddings**: before the input reaches attention, each token's vector has a position-specific vector added to it. After that addition, Q and K can read both "what I am" and "where I am" from the same vector, so similarity scores can depend on position as well as content. In `AttentionModel`, `self.pos_embed` does exactly this — a learned lookup from position index to vector, added to the token embedding. For the reverse task position is the *entire* signal (tokens are arbitrary integers); the learned attention pattern is purely positional. For translation, position matters for syntax ("dog bites man" ≠ "man bites dog") but token content matters too — real attention routes on both.

**How a decoder uses attention (the translation example).** Worth pinning down because this is where the mechanism got its name. In encoder-decoder NMT, attention is **cross-attention** — queries come from the decoder, keys and values from the encoder. (This lab builds **self-attention**, where Q/K/V all come from the same sequence. The math is identical; only the source of the tensors changes.) The decoding procedure:

1. The encoder runs over the full source sentence once, producing one hidden state per source token.
2. The decoder generates the target **one token at a time, left to right**. The decoder chooses target *positions* simply by being a sequential loop: step 1, step 2, step 3, … "Where am I writing now" is not an attention decision — the decoder already knows, it is on step `t`.
3. At each decode step, the decoder's current state becomes a query against the encoder's keys and values. Attention returns a weighted average of source representations: "for the word I am about to emit, here is the most relevant source information right now."
4. The decoder combines that weighted source representation with its own state and emits a token via softmax over the target vocabulary.

So in encoder-decoder NMT: the decoder decides *output position* (sequentially by loop) and *output token* (softmax over vocab). Attention decides **which source positions to read when generating the current output position**. For "the blue house" → "la maison bleue": emitting *"maison"* concentrates attention on "house"; emitting *"bleue"* concentrates on "blue"; the adjective-noun reordering falls out of the alignment attention learns.

## What the model learns

**The reverse task.** Given a random integer sequence of length `L`, output the reversed sequence. That's it — no language, no real data, just `[a, b, c, d] → [d, c, b, a]`.

Why reverse and not copy: copy is solved by `attn = I` (each position attends to itself), which is trivially achievable by the positional embedding alone and makes attention look boring. Reverse requires `attn[i, j] = 1 if j == L-1-i else 0` — an anti-diagonal — which forces the query at position i to use positional information to pick out the key at position `L-1-i`. Mechanically interesting, visually striking.

```
 input tokens                    logits over vocab
(B, L)  →  [embed + pos]  →  [MultiHeadAttention]  →  [Linear head]  →  (B, L, vocab)
```

The model has exactly **one** attention layer. No MLP. No LayerNorm. No residual. If it reaches high accuracy, attention alone is doing the work.

**What Q, K, V each learn.** Given an input sequence of length `L`, at each output position `i`: the query `Q[i]` encodes "I am at position `i`, looking for position `L-1-i`"; each key `K[j]` encodes "I am at position `j`"; each value `V[j]` carries the token originally at input position `j`. The dot product `Q[i] · K[j]` peaks at `j = L-1-i`, so softmax concentrates the attention weight there. The weighted sum is ~100% of `V[L-1-i]`, and the linear head reads that vector to predict the token that was at position `L-1-i`. Q and K only need to encode position; only V has to carry token identity — that division of labor is emergent from training, not hardcoded.

**What multi-head attention buys you.** A single attention layer expresses only one routing pattern at a time. Multi-head runs `H` parallel attentions on the same input — `H` independent query/key/value projections, `H` outputs, all computed in parallel with no cross-talk. Those `H` outputs are then **concatenated** (not selected): every head's contribution survives into the final output. The question is just *with what weight* each head contributes to each output dimension, and that is what the **output projection** (`out_proj` in code) learns. Think of it as consulting `H` independent experts who each look at the input from their own angle; the output projection is the learned blend across their summaries. The reverse task only needs one routing pattern, so multiple heads typically converge to variants of the same anti-diagonal (you will see this in `visualize.py`); in a deep transformer on richer tasks, heads specialize on different relationships (syntax, coreference, long-range structure) — that emergent specialization is where multi-head pays off.

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

`attention.py` implements a drop-in equivalent of `torch.nn.MultiheadAttention`: same scaled dot-product kernel, same Q/K/V split into heads, same output projection, and the same parameter layout (`in_proj_weight`, `in_proj_bias`, `out_proj.weight`, `out_proj.bias`) — so weights copy 1-to-1 between the two. The same module handles both regimes used in this course: unmasked self-attention (the reverse task here, and DiT's patch-token attention later) and causal-masked self-attention (lab 1.4 / nanoGPT).

`verify.py` copies a fresh set of weights from `torch.nn.MultiheadAttention` into ours and checks that the two produce identical results — both directions:

- **Forward** — outputs and per-head attention weights match for unmasked and causal cases, plus the causal upper-triangle is exactly zero.
- **Backward** — gradients w.r.t. the input match. This is the parity that matters for training: identical gradients mean an optimizer step produces the same weight update in both implementations, so anything you train against `attention.py` would have trained the same way against PyTorch's built-in.

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

If forward outputs match at the same weights and backward gradients match at the same inputs, the math is right — both at inference time and across training updates.

### 3. How scaled dot-product attention works

Open `attention.py` and read the `scaled_dot_product_attention` docstring. The whole operation is three lines:

```python
scores = Q @ Kᵀ / sqrt(d_k)     # (L_q, L_k) — each row is "query i's similarity to all keys"
attn   = softmax(scores, -1)     # row-normalize: each query distributes attention probability 1.0
out    = attn @ V                # each output row is a convex combination of value rows
```

**The one-line summary.** `attn` is a matrix of weights between `L_q` queries and `L_k` keys, where each row sums to 1 (softmax across keys). It's used as the weights in a weighted average of the value vectors — that average is the output. Everything else in the kernel is just *how* those weights are computed: from scaled query-key similarity.

**What `D` is.** `D = embed_dim` is the shared width of input token features, query / key / value vectors, and the attention output. Each token enters as a `D`-dim vector, gets projected into `D`-dim Q/K/V, and leaves as a `D`-dim attended vector. Bigger `D` = richer per-token features and more capacity in each weighted-average output.

**Why `sqrt(d_k)`?** The variance of `q · k` grows linearly in `d_k`. Without scaling, softmax saturates near a one-hot vector and its gradient vanishes. Dividing by `sqrt(d_k)` keeps the pre-softmax logits roughly unit-variance at initialization.

**Why multi-head?** Multi-head attention divides the `D` features into `H` groups of width `d_head = D/H`, and each head runs attention on **only its own partition** — its own Q/K/V slice, its own `(L_q, L_k)` weight matrix, its own weighted average. The `H` outputs are concatenated back to width `D` and linearly mixed by `out_proj`. Same parameters, same compute, but `H` independent routing patterns over `H` independent feature subspaces.

The intuition for *why* this helps: a dot product of two width-`D` vectors collapses `D` features into a single number — one verdict on "do these two tokens match?" If features 0–15 say "syntactically yes" but features 16–31 say "semantically no," the contributions cancel inside the sum and the score is mush. With one head, the single `(L_q, L_k)` matrix has to compromise across all the different kinds of similarity the features encode. Multi-head lets you compute `H` different similarity scores per `(q, k)` pair, each measuring a different *axis* of similarity, and route a different channel of values per score — sharp specialized routing instead of one generalist matrix.

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
