# Module 1.4 — Transformer (nanoGPT)

> Part 1 — Building Blocks · [DiT from Scratch](../README.md)

**Goal**: stack causal self-attention with MLPs, layer norms, and residuals to get a real decoder-only transformer. Train it as a character-level language model on TinyShakespeare and generate plausible Shakespeare from a prompt. Now that you've proved the attention kernel from-scratch in lab 1.3, this lab uses PyTorch's built-in `F.scaled_dot_product_attention` directly — same math, faster wrapper.

**Why this matters for DiT**: a DiT block is structurally identical to a GPT block — multi-head attention plus an MLP, both wrapped in pre-norm + residual. The only DiT-specific addition is **AdaLN-Zero conditioning** (covered in lab 3.1).

**Deliverables**:
- `gpt.py` — `CausalSelfAttention` (using `F.scaled_dot_product_attention(..., is_causal=True)`), `Block` (LN → attn → +res, LN → MLP → +res), `GPT` (embed → N blocks → final LN → linear head), and a `generate` method.
- `train.py` — auto-downloads TinyShakespeare, trains char-level GPT, periodically samples from the model so you can watch the text get more coherent over training.
- `sample.py` — load a checkpoint and generate text from a prompt with adjustable temperature and top-k.
- `demo/app.py` — Gradio webapp for interactive sampling.

## Compared with nanoGPT

Karpathy's nanoGPT ships three dataset configs (`data/<name>/prepare.py`). This lab matches the smallest one — char-level on TinyShakespeare, the toy/debug option. The other two are BPE-tokenized at progressively larger scales:

| Name | Tokenizer | Dataset | Size | Purpose |
| --- | --- | --- | --- | --- |
| **`shakespeare_char`** | char-level | TinyShakespeare | 1 MB | toy/debug — **what this lab matches** |
| `shakespeare` | tiktoken (GPT-2 BPE) | TinyShakespeare | 1 MB | smallest BPE example, useful for fine-tuning |
| `openwebtext` | tiktoken (GPT-2 BPE) | OpenWebText | ~40 GB raw, ~9 B tokens | headline GPT-2 124M reproduction |

The same `gpt.py` architecture handles all three — they differ only in tokenizer, dataset, and training scale. Once you've trained the char-level version here, scaling up to BPE + a bigger corpus is a swap of the data-prep step, not the model.

## What the model learns

**TinyShakespeare** is ~1MB of plays from the First Folio, concatenated into one file. ~65 unique characters (uppercase + lowercase + punctuation + newline). At char-level there's no tokenizer to install — every character is a token, and the vocab is just `sorted(set(text))`.

The first few lines of `data/input.txt` give a sense of what the model is fitting:

```text
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.
```

That format — `SPEAKER:\nLINE OF DIALOGUE\n\n` — repeats throughout the file, which is why a trained model produces text that *looks* like a play even though nothing in the architecture knows what a play is.

The model is **autoregressive**: it generates one token at a time, and each new token is conditioned on every token that came before it (the prompt *and* the model's own previous outputs). Given the prompt `ROMEO:` it predicts the next character, appends it, predicts the next, and so on — feeding its growing output back in as context at every step. Sampling from `"ROMEO:"` at each stage looks roughly like this (representative — your run will differ):

```text
val_loss ≈ 4.2   (random init — uniform over 65 chars)
  ROMEO:qX;3J!~Mz $vkP'fY:9.gNB2hcW  *Lr&[7

val_loss ≈ 3.0   (unigram — right letter/space frequencies, no structure)
  ROMEO:e ot a thnoiea itre oi e htra
  es ohn oea m hntoe saiet ondoer thi

val_loss ≈ 2.0   (bigrams + spacing — fake words with real word shapes)
  ROMEO: the wast ond hion of fas ther
  whe and the sice ar mest of houre an

val_loss ≈ 1.5   (real English words; learned the "SPEAKER:\n..." format)
  ROMEO:
  I will the hand of the king,
  And speak my heart to be the lord.

val_loss ≈ 1.3   (Shakespeare-flavored phrasing, turn-taking dialogue)
  ROMEO:
  Wherefore art thou so heavy, gentle friend?
  I pray thee, speak no more of this sad hour.

  JULIET:
  Nay, my good lord, mine eyes do weep for thee.
```

Notice the format itself is *learned*, not hard-coded: at val_loss ≈ 1.5 the model has figured out from the data that a speaker label is followed by a newline and then dialogue, and that a blank line separates speakers. Nothing in the training loop tells it this — it's just what minimizes next-character cross-entropy on the corpus.

Reaching val loss ≈ 1.3 in 5000 steps on the default config typically gives Shakespeare-flavored output: readable English, recognizable rhythm, but no semantic content (the model has no idea what it's saying). The training goal isn't to produce *correct* Shakespeare — it's to learn the **distribution** of the corpus well enough that samples are indistinguishable in style. That same objective, applied to images instead of characters, is exactly what DiT is doing in lab 3.

## Files

| File | What it is |
| --- | --- |
| `gpt.py` | `CausalSelfAttention`, `MLP`, `Block`, `GPT` (with weight tying + GPT-2 init). Attention via `F.scaled_dot_product_attention` with `is_causal=True` |
| `train.py` | Downloads TinyShakespeare, trains with AdamW + cosine schedule + grad clip, prints samples every 1000 steps |
| `sample.py` | Loads checkpoint, generates from a prompt with temperature and top-k |
| `demo/app.py` | Gradio webapp — type a prompt, see the completion |

## Instructions

### 1. Set up

Python 3.9+. From `lab1.4/`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Training auto-selects device: CUDA → Apple MPS → CPU. The dataset auto-downloads on first run (~1MB).

### 2. Use PyTorch's built-in attention kernel

In lab 1.3 you wrote `scaled_dot_product_attention` from scratch and proved it bit-for-bit equivalent to `torch.nn.MultiheadAttention` (forward, causal-masked, *and* backward). Now that the math is yours, there's no reason to keep using your own implementation in this lab — `gpt.py` calls PyTorch's built-in directly:

```python
out = F.scaled_dot_product_attention(
    q, k, v,
    dropout_p=self.dropout_p if self.training else 0.0,
    is_causal=True,
)
```

Two things this gets you:

- **Speed.** PyTorch's built-in dispatches to **Flash Attention** on supported hardware (CUDA, MPS) — fused softmax+matmul that runs in less memory and faster than a naïve unfused implementation. Same mathematical result, much better wall-clock.
- **Less plumbing.** `is_causal=True` applies the upper-triangular mask inside the kernel, with no need to materialize an `(L, L)` boolean tensor or pass it through every block.

### 3. Block structure

Each `Block` is **pre-norm**: LayerNorm before each sublayer, residual added after.

```
x ─┬─► LN ─► MHA ─┐ ┌─► LN ─► MLP ─┐
   │              ▼ │              ▼
   └─────────────►⊕─┴─────────────►⊕─► out
```

**Why pre-norm.** LayerNorm before each sublayer (instead of after) keeps an unnormalized identity path through every block. Gradients flow back through the residual at every depth without vanishing, so deep stacks train stably. Post-norm needs careful warmup to do the same — almost all modern transformers (GPT-2, LLaMA, DiT) use pre-norm.

**Why an MLP after attention.** Attention is a *linear* weighted average of value vectors. Stacking N attention layers without MLPs collapses to one big linear op (linear ∘ linear = linear). The MLP — `Linear → GELU → Linear` with a 4× hidden expansion — is what makes the network actually deep. Each block's MLP non-linearly transforms the attention output so the next block has something new to route over.

### 4. Train

```bash
python train.py
```

Default: 6 layers, 6 heads, embedding dim 384, block size 256, batch size 64, 5000 steps. ~10M parameters. ~5–10 minutes on M3 MPS to val loss ≈ 1.3.

Per-eval log:

```
step    1  |    0.4s  |  lr 3.00e-06  |  train 4.1719  val 4.1716
step  500  |   55.0s  |  lr 3.00e-04  |  train 2.1140  val 2.1241
step 1000  |  108.4s  |  lr 2.91e-04  |  train 1.6531  val 1.6622
step 2000  |  219.1s  |  lr 2.50e-04  |  train 1.4123  val 1.4245
step 5000  |  555.6s  |  lr 3.00e-05  |  train 1.2890  val 1.3076
```

Every 1000 steps the script samples 200 chars from the model so you can literally watch the text get more coherent — early on it's gibberish punctuation, later you see proper Shakespeare formatting and dialogue.

The training recipe is standard for small transformers:
- **AdamW** with `betas=(0.9, 0.95)`, `weight_decay=0.1` — `0.95` for the second moment is gentler on small batches than the default `0.999`.
- **Cosine LR schedule with warmup** — 100-step linear warmup from 0, then cosine decay from `3e-4` to `3e-5` over the remaining steps.
- **Gradient clipping at 1.0** — prevents the rare large-loss batch from blowing up the parameters.
- **Dropout 0.1** — applied inside each MLP and after attention; small but visibly closes the train-val gap.

### 5. Sample

After training:

```bash
python sample.py --prompt "ROMEO:"
python sample.py --prompt "JULIET:" --temperature 0.6 --top-k 40
python sample.py --prompt "What" --temperature 1.5    # spicier, more typos
```

Knobs:
- **`--temperature`** — divides the logits before softmax. Low (0.5) makes the distribution sharper: more confident, more repetitive. High (1.5) flattens it: more diverse, more typos.
- **`--top-k`** — keep only the top-k most likely chars at each step, set the rest to `-inf`. Cuts the long tail of unlikely tokens; usually improves quality at any temperature.

### 6. Interactive demo

```bash
cd demo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py   # opens http://127.0.0.1:7860
```

Type a prompt and play with the sliders. Watch how temperature changes the output's character — at 0.3 the model loops on common phrases; at 1.5 it invents new pseudo-Elizabethan words.

## Study the source code

Two ideas hold up the whole lab — the **transformer block** and the **sampling loop**. Both live in `gpt.py`. Read them in this order.

### The transformer block

Build it from the inside out. Each piece below assumes the one above.

**`CausalSelfAttention`** — Compare to your hand-rolled version in lab 1.3 — the math is identical; the fused kernel is just faster and skips materializing the `(L, L)` causal mask.

```python
out = F.scaled_dot_product_attention(
    q, k, v,
    dropout_p=self.dropout_p if self.training else 0.0,
    is_causal=True,
)
```

**`MLP`** — `Linear(D, 4D) → GELU → Linear(4D, D) → Dropout`, applied independently at every sequence position:

```python
def forward(self, x):
    return self.dropout(self.fc2(F.gelu(self.fc1(x))))
```

Attention is a *linear* weighted average of value vectors — without an MLP between attention layers, the whole stack collapses to one big linear op (linear ∘ linear = linear). The MLP's GELU is what makes the network actually deep.

#### The pre-norm sublayer pattern

Before `Block` puts these two sublayers together, two more primitives are doing all the wiring: **LayerNorm** before each sublayer, and a **residual connection** around it.

**LayerNorm.** Normalizes the activation vector at each position to zero mean and unit variance, then applies a learnable affine. For a single position's vector $x \in \mathbb{R}^D$:

$$
\mu = \frac{1}{D}\sum_{i=1}^{D} x_i
\qquad
\sigma^2 = \frac{1}{D}\sum_{i=1}^{D} (x_i - \mu)^2
$$

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
\qquad
\text{LN}(x)_i = \gamma_i \, \hat{x}_i + \beta_i
$$

where $\gamma, \beta \in \mathbb{R}^D$ are learned per-feature scale and shift, and $\epsilon$ is a small constant (default `1e-5`) for numerical stability.

In `gpt.py` this is one line — `self.ln1 = nn.LayerNorm(embed_dim)`.

**Residual connection.** A straight-through identity path that the sublayer adds to:

```python
x = x + sublayer(x)   # not x = sublayer(x)
```

The output is whatever the sublayer learned to *add* to `x`, not a replacement. Two consequences:
- **Gradients always have an identity path.** `∂(x + f(x))/∂x = I + ∂f/∂x`, so gradients reach the input even when the sublayer's Jacobian is small or saturated. This is what makes >100-layer networks trainable.
- **Sublayers can learn to do nothing.** If a particular block isn't useful, its sublayer outputs near zero and the residual just passes the input through. No layer is ever "in the way" — layers only contribute when they help.

The two compose into the **pre-norm sublayer pattern** that shows up everywhere in modern transformers:

```python
x = x + sublayer(LayerNorm(x))
```

LayerNorm gives the sublayer a clean input; the residual preserves the original. That's the recipe for *both* halves of `Block`.

**`Block`** — pre-norm wiring of the two sublayers:

```python
x = x + self.attn(self.ln1(x))   # residual + (LN → attention)
x = x + self.mlp(self.ln2(x))    # residual + (LN → MLP)
```

That's the entire block.

```
x ─┬─► LN ─► MHA ─┐ ┌─► LN ─► MLP ─┐
   │              ▼ │              ▼
   └─────────────►⊕─┴─────────────►⊕─► out
```

Two properties worth pausing on:
- **The residual path is never normalized.** LayerNorm sits *before* each sublayer, not on the skip connection — so an unnormalized identity flows through every block, and gradients reach early layers without vanishing.
- **The block doesn't know it's causal.** The mask lives inside `CausalSelfAttention` via `is_causal=True`. Swap the attention sublayer for a bidirectional one and you have the ViT/DiT block — same shape, no other changes needed.

**`GPT`** — `embed → dropout → N × Block → final LN → linear head`. Three details worth knowing:

- **Weight tying.** The output head's row for token `t` is the *same vector* as the embedding for token `t`:

  ```python
  self.head = nn.Linear(n_embd, vocab_size, bias=False)
  self.head.weight = self.token_embed.weight   # tie input embed and output head
  ```

  Saves `vocab × n_embd` parameters and improves perplexity in nearly every published comparison.

- **GPT-2 residual init scaling.** Projections that feed the residual stream (`c_proj` in attention, `fc2` in MLP) are re-initialized with std `0.02 / √(2·n_layer)` so activation variance stays flat through depth:

  ```python
  for p_name, p in self.named_parameters():
      if p_name.endswith("c_proj.weight") or p_name.endswith("fc2.weight"):
          nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))
  ```

- **`forward`** returns logits during inference, and `(logits, loss)` during training:

  ```python
  if targets is None:
      return logits, None
  loss = F.cross_entropy(
      logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
  )
  return logits, loss
  ```

  Same forward pass, same logits; only the loss head is conditional.

### The sampling loop

`GPT.generate` is the entire inference algorithm:

```python
# idx is the running token sequence. On entry it's the prompt; every
# iteration appends one freshly sampled token via torch.cat, so L grows
# by 1 each step. After max_new_tokens iterations idx has length
# prompt_len + max_new_tokens and is what we return.
for _ in range(max_new_tokens):
    # idx_cond is idx truncated to the last block_size tokens — the
    # actual input to the model this step. Anything older falls off
    # the front of the context window.
    idx_cond = idx[:, -self.block_size:]
    # In PyTorch, calling a module instance (self(x)) invokes
    # nn.Module.__call__, which runs hooks then calls self.forward(x).
    logits, _ = self(idx_cond)
    # The logits at position i are the model's prediction for the token
    # at position i+1. Positions 0..L-2 predict tokens we already have
    # — useful for computing training loss, useless for generation.
    # The model just did a full forward pass over L positions but we
    # throw away L-1 of them. Production code would use a KV cache so
    # each step costs O(L) instead of O(L²).
    #
    # Temperature rescales the logits before softmax:
    #   T → 0:  distribution → one-hot on argmax (greedy/deterministic)
    #   T = 1:  the model's raw distribution
    #   T → ∞:  logits → 0, distribution → uniform (pure random)
    logits = logits[:, -1, :] / temperature
    # Top-k: keep the k highest-scoring tokens, set the rest to -inf
    # so they get probability 0 after softmax. Cuts the long tail of
    # unlikely candidates so a bad random draw can't derail generation.
    #
    # Walk-through with V=8, k=3:
    #   logits = [ 5.0, -1.0,  8.0,  2.0,  9.0,  0.0,  6.0,  3.0]
    #   top 3   = [9.0, 8.0, 6.0]            (v, sorted descending)
    #   cutoff  = 6.0                         (v[:, [-1]], the k-th)
    #   logits = [-inf, -inf,  8.0, -inf,  9.0, -inf,  6.0, -inf]
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("inf")
    probs = F.softmax(logits, dim=-1)
    # The random draw is what gives generation diversity; argmax here
    # would loop on common phrases. Temperature + top-k above shape
    # the distribution; this is the unbiased dice roll at the end.
    next_token = torch.multinomial(probs, num_samples=1)
    idx = torch.cat([idx, next_token], dim=1)
return idx
```

One iteration, end to end:
1. **Crop** the running context to the last `block_size` tokens — the model can't attend further back than that, so feeding more is wasted compute.
2. **Forward** runs all N blocks over the prefix and returns logits of shape `(B, L, vocab)`. Only the *last position*'s logits matter — they're the model's distribution over the next character.
3. **Temperature** divides the logits before softmax. Lower → sharper → more confident, more repetitive. Higher → flatter → more diverse, more typos.
4. **Top-k** keeps only the `k` highest-logit candidates and sets the rest to `-inf`. Trims the long tail of unlikely tokens; usually improves sample quality at any temperature.
5. **Multinomial** draws one token according to the post-softmax probabilities. (Use `argmax` instead for greedy/deterministic decoding.)
6. **Append** the sampled token to the running context and loop. Each step's input is the previous step's output — that's what makes generation *autoregressive*.

This loop is `O(L²)` per generated token because the full prefix is re-fed to every block on every step. Production code caches each block's per-step K/V tensors so each new step is `O(L)` — but the math is identical; KV caching just memoizes values that weren't going to change.

### Suggested exercises

- **Disable weight tying.** Comment out `self.head.weight = self.token_embed.weight` in `GPT.__init__` and retrain. Compare final val loss and parameter count.
- **Swap to post-norm.** Move the LayerNorms in `Block.forward` to *after* each sublayer. Watch the loss curve — much less stable, may not converge without warmup tuning.
- **Print attention weights.** Replace the fused call with the unfused formulation `(Q @ Kᵀ / √D).softmax(-1) @ V` so you can return the `(L, L)` attention map. Visualize for a generated sample — late-layer heads should concentrate on a few semantically relevant past positions.
- **Greedy decoding.** Replace `multinomial` with `argmax` in `generate`. Output becomes deterministic per prompt — and usually noticeably worse, looping on common phrases.
- **Add KV caching.** Cache `k` and `v` inside `CausalSelfAttention` across generation steps so each new token costs `O(L)` instead of `O(L²)`. Measure the speedup at `max_new_tokens=2000`.

## Discussion

**Why this is "a transformer block," not just attention.** The composable unit of every modern LLM (and DiT) is the *block*: pre-norm attention + pre-norm MLP, both residual. Once you have one block, you stack N of them — that's the whole architecture. Everything in DiT, ViT, GPT, BERT, and LLaMA is variations on what each sublayer does to the per-token residual stream.

**Why next-char prediction is the right objective for self-supervised learning on text.** No labels needed — the input *is* the target, shifted by one. Every character of the corpus is a training signal. The model that minimizes this loss has learned the joint distribution of text, and the same loss objective then trivially gives generation: just sample the predicted distribution one token at a time.

**Why weight tying.** The output head's row for token `t` and the input embedding for token `t` should both be "the direction in feature space that means token `t`." Forcing them to share parameters cuts `vocab_size × n_embd` parameters and improves perplexity in nearly every published comparison. Cheap, mandatory.

**Why the GPT-2 residual init scaling.** Each block adds two residual contributions (the attention `out_proj` and the MLP `fc2`). Without rescaling, activation variance grows roughly linearly with depth, and a deep stack diverges early in training. Initializing those projections with std `0.02 / sqrt(2 × n_layer)` keeps the residual stream's variance flat through the depth.

**What changes for DiT.** Same Block structure, but:
- Tokens are flattened image patches (lab 3.1's `patchify`) instead of characters.
- No causal mask — every patch can attend to every other patch (image generation is non-autoregressive).
- LayerNorms are *modulated* by a timestep + class embedding (AdaLN-Zero) instead of being plain LN.
- The "next token" target becomes "denoise this noisy patch" — predicted via flow matching or DDPM (lab 2).

The attention kernel, the MLP shape, the residual structure — none of those change. That's the payoff of getting them right here in isolation.
