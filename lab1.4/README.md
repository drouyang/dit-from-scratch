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

The whole lab is ~500 lines across three files. Read them top-to-bottom in this order — each one is short enough to hold in your head all at once.

### `gpt.py` (233 lines) — the model

Read the four classes in order; each builds on the previous one.

- **`CausalSelfAttention`** (`gpt.py:36`) — projects `x` to Q, K, V with one fused linear, splits into heads, calls `F.scaled_dot_product_attention(..., is_causal=True)`, and projects back. The whole multi-head attention layer is ~25 lines because the kernel is one function call. Compare against your hand-rolled version in lab 1.3 to see what the wrapper hides.
- **`MLP`** (`gpt.py:90`) — `Linear(n_embd, 4*n_embd) → GELU → Linear(4*n_embd, n_embd) → Dropout`. The 4× expansion is convention from the original Transformer paper; nothing in the math requires it, but every modern LLM keeps it.
- **`Block`** (`gpt.py:110`) — pre-norm wiring: `x = x + attn(LN(x))` then `x = x + mlp(LN(x))`. This is the *unit* the rest of deep learning is built out of. Note how the residual paths never pass through a LayerNorm — that's what makes "pre-norm" pre-norm.
- **`GPT`** (`gpt.py:142`) — embed tokens + positions, stack N blocks, final LayerNorm, linear head. Three details worth pausing on:
  - **Weight tying** at `gpt.py:170` — `self.head.weight = self.token_embed.weight`. One line, makes the input embedding and output projection share parameters.
  - **GPT-2 residual init scaling** at `gpt.py:173-179` — projections that feed into a residual stream are initialized with std `0.02 / sqrt(2 * n_layer)` to keep activation variance flat through depth.
  - **`generate`** at `gpt.py:212` — the autoregressive sampling loop. Crops context to `block_size`, applies temperature and top-k, samples one token, appends, repeats. This is the entire inference algorithm.

### `train.py` (200 lines) — the training loop

- **`get_batch`** (`train.py:74`) — picks `batch_size` random offsets into the token tensor, slices `block_size` chars at each, returns `(x, y)` where `y` is `x` shifted by one. This is the *whole* dataset pipeline for char-level — no DataLoader, no collation, just slicing.
- **`estimate_loss`** (`train.py:85`) — eval-mode forward over fixed-size train and val splits. Called every `--eval-every` steps so the printed losses are stable averages rather than single-batch noise.
- **`cosine_lr`** (`train.py:100`) — linear warmup for `--warmup` steps, then cosine decay from `max_lr` to `min_lr` over the remaining steps. Boring but standard.
- **`main`** (`train.py:110`) — the actual loop: build model → AdamW with `(0.9, 0.95)` betas and `weight_decay=0.1` → for each step, set LR by schedule, forward, backward, clip grads at 1.0, step. Every `--sample-every` steps it generates 200 chars from the model and prints them so you can watch the text get more coherent in real time.

### `sample.py` (61 lines) — load + generate

The whole file is checkpoint loading + a thin CLI around `model.generate`. Read it once to see how a trained checkpoint is reconstructed (`state_dict`, `config`, `stoi`/`itos` for the char vocab) and how the same `generate` method from `gpt.py` is reused at inference time.

### Suggested exercises

- **Print attention weights.** Modify `CausalSelfAttention.forward` to also return the attention pattern (you'll need to call the unfused version, since `F.scaled_dot_product_attention` doesn't expose them). Visualize for a generated sequence — at later layers you should see attention concentrated on semantically relevant earlier tokens.
- **Disable weight tying.** Comment out `gpt.py:170` and retrain. Compare final val loss and parameter count.
- **Swap to post-norm.** Move the LayerNorm in `Block.forward` to *after* each sublayer. Watch the loss curve — it'll be much less stable, and may need warmup tuning to converge at all.
- **Larger context.** Bump `--block-size` from 256 to 512. Memory and time per step roughly double; val loss should drop.

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
