# Module 1.4 — Transformer (nanoGPT)

> Part 1 — Building Blocks · [DiT from Scratch](../README.md)

**Goal**: stack the multi-head attention you wrote (and verified) in lab 1.3 with MLPs, layer norms, and residuals to get a real decoder-only transformer. Train it as a character-level language model on TinyShakespeare and generate plausible Shakespeare from a prompt.

**Why this matters for DiT**: a DiT block is structurally identical to a GPT block — multi-head attention plus an MLP, both wrapped in pre-norm + residual. The only DiT-specific addition is AdaLN-Zero conditioning, which modulates the LayerNorms from a timestep + class embedding. Get the GPT block right here and DiT becomes "this exact block, but with image patches instead of text tokens, and conditioning on the LN."

**Deliverables**:
- `gpt.py` — `Block` (LN → MHA(causal) → +res, LN → MLP → +res), `GPT` (embed → N blocks → final LN → linear head), and a `generate` method.
- `train.py` — auto-downloads TinyShakespeare, trains char-level GPT, periodically samples from the model so you can watch the text get more coherent over training.
- `sample.py` — load a checkpoint and generate text from a prompt with adjustable temperature and top-k.
- `demo/app.py` — Gradio webapp for interactive sampling.

## What the model learns

**TinyShakespeare** is ~1MB of plays from the First Folio, concatenated into one file. ~65 unique characters (uppercase + lowercase + punctuation + newline). At char-level there's no tokenizer to install — every character is a token, and the vocab is just `sorted(set(text))`.

**The task**: next-character prediction. Given a context window of `block_size` previous chars, predict the next one. Train with cross-entropy on shifted sequences:

```
input  = tokens[0..L-1]
target = tokens[1..L]
```

The causal mask ensures position `i` never sees position `i+1` during training, so each prediction is forced to depend only on the past — exactly the constraint at inference time. No train/test mismatch for autoregression.

```
 tokens (B, L)  →  [embed + pos]  →  [Block × N]  →  [LN]  →  [linear head]  →  logits (B, L, vocab)
                                       ▲
                                  causal mask
```

**What "learning" looks like over training**:

| val_loss | what the model has learned |
| --- | --- |
| ≈ 4.2 | random over 65 chars (entropy of uniform distribution) |
| ≈ 3.0 | unigram — letter and space frequencies |
| ≈ 2.0 | bigrams + spacing — words have plausible character transitions; spaces and newlines fall in plausible places |
| ≈ 1.5 | actual English words; speaker names ("ROMEO:") capitalized correctly |
| ≈ 1.3 | short phrases, archaic English ("thou", "thee", "wherefore"), turn-taking dialogue patterns |

Reaching val loss ≈ 1.3 in 5000 steps on the default config typically gives Shakespeare-flavored output: readable English, recognizable rhythm, but no semantic content (the model has no idea what it's saying).

## Files

| File | What it is |
| --- | --- |
| `gpt.py` | `MLP`, `Block`, `GPT` (with weight tying + GPT-2 init); imports `MultiHeadAttention` and `causal_mask` from `lab1.3/attention.py` — your verified kernel |
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

### 2. The architecture, by reuse

`gpt.py` deliberately reuses what you already built:

```python
sys.path.insert(0, "../lab1.3")
from attention import MultiHeadAttention, causal_mask
```

The same `MultiHeadAttention` whose forward and backward passes you proved bit-for-bit equivalent to PyTorch's built-in (lab 1.3 / `verify.py`) is now the attention sublayer of every transformer block in this lab. Lab 1.3 trained it on a toy task to see the *kernel* learn; lab 1.4 trains it inside a real architecture to see the *system* learn.

| Lab | Mask | Task | What attention learns |
| --- | --- | --- | --- |
| 1.3 | none | reverse | positional routing — anti-diagonal attention |
| 1.4 | causal | next-char | content-based routing over past tokens |

### 3. Block structure

Each `Block` is **pre-norm**: LayerNorm before each sublayer, residual added after.

```
       x  ──┬──► LN ──► MHA(causal) ──► dropout ──┐
            │                                      ▼
            └─────────────────────► + ◄────────────┘
                                    │
                             ┌──────┴──────┐
                             │             │
                             │             ▼
                             │   LN ──► MLP (Linear → GELU → Linear → dropout)
                             │             │
                             ▼             │
                             + ◄───────────┘
                             │
                             ▼  out
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
