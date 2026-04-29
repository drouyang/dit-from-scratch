# Module 1.5 — GPT-2 family (optional)

> Part 1 — Building Blocks · [DiT from Scratch](../README.md)

**Goal**: take the same `gpt.py` you built in lab 1.4, write a `from_pretrained` classmethod that loads OpenAI's official GPT-2 weights from HuggingFace, and compare the four GPT-2 sizes (124M / 355M / 774M / 1.5B) qualitatively and on standard benchmarks. The point isn't to train anything — it's to make concrete the claim that "GPT-1 → GPT-2 → GPT-3 is the same model bigger."

**Why this is optional**: nothing here is needed for DiT (lab 3). Skip if you're racing toward the diffusion modules. Do it if you want the scaling-law lesson made tangible on weights you can actually load and run.

## What's the same, what's different

Everything from lab 1.4's `gpt.py` carries over: `CausalSelfAttention`, `MLP`, `Block`, `GPT`, `generate`. The only changes are:

- Defaults: `block_size=1024`, `dropout=0.0` (the HF release expects these).
- `MLP.forward` uses `F.gelu(..., approximate="tanh")` to match HF GPT-2's `gelu_new` activation. The exact erf-based GELU gives slightly different numerics; the tanh approximation is what GPT-2 was trained with.
- A `GPT.from_pretrained(model_type)` classmethod. The interesting code.

The four GPT-2 sizes are the *same architecture* with three hyperparameters scaled:

| Name | Params | `n_layer` | `n_embd` | `n_head` | `head_dim` | `block_size` | fp32 size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [`gpt2`](https://huggingface.co/gpt2) | 124M | 12 | 768 | 12 | 64 | 1024 | ~500 MB |
| [`gpt2-medium`](https://huggingface.co/gpt2-medium) | 355M | 24 | 1024 | 16 | 64 | 1024 | ~1.4 GB |
| [`gpt2-large`](https://huggingface.co/gpt2-large) | 774M | 36 | 1280 | 20 | 64 | 1024 | ~3.1 GB |
| [`gpt2-xl`](https://huggingface.co/gpt2-xl) | 1.5B | 48 | 1600 | 25 | 64 | 1024 | ~6.2 GB |

**Where the params come from.**

```
total ≈ 12 · n_layer · n_embd²              (Transformer blocks)
      + (vocab_size + block_size) · n_embd  (embeddings)
```

Each block contributes about `12 · n_embd²` params: attention (`4 · n_embd²` — QKV projection is `3 · n_embd²`, output projection is `n_embd²`), plus MLP (`8 · n_embd²` — `fc1` is `4 · n_embd²`, `fc2` is another `4 · n_embd²`). Smaller bias and LayerNorm terms (about `13 · n_embd` per block) round out the rest.

Quick check for `gpt2` (124M): `12 · 12 · 768²` ≈ 85M (blocks) + `(50257 + 1024) · 768` ≈ 39M (embeddings) ≈ 124M ✓.

The embedding contribution is **fixed** across all four GPT-2 sizes (`vocab_size` and `block_size` don't change). Its share shrinks with scale — ~31% of `gpt2`, ~15% of `gpt2-medium`, ~9% of `gpt2-large`, ~6% of `gpt2-xl`. At LLaMA-70B+ scale, embeddings are rounding error and `n_layer · n_embd²` is the whole story.

`block_size` is the **context window** — the maximum number of token positions the model can attend over. At each position the model produces `Q`, `K`, `V` vectors of size `n_embd`. Each of those vectors is then split into `n_head` vectors of dim `head_dim` — one per head.

`head_dim = n_embd / n_head` is held constant at 64 across all four GPT-2 sizes — width grows by *adding more heads*, not bigger heads. Layers and width scale together. `head_dim` 64 was the early-era standard (original 2017 Transformer, GPT-2, BERT, ViT). Modern decoder-only LLMs at 7B+ scale — GPT-3, LLaMA-1/2/3, Mistral, Qwen, DeepSeek — moved to `head_dim = 128` for better tensor-core alignment and arithmetic intensity. Both 64 and 128 are widely used today; FlashAttention has fast paths for exactly these two values. The "right" `head_dim` is era- and scale-dependent, not a single number.

All four GPT-2 sizes were trained on the same WebText corpus (~40 GB), with the same BPE tokenizer (50257 tokens).

## How `from_pretrained` works

HF GPT-2's parameter layout differs from ours in two ways. The classmethod handles both.

**1. Naming.** HF uses these names:

```
transformer.wte.weight              ← token embeddings
transformer.wpe.weight              ← position embeddings
transformer.h.{i}.ln_1.{weight,bias}
transformer.h.{i}.attn.c_attn.{weight,bias}
transformer.h.{i}.attn.c_proj.{weight,bias}
transformer.h.{i}.ln_2.{weight,bias}
transformer.h.{i}.mlp.c_fc.{weight,bias}
transformer.h.{i}.mlp.c_proj.{weight,bias}
transformer.ln_f.{weight,bias}
lm_head.weight                       ← tied to wte
```

Ours uses these:

```
token_embed.weight
pos_embed.weight
blocks.{i}.ln1.{weight,bias}
blocks.{i}.attn.c_attn.{weight,bias}
blocks.{i}.attn.c_proj.{weight,bias}
blocks.{i}.ln2.{weight,bias}
blocks.{i}.mlp.fc1.{weight,bias}
blocks.{i}.mlp.fc2.{weight,bias}
ln_f.{weight,bias}
head.weight                          ← tied to token_embed
```

The remap is just six string substitutions. (See `hf_to_ours` inside `gpt.py`.)

**2. Conv1D vs Linear.** HF stores attention/MLP linears as `transformers.modeling_utils.Conv1D`, which is a linear layer with weight stored as `(in_features, out_features)` instead of PyTorch's `(out_features, in_features)`. So four weight tensors per block need a `.t()` transpose during the copy:

```
attn.c_attn.weight   (3D, D)   →   transposed to (D, 3D)
attn.c_proj.weight   (D, D)    →   transposed to (D, D)
mlp.c_fc.weight      (4D, D)   →   transposed to (D, 4D)
mlp.c_proj.weight    (D, 4D)   →   transposed to (4D, D)
```

Biases stay as-is. After remap + transpose, every tensor in our `state_dict` is filled. Skip two HF-only buffers (`.attn.bias` is a materialized causal mask we don't need since `F.scaled_dot_product_attention(is_causal=True)` handles masking; `.attn.masked_bias` is the legacy `-1e4` scalar). And skip `lm_head.weight` — it's already loaded via `wte` thanks to weight tying.

That's the entire mechanism. ~30 lines of remap + 4 transposes + 2 skips = OpenAI's GPT-2 running inside `gpt.py`.

## Setup

Python 3.9+. From `lab1.5/`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

First load of any size triggers a HuggingFace download (~500 MB for `gpt2`, ~6 GB for `gpt2-xl`) cached under `~/.cache/huggingface/`.

## Sample

Generate text from any size:

```bash
python sample.py --model gpt2          --prompt "Once upon a time"
python sample.py --model gpt2-medium   --prompt "The capital of France is" --temperature 0.6
python sample.py --model gpt2-xl       --prompt "def fibonacci(n):"        --top-k 40
```

The CLI is the same as lab 1.4's `sample.py`, with `--model` selecting the size. Tokenizer is `tiktoken.get_encoding("gpt2")` (the BPE GPT-2 was trained with).

## Side-by-side comparison

Run the same prompts through all four sizes:

```bash
python compare.py
```

This loads each model, generates completions for a fixed set of prompts, frees the model, and moves on — so you only need ~6 GB of RAM at peak (whichever is biggest), not ~11 GB for all four at once. With the default 5 prompts × 4 models × 80 tokens, expect ~2-5 minutes on M3 MPS.

Custom prompts and a subset of models:

```bash
python compare.py --models gpt2 gpt2-xl \
                  --prompts "Once upon a time" "The president of the United States" \
                  --max-tokens 100
```

The output is grouped by prompt so you can read the four completions back-to-back. Read 124M → 1.5B in order; the quality jump is most visible on completion-style prompts ("Once upon a time …") and weakest on factual ones (none of these are reliable for facts).

## Benchmarks

The right tool here is **EleutherAI's `lm-evaluation-harness`** — the standard evaluation suite for language models. It handles tokenization, batching, scoring, and outputs cleanly.

Install:

```bash
pip install lm-eval accelerate
```

`accelerate` is required for HF's `device_map` placement (used internally by `lm-eval` to load the model onto MPS / CUDA); without it you get an `ImportError` on the first run.

Run on each size:

```bash
for m in gpt2 gpt2-medium gpt2-large gpt2-xl; do
  lm_eval --model hf --model_args pretrained=$m \
          --tasks lambada_openai,wikitext,hellaswag,winogrande,arc_easy,piqa \
          --batch_size 8 \
          --output_path results/$m.json
done
```

What each task tells you (lower perplexity / higher accuracy is better):

- **`lambada_openai`** — last-word prediction over book passages. Tests *long-range* coherence; the GPT-2 paper's headline result.
  *Example:* "…my mother is going to die. She'll be gone, just like Dad. There won't be anyone left to take care of me. I'll be all ___."  → expected: **alone**. The whole paragraph is needed to predict that one word.

- **`wikitext`** — perplexity on Wikipedia. Classic LM benchmark, what the GPT-2 paper reports.
  *Example:* "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in ___, France." Score is per-token surprise across long passages — no single answer; lower perplexity = the model finds Wikipedia text less surprising.

- **`hellaswag`** — sentence-completion multiple choice. Common sense; near random at 124M, noticeable gains by 1.5B.
  *Example:* "A woman is outside with a bucket and a dog. The dog is running around trying to avoid a bath. She…"
    a) rinses the bucket off with soap and blow dries the dog's head
    b) uses a hose to keep it from getting soapy
    c) **gets the dog wet, then it runs away again** ← correct
    d) gets into a bath tub with the dog
  The wrong options are designed to *sound* fluent so a weak LM can't pick the right one by surface features alone.

- **`winogrande`** — pronoun-resolution common sense.
  *Example:* "The trophy doesn't fit into the brown suitcase because **it** is too large." Does "it" refer to the trophy or the suitcase? (**trophy**.) Same sentence with "small" instead of "large" flips the answer to **suitcase**. Pattern-matching alone fails — you need a model of what physically can/can't fit.

- **`arc_easy`** — grade-school multiple-choice science questions.
  *Example:* "What is the main cause of earthquakes?  a) rain  b) wind  c) **movement of tectonic plates**  d) clouds." Roughly 3rd–9th-grade level science.

- **`piqa`** — physical common sense.
  *Example:* "Goal: separate egg whites from yolks.  a) **Crack the egg into your hands; let the whites slip through your fingers, leaving the yolk.**  b) Crack the egg into a glass of water; the yolk will dissolve." Tests "how do everyday objects behave" knowledge.

Expect ~5-30 minutes per model on a laptop GPU for the full suite.

**Reference numbers from the GPT-2 paper** (so you know roughly what to expect):

| Model | WikiText-103 ppl | LAMBADA ppl | LAMBADA acc |
| --- | --- | --- | --- |
| 124M | 29.41 | 35.13 | 45.99% |
| 355M | 22.76 | 15.60 | 55.48% |
| 774M | 19.93 | 10.87 | 60.12% |
| 1.5B | 17.48 |  8.63 | 63.24% |

The smooth monotonic improvement across all metrics, with no architectural change, is the lesson.

**Don't bother running**: MMLU, GSM8K, HumanEval, BIG-Bench Hard. All four GPT-2 sizes will score near random / 0%. Those benchmarks are designed for ≥7B parameter models. Including them tells you "everything is broken at small scale," not a scaling story.

## What you'll observe

A handful of things you can predict from the table and then verify by reading samples:

- **Coherence over distance**: 124M loses thread within ~50 tokens; 1.5B can keep a paragraph on-topic. (LAMBADA accuracy in the table tracks this — long-range completion improves from 46% to 63%.)
- **Factual recall stays bad at every size.** Even 1.5B confidently fabricates dates, capitals, and people. None of these are reliable for facts; that's a property of base-LM-on-WebText, not a scale problem.
- **Diminishing returns are visible.** 124M → 355M is a bigger qualitative jump than 774M → 1.5B. To get the next visible jump beyond 1.5B you need to add something else (instruction tuning, much more compute, RLHF) — which is the GPT-3 / ChatGPT story.
- **Inference speed scales roughly linearly with parameters.** On M3 MPS, expect ~80 tok/s for 124M and ~10 tok/s for 1.5B at default settings.

## Discussion

**Why "the same `gpt.py`" works.** The GPT-2 paper trained four sizes on identical data, identical objective, identical tokenizer. The only thing that varies is `(n_layer, n_embd, n_head)`. So if `gpt.py` correctly implements the GPT-2 block, the same code with different hyperparameters and pretrained weights *is* GPT-2 at any of those scales. The architecture is genuinely the same — and that's the bet of the GPT line of work.

**Why `from_pretrained` matters pedagogically.** Most ML library code is mysterious because you never see what `model = AutoModel.from_pretrained("gpt2")` actually does. Implementing it once — the rename + transpose + tying — demystifies the entire `transformers` library. After this lab, "loading a pretrained checkpoint" is no longer magic; it's a string substitution and a transpose.

**What scale fixes, what scale doesn't.** Scale fixes coherence, fluency, common-sense pattern matching, in-context learning ability. Scale does not fix factuality, calibration, instruction-following, or any property the training data didn't directly demonstrate. Reading the four GPT-2 sizes side-by-side is the cheapest way to internalize that distinction.

**Connecting back to DiT.** A DiT block is structurally identical to a GPT block. The same scaling intuition applies: bigger DiTs trained on more data make better images, holding architecture fixed. DiT's *architectural* contribution is the AdaLN-Zero conditioning, not the block itself.
