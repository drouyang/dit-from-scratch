# Module 4.5 — Distribution

> Part 4 — Video DiT in Production · [DiT from Scratch](../README.md)

**Goal**: take what you trained in lab 4.3 and *publish it in three forms* each audience expects to install:

1. **LoRA on the HuggingFace Hub** — developer audience; loadable in three lines of `diffusers` code.
2. **ComfyUI custom node** — power-user / hobbyist audience; one-click install via ComfyUI Manager, becomes a graph node they can compose with everything else.
3. **Quantized weights** — small-GPU audience; 8-bit or 4-bit variants that fit on a 12 GB consumer card instead of needing 24 GB+.

**Why "distribution" not "deployment"**: distribution publishes *artifacts* so other people can run them on their own hardware. Deployment runs the model on *your* hardware behind an API. They overlap, but they're separate concerns — this lab is the first. Hosting (Modal, Replicate, fal.ai) is a real thing you might want to do; it just isn't what this lab teaches.

**Why this matters**: a trained model nobody else can run is barely shipped. The WAN ecosystem has all three of these distribution shapes live in production — Civitai and HF for LoRA / model weights, the Comfy Registry and ComfyUI-Manager for nodes, and HF-hosted quantized variants (`-nf4`, `-int8` repos) for users with smaller GPUs. After this lab, that's where your trained LoRA lives.

## What you're shipping

```
lab 4.3 produced:                        lab 4.5 turns it into:
─────────────────                        ─────────────────────────────
runs/my-style/                  ─►       1. HF model repo with model card
  ├── lora_step02000.safetensors          (loadable via pipe.load_lora_weights)
  └── adapter_config.json
                                ─►       2. ComfyUI custom node
                                          (GitHub repo, optionally listed in
                                           ComfyUI-Manager / Comfy Registry)

Wan-2.1 base (separately):       ─►      3. NF4-quantized HF variant
  ~2.6 GB bf16 transformer                (~0.7 GB; fits on 12 GB GPUs)
```

Each path serves a different audience and a different use shape. Most serious releases do all three.

## Setup

```bash
cd lab4.5/
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Path 1 needs a HuggingFace write token (`huggingface-cli login`). Path 2 has no special prereqs — you'll be writing code, not running it. Path 3 needs a CUDA GPU (`bitsandbytes` doesn't support MPS or CPU); skip it on Mac and run from a rented GPU.

## Path 1 — Publish your LoRA on HuggingFace Hub

Goal: any developer should be able to load your LoRA in three lines:

```python
from diffusers import WanPipeline
pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", ...).to("cuda")
pipe.load_lora_weights("your-username/wan-mystyle-lora")
```

For that to work, your LoRA needs to live on HuggingFace with the right files at the repo root (`adapter_config.json` + the `.safetensors`) and a model card that documents the trigger word, recommended scale, and base model.

### Steps

```bash
huggingface-cli login          # paste a write token from https://huggingface.co/settings/tokens
```

Upload from Python (`peft` already wrote both required files during training in lab 4.3):

```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("your-username/wan-mystyle-lora", repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path="../lab4.3/runs/my-style/",     # contains .safetensors + adapter_config.json
    repo_id="your-username/wan-mystyle-lora",
    repo_type="model",
)
```

### The model card

A `README.md` at the repo root *is* the model card — HF parses YAML frontmatter for discovery and renders the body. Minimal version:

```markdown
---
base_model: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
tags:
  - text-to-video
  - lora
  - wan
  - diffusers
license: apache-2.0
---

# WAN 2.1 — "mystyle" LoRA

Style LoRA fine-tuned on 50 short clips of [your domain].
**Trigger word:** `mystyle`. Recommended scale 0.8–1.2.

## Usage

\`\`\`python
import torch
from diffusers import WanPipeline

pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", torch_dtype=torch.bfloat16,
).to("cuda")
pipe.load_lora_weights("your-username/wan-mystyle-lora")

video = pipe(prompt="in mystyle, a cat on a sofa", num_frames=33).frames[0]
\`\`\`

## Training

- **Base**: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
- **Rank**: 16, **alpha**: 16
- **Steps**: 2000, **lr**: 1e-4
- **Hardware**: 1× H100 80 GB
- **Dataset**: 50 clips, 480p × 33 frames
```

The `tags` drive HF's discovery filters; `text-to-video` + `lora` + `wan` is the canonical set. The `base_model` field cross-references your LoRA against the WAN base card so the relationship shows up on both pages.

### Optional mirror: Civitai

Civitai is the de facto LoRA marketplace for the SD/FLUX/WAN community. The upload UI is web-only (no CLI), the model card format is similar but adds preview images and trigger-word fields. Mirroring isn't required, but it's where most hobbyist users discover LoRAs — worth doing if reach matters.

## Path 2 — Ship a ComfyUI custom node

Goal: a hobbyist can find your node in ComfyUI-Manager, click install, and it appears in their node graph as a single drop-in box.

### Custom node vs workflow JSON

Two things sometimes confused: a **workflow JSON** is a *graph spec* a user drops onto their ComfyUI canvas to recreate a particular node arrangement (no new code; just connections between existing nodes). A **custom node** is *code* — Python that runs inside ComfyUI and exposes new graph operations. This lab ships a custom node because:

- Your trained LoRA pairs with specific defaults (recommended CFG range, trigger words). Baking those into a node gives users a one-click experience.
- A node is the canonical packaging unit. The Comfy Registry, ComfyUI-Manager, and most installation paths are built around node packages, not workflow JSON.
- Workflow JSON depends on whatever community wrappers happen to be installed (`comfyui-wanwrapper` etc.); a custom node has explicit `requirements.txt` and runs even on a barebones ComfyUI.

### What's in `comfy_node/`

Look at the `comfy_node/` subdirectory — it's a complete, installable custom node:

```
comfy_node/
├── __init__.py           # exports NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS
├── nodes.py              # WanLoraSampler — the actual node implementation
├── pyproject.toml        # manifest for Comfy Registry (registry.comfy.org)
├── requirements.txt      # pip deps the user installs after cloning
└── README.md             # what users see on GitHub
```

The two ComfyUI-mandated exports in `__init__.py`:

```python
NODE_CLASS_MAPPINGS         = {"WanLoraSampler": WanLoraSampler}    # registers the class
NODE_DISPLAY_NAME_MAPPINGS  = {"WanLoraSampler": "WAN LoRA Sampler"} # what shows in the UI
```

A node class needs four class-level attributes that ComfyUI reads via reflection:

| Attribute | What it does |
|---|---|
| `INPUT_TYPES` (classmethod) | Returns a `{"required"/"optional": {name: type_spec}}` dict. ComfyUI generates the right widget per type — `"STRING"` becomes a textbox, `"INT"` a slider, etc. |
| `RETURN_TYPES` | Tuple of output types. We return `("IMAGE",)` — a frame batch ComfyUI knows how to wire into video savers, upscalers, etc. |
| `FUNCTION` | Name of the method ComfyUI calls per execution. Must match a method on the class. |
| `CATEGORY` | Where in the node-tree menu users find this node. We use `"WAN"`. |

Read `comfy_node/nodes.py` — at ~80 lines it's the whole node, including model caching, optional LoRA application, and the conversion from `diffusers`' PIL output to ComfyUI's `(T, H, W, 3)` float32 IMAGE tensor.

### Distribution channels

Three tiers, in increasing formality:

1. **Just push to GitHub.** Anyone with ComfyUI clones your repo into `ComfyUI/custom_nodes/`, runs `pip install -r requirements.txt`, restarts. Done. This is how `kijai/ComfyUI-WanVideoWrapper` (the most popular community WAN wrapper) is distributed.
2. **List in ComfyUI-Manager.** Submit a PR to [`ltdrdata/ComfyUI-Manager`](https://github.com/ltdrdata/ComfyUI-Manager) adding your node to `custom-node-list.json`. Now users can install it via in-app search instead of `git clone`.
3. **Publish to the Comfy Registry.** [`registry.comfy.org`](https://registry.comfy.org) is the official versioned registry. Run `comfy-cli publish` (after `pip install comfy-cli`); your `pyproject.toml`'s `[tool.comfy]` section is the registration metadata.

Most node authors do (1), then (2) once it's stable; only some bother with (3). For a lab artifact, (1) is enough.

## Path 3 — Distribute quantized weights

Goal: users with smaller GPUs (12–16 GB consumer cards instead of 24 GB+) can run your model at all.

### Why bother

Quantization compresses transformer weights from bf16 (2 bytes/param) to int8 (1 byte) or NF4 (~0.5 byte):

| Format | Size (Wan-1.3B) | Where it fits |
|---|---|---|
| bf16 (baseline) | ~2.6 GB | 24 GB+ GPU |
| int8 | ~1.3 GB | 16 GB GPU comfortably |
| NF4 (4-bit) | ~0.7 GB | 12 GB consumer GPU |

The trade-off is a small quality drop and a measurable inference *slowdown* on some GPUs (int8/NF4 matmuls aren't universally faster than bf16 — they're better on memory-bound layers, often worse on TFLOPS-bound ones). What you're buying is *fit*: 4090 / 4070 / 3060 owners can run your model at all instead of OOMing on load.

### How

`quantize.py` produces an NF4 or int8 variant of WAN's transformer using `diffusers`' `BitsAndBytesConfig` (which calls into `bitsandbytes` under the hood). Run it on a CUDA GPU:

```bash
python quantize.py --quant nf4 \
                   --output Wan2.1-T2V-1.3B-NF4 \
                   --push-to-hub your-username/wan-2.1-1.3b-nf4
```

It loads the bf16 transformer, quantizes during load, saves the quantized weights, and optionally pushes to HF as a separate model variant. Users then load:

```python
import torch
from diffusers import WanPipeline, WanTransformer3DModel, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16)
transformer = WanTransformer3DModel.from_pretrained(
    "your-username/wan-2.1-1.3b-nf4", quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
)
pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    transformer=transformer, torch_dtype=torch.bfloat16,
).to("cuda")
pipe.load_lora_weights("your-username/wan-mystyle-lora")    # your LoRA composes cleanly
```

Note that **only the transformer is quantized** in this recipe. The VAE and text encoder stay in bf16 — they're already small (~84 M and ~120 M for Wan-2.1's umT5-XXL) and matter less than the transformer for VRAM.

### What about other quantization libraries?

- **`torchao`** (PyTorch's official quant lib) — newer, native integration with `torch.compile`, but `diffusers` integration is less mature than `bitsandbytes`.
- **`optimum-quanto`** — HF's quantization wrapper; broader hardware support including Apple Silicon.
- **GGUF** — common in the LLM world (`llama.cpp`); some community tools convert diffusion checkpoints, but the WAN ecosystem doesn't standardize on it.

For *distributing on HF and pairing with diffusers + LoRA*, `bitsandbytes` via `BitsAndBytesConfig` is the path of least resistance.

## Discussion

### What you've actually shipped

After this lab and 4.2, the chain is:

```
1. trained a LoRA on a custom dataset                      (lab 4.3)
2. published it on HF with a model card                    (Path 1)
3. wrote and shipped a ComfyUI custom node that uses it    (Path 2)
4. published a quantized variant for small-GPU users       (Path 3)
```

Each path doubles the addressable audience: HF reaches developers, ComfyUI reaches hobbyists, the NF4 variant reaches the long tail of consumer-GPU users. None of the three are technically novel; the *combination* is what a small-team production launch in 2026 looks like.

### Cost

| Stage | What you paid |
|---|---|
| Lab 4.2: 2000-step LoRA train | $10–20 (rented H100, 3–4 hours) |
| Path 1: HF upload | $0 |
| Path 2: ComfyUI custom node | $0 (laptop work) |
| Path 3: quantization | a few dollars (one rented GPU run, ~10 minutes) |

End-to-end this curriculum: pretraining is $1M+ and out of reach; **LoRA + distribution is sub-$30 and absolutely reachable.** That's the whole point of Part 4.

### Where to go deeper

- [HuggingFace `peft` LoRA loading conventions](https://huggingface.co/docs/peft/conceptual_guides/lora) — multi-adapter merging, `set_adapters([scale1, scale2, ...])`, hub integration nuances.
- [ComfyUI custom-node docs](https://docs.comfy.org/custom-nodes/overview) — full `INPUT_TYPES` spec, hidden inputs (`UNIQUE_ID`, `PROMPT`), node validation, lazy evaluation.
- [`comfy-cli` and the Comfy Registry](https://github.com/Comfy-Org/comfy-cli) — official versioned-publishing path.
- [`diffusers` quantization guide](https://huggingface.co/docs/diffusers/quantization/overview) — current status of `bitsandbytes`, `torchao`, `optimum-quanto` integrations and which models support which.
- [`bitsandbytes` README](https://github.com/bitsandbytes-foundation/bitsandbytes) — NF4 vs int8 trade-offs, CUDA version requirements, multi-GPU caveats.
