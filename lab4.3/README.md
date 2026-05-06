# Module 4.3 — Deployment

**Goal**: take what you've built (WAN base + your lab 4.2 LoRA) and make it usable by other people. Three production paths, three different audiences:

1. **diffusers integration on HuggingFace** — developer audience; loadable in three lines of Python.
2. **ComfyUI workflow** — power-user / hobbyist audience; visual node graph, no code.
3. **Hosted inference endpoint** (Gradio + FastAPI) — end-user audience; type a prompt in a browser, or POST to a JSON API.

A trained model nobody else can run is barely shipped. Distribution is the second half of the job, and the WAN ecosystem has all three of these deployment shapes live in production right now (Civitai for HF/LoRA, ComfyUI workflows shared on Reddit and ComfyDeploy, Replicate / Modal / Lambda Inference for hosted endpoints).

## Three deployment shapes

| | diffusers integration | ComfyUI workflow | Hosted endpoint |
|---|---|---|---|
| **Audience** | Developers, researchers | Power users, hobbyists | End users — anyone with a browser / curl |
| **Artifact** | LoRA `.safetensors` + model card on HuggingFace | Workflow JSON + reference to LoRA URL | Running server on rented GPU |
| **Caller's hardware** | They need a GPU | They need a GPU | None — your server runs the GPU |
| **Latency** | Cold-start: model download (~5GB); warm: depends on caller's GPU | Local; depends on user's GPU | Cold-start: model load (~30s); warm: ~30–60s/video |
| **Distribution cost** | Free (HF pays for storage) | Free (link to JSON + LoRA URL) | $1–3/hr per active GPU |
| **Composability** | Stack multiple LoRAs in one `pipe.load_lora_weights(...)` | Drag-and-drop nodes; visual stacking | You define the API surface |
| **Live example** | [Civitai LoRA model cards](https://civitai.com) | Reddit r/StableDiffusion, ComfyDeploy | [Replicate](https://replicate.com), [Lambda Inference](https://lambdalabs.com/inference) |

Pick based on who you want to reach:

- **Hobbyists who want to remix?** → ComfyUI workflow.
- **Devs who'll integrate it?** → diffusers / HuggingFace upload.
- **Random users with no setup?** → hosted endpoint.

In practice, a serious release does *all three*: HF for the artifact, a ComfyUI workflow for visibility, and a hosted endpoint as the demo. We'll walk through each.

## Setup

```bash
cd lab4.3/
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

You'll also need a HuggingFace account with a write token for Path 1.

## Path 1: diffusers integration (HuggingFace upload)

The goal is for any developer to be able to load your LoRA in three lines:

```python
from diffusers import WanPipeline
pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", ...).to("cuda")
pipe.load_lora_weights("your-username/wan-mystyle-lora")
```

For that to work, your LoRA needs to live on HuggingFace with the right metadata.

### Step-by-step

```bash
huggingface-cli login   # paste a write token from https://huggingface.co/settings/tokens
```

Create a new model repo on HF (web UI or `huggingface-cli repo create`). Then upload your LoRA folder:

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="../lab4.2/runs/my-style/",          # contains .safetensors + adapter_config.json
    repo_id="your-username/wan-mystyle-lora",
    repo_type="model",
)
```

`peft` writes both `lora_step02000.safetensors` *and* `adapter_config.json` during training; both are needed for `pipe.load_lora_weights(...)` to work without arguments.

### Model card

A `README.md` at the repo root *is* the model card — HF parses YAML frontmatter and renders the body. Minimal version:

```markdown
---
base_model: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
tags:
  - text-to-video
  - lora
  - wan
license: apache-2.0
---

# WAN 2.1 — "mystyle" LoRA

Style LoRA fine-tuned on 50 short clips of [your domain].
**Trigger word:** `mystyle`.

## Usage

\`\`\`python
import torch
from diffusers import WanPipeline, AutoencoderKLWan

model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights("your-username/wan-mystyle-lora")

video = pipe(prompt="in mystyle, a cat on a sofa", ...).frames[0]
\`\`\`

## Training details

- **Base**: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
- **LoRA rank**: 16
- **Steps**: 2000
- **Hardware**: 1× H100 80 GB
- **Dataset**: 50 clips, 480p × 33 frames
- **License**: same as base (apache-2.0)
```

The frontmatter tags drive HF's discovery surface — `text-to-video` + `lora` + `wan` makes your model show up in the right filter views. The `base_model` field links to the WAN base model card, which is how cards get cross-referenced.

That's all there is. Push the folder, write the card, you're done.

## Path 2: ComfyUI workflow

ComfyUI is a node-graph UI for diffusion models — the dominant tool for hobbyist video work. Sharing a "workflow" means sharing a JSON file that describes a node graph; the user drops it into their ComfyUI instance and gets a working pipeline they can tweak.

### High-level steps

1. **Install ComfyUI** locally or on your rented GPU: clone `comfyanonymous/ComfyUI`, `pip install -r requirements.txt`, run `python main.py`. The web UI is at `http://localhost:8188`.
2. **Install the WAN custom nodes.** ComfyUI's stock node set doesn't include WAN; you need either [`comfyui-wanwrapper`](https://github.com/kijai/ComfyUI-WanVideoWrapper) (most popular, supports T2V/I2V across the WAN family) or the official Wan-Video nodes if those have shipped by the time you read this. Install via ComfyUI Manager (search "WanVideo") or by cloning into `ComfyUI/custom_nodes/`.
3. **Build the graph** in the web UI:
   - `WanVideoModelLoader` → load `Wan2.1-T2V-1.3B`
   - `WanVideoLoraSelect` → point at your LoRA `.safetensors` (local file or HF URL)
   - `WanVideoTextEncode` → prompt
   - `WanVideoSampler` → connects model + text + (optional) noise → latent video
   - `WanVideoDecode` (uses 3D VAE) → final video frames
   - `VHS_VideoCombine` → save as `.mp4`
4. **Export the workflow.** Menu → Save → JSON. Ship the JSON alongside your LoRA on HF (or post to Reddit / Civitai). Other users drop the JSON onto their ComfyUI canvas (drag-and-drop) and they're up.

### Why a workflow JSON, not a Python script

ComfyUI's audience does *not* want to run `python sample.py`. They want a node graph they can drag connections in, swap LoRAs across, batch over prompts visually, plug in upscalers and frame interpolators. The JSON is a portable description of one such graph.

If your LoRA is also published on HF (Path 1), the workflow JSON can reference its URL directly — most ComfyUI WAN nodes accept either a local path or a `huggingface.co/...` URL. That's the canonical "I shipped on HF and I have a workflow" combo.

### Limitations

- Custom node names move around. `WanVideoModelLoader` may be renamed in a future release of the wrapper. When in doubt, look at the latest wrapper repo's README.
- ComfyUI workflows are tied to specific node versions. If you ship a JSON, pin the wrapper version (or attach a `requirements.txt`-equivalent).
- Spec gets messy fast — production workflows have 50+ nodes (text encoders, schedulers, upscalers, frame interpolators, savers). Start minimal.

## Path 3: Hosted inference endpoint

`serve.py` runs WAN-2.1 T2V-1.3B behind both a Gradio UI and a FastAPI JSON endpoint, sharing one loaded pipeline. Same model object; two front doors.

### Run it locally on a rented GPU

```bash
python serve.py                                                # base WAN, no LoRA
python serve.py --lora ../lab4.2/runs/my-style/lora_step02000.safetensors
```

Then:

- **Browser UI** at `http://<host>:7860/` — Gradio with prompt, frames, steps, CFG, seed, LoRA-scale knobs.
- **JSON API** at `http://<host>:7860/api/generate`:

```bash
curl -X POST http://localhost:7860/api/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "a cat dancing in a field of flowers", "num_frames": 49, "steps": 30}' \
     --output out.mp4
```

The endpoint returns the mp4 directly (`Content-Type: video/mp4`). For production you'd swap the `FileResponse` for an upload to S3 + return a signed URL — but for a single-tenant demo this is fine.

### Two-front-door design

```
                  ┌────────────────────────────────┐
                  │  uvicorn / FastAPI process     │
   browser ──────►│   ┌─── Gradio UI  ─┐           │
                  │   │                │           │
   curl   ──────► │   ├── /api/generate ┴► generate(prompt, ...) ──► mp4
                  │   │                                   │
                  │   └── shared WanPipeline (loaded once)┘
                  └────────────────────────────────┘
```

One model in VRAM, two interfaces. Loading the WAN base + LoRA is ~30 seconds and ~6GB of VRAM at bf16; you don't want to pay that twice.

`gr.mount_gradio_app(app, demo, path="/")` is the line that does the mounting. After it, the same `uvicorn` process serves both surfaces.

### Deploy to a real host

Local `python serve.py` works on a rented bare-metal GPU. For a more "click and forget" deploy, two common shapes:

- **Modal** ([`modal.com`](https://modal.com)) — serverless GPU. Wrap `serve.py`'s `generate()` in a `@modal.function(gpu="A10G")`, expose as a `@modal.web_endpoint(method="POST")`. Cold starts are ~30–60s (download + load); warm requests run normally.
- **Replicate** ([`replicate.com`](https://replicate.com)) — package `serve.py` as a `cog.yaml` predictor, push, and Replicate hosts it for you with billing-per-second. The trade-off is less control (their request format) for more turnkey.

Both let you autoscale to zero when idle, which matters for hobby projects (a 4090 idle for 24 hours costs $50; an autoscale-to-zero endpoint costs $0).

## Files

| File | What it is |
| --- | --- |
| `serve.py` | Single-process Gradio UI + FastAPI JSON endpoint. Loads `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` + optional LoRA, exposes `/` (UI) and `/api/generate` (JSON). |
| `requirements.txt` | `diffusers>=0.36`, `peft`, `gradio`, `fastapi`, `uvicorn`, `huggingface_hub`. |

## Discussion

### What you've actually shipped

After this lab and 4.2, the chain is:

```
1. trained a LoRA on a custom dataset                    (lab 4.2)
2. uploaded it to HuggingFace with a model card          (Path 1)
3. authored a ComfyUI workflow that uses it              (Path 2)
4. stood up a hosted endpoint that serves it             (Path 3)
```

That is, end-to-end, exactly what a small-team production launch looks like in May 2026. The pieces aren't novel; the *pipeline* is the deliverable.

### Cost shape, end-to-end

| Stage | What you paid |
|---|---|
| Lab 4.2: 2000-step LoRA train | $10–20 (rented H100, 3–4 hours) |
| Path 1: HF upload | $0 |
| Path 2: ComfyUI workflow | $0 (you did this on your laptop) |
| Path 3: hosted endpoint, idle | $0 with autoscale-to-zero |
| Path 3: hosted endpoint, per generation | ~$0.01–0.05 per video on Modal A10G |

The economics of this curriculum: pretraining is $1M+ and out of reach; **LoRA + deployment is sub-$50 and absolutely reachable.** That's the whole point of Part 4.

### Where to go deeper

- [HuggingFace `peft` docs](https://huggingface.co/docs/peft) — LoRA loading conventions, multi-adapter merging, hub integration.
- [ComfyUI custom node docs](https://docs.comfy.org/) — how to write your own nodes if your LoRA needs special preprocessing.
- [Modal docs](https://modal.com/docs) and [Replicate's `cog` docs](https://github.com/replicate/cog) — turnkey GPU hosting.
- [`diffusers` Wan pipeline docs](https://huggingface.co/docs/diffusers/api/pipelines/wan) — current API for WAN 2.1 in stable releases, and the WAN 2.2 extensions on `main`.
