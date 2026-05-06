"""Hosted inference endpoint for Wan-2.1 T2V-1.3B (+ optional LoRA from lab 4.2).

Single process serves both a Gradio UI and a FastAPI JSON endpoint, sharing
one loaded pipeline.

Run locally (rented GPU):
    python serve.py
    python serve.py --lora ../lab4.2/runs/my-style/lora_step02000.safetensors

Then visit http://<host>:7860/  for the Gradio UI, or POST to
                http://<host>:7860/api/generate  for the JSON / mp4 endpoint.

Hardware: needs a CUDA GPU with ~12 GB+ VRAM (e.g., 4090). MPS / CPU not
supported by the underlying WanPipeline — see lab 4.1's compute notes.
"""

import argparse
import tempfile

import gradio as gr
import torch
import uvicorn
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel


# ---------- pipeline (loaded once at startup) ---------------------------------

PIPE: WanPipeline | None = None
HAS_LORA: bool = False
MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def load_pipeline(lora_path: str | None) -> None:
    global PIPE, HAS_LORA
    print(f"loading {MODEL_ID}...")
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=torch.float32,
    )
    PIPE = WanPipeline.from_pretrained(
        MODEL_ID, vae=vae, torch_dtype=torch.bfloat16,
    ).to("cuda")
    if lora_path:
        print(f"loading LoRA from {lora_path}...")
        PIPE.load_lora_weights(lora_path)
        HAS_LORA = True
    print("ready.")


# ---------- shared generation function ----------------------------------------

def generate(
    prompt: str,
    num_frames: int = 49,
    steps: int = 30,
    guidance: float = 5.0,
    seed: int = 42,
    lora_scale: float = 1.0,
) -> str:
    """Generate one video and return the path to the saved mp4."""
    if HAS_LORA:
        PIPE.set_adapters("default", adapter_weights=lora_scale)

    generator = torch.Generator("cuda").manual_seed(seed)
    output = PIPE(
        prompt=prompt,
        height=480, width=832,
        num_frames=num_frames,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    ).frames[0]

    f = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    export_to_video(output, f.name, fps=16)
    return f.name


# ---------- FastAPI surface ---------------------------------------------------

app = FastAPI(title="WAN 2.1 T2V")


class GenRequest(BaseModel):
    prompt: str
    num_frames: int = 49
    steps: int = 30
    guidance: float = 5.0
    seed: int = 42
    lora_scale: float = 1.0


@app.post("/api/generate")
def generate_endpoint(req: GenRequest):
    path = generate(
        req.prompt, req.num_frames, req.steps, req.guidance, req.seed, req.lora_scale,
    )
    return FileResponse(path, media_type="video/mp4", filename="out.mp4")


# ---------- Gradio UI ---------------------------------------------------------

with gr.Blocks(title="WAN 2.1 T2V") as demo:
    gr.Markdown("# WAN 2.1 T2V — text → video")
    gr.Markdown(
        "Type a prompt; tweak the knobs; get a 3-second video. "
        f"Model: `{MODEL_ID}`."
    )
    prompt = gr.Textbox(label="Prompt", value="a cat sitting on a wooden chair, golden hour")
    with gr.Row():
        num_frames = gr.Slider(17, 81, value=49, step=4, label="Num frames (16fps)")
        steps      = gr.Slider(10, 50, value=30, step=1, label="Sampling steps")
        guidance   = gr.Slider(1.0, 10.0, value=5.0, step=0.5, label="CFG scale")
    with gr.Row():
        seed       = gr.Number(value=42, precision=0, label="Seed")
        lora_scale = gr.Slider(0.0, 1.5, value=1.0, step=0.05, label="LoRA scale (no-op if no LoRA)")
    btn = gr.Button("Generate", variant="primary")
    out = gr.Video(label="Output")
    btn.click(
        generate,
        inputs=[prompt, num_frames, steps, guidance, seed, lora_scale],
        outputs=out,
    )

# Mount Gradio at "/" so both / (UI) and /api/generate (JSON) live in one server.
app = gr.mount_gradio_app(app, demo, path="/")


# ---------- entry point -------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lora", default=None,
                   help="Path to a trained LoRA .safetensors (e.g. from lab 4.2)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()

    load_pipeline(args.lora)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
