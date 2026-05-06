"""Gradio demo for the lab 3.2 latent text-to-image DiT.

Type a prompt, get a small grid of decoded images. The four knobs match
`sample.py`'s CLI: CFG scale, Euler steps, batch size (n images per prompt),
and seed.

CLIP / SD-VAE / DiT are loaded once at startup and reused — only the
generation step runs per click.

Run:
    cd demo
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python app.py
"""

import sys
from pathlib import Path

import gradio as gr
import torch

# Import siblings from the lab3.2 directory above this demo subdir.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dit import DiT  # noqa: E402
from flow import fm_euler_sample  # noqa: E402
from text_encoder import CLIPTextEncoder  # noqa: E402
from vae import SDVae  # noqa: E402


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


CKPT_PATH = Path(__file__).resolve().parent.parent / "model.pt"

if not CKPT_PATH.exists():
    raise FileNotFoundError(
        f"checkpoint not found at {CKPT_PATH}. "
        f"Run `python train.py` from lab3.2/ first."
    )

device = get_device()
print(f"device: {device}")

# Load everything once at startup. CLIP and VAE are pretrained / no_grad;
# only the DiT consumes the saved checkpoint.
print("loading CLIP text encoder...")
text_enc = CLIPTextEncoder().to(device)

print("loading SD-VAE...")
vae = SDVae().to(device)

print("loading DiT checkpoint...")
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
cfg = ckpt["config"]
model = DiT(**cfg, p_uncond=0.1).to(device)
model.load_state_dict(ckpt["state_dict"])
model.eval()
print("ready.")


@torch.no_grad()
def generate(prompt, n_images, steps, cfg_scale, seed):
    if not prompt or not prompt.strip():
        prompt = "a pokemon"
    n = max(1, int(n_images))
    torch.manual_seed(int(seed))

    # Same prompt repeated `n` times so each row of the gallery is one
    # independent draw at the same conditioning.
    prompts = [prompt] * n
    text_tokens, text_pooled, text_mask = text_enc.encode(prompts, device)

    shape = (cfg["latent_channels"], cfg["latent_size"], cfg["latent_size"])
    z = fm_euler_sample(
        model, n, n_steps=int(steps), shape=shape,
        text_tokens=text_tokens, text_pooled=text_pooled, text_mask=text_mask,
        cfg_scale=float(cfg_scale), device=device,
    )
    images = vae.decode(z).clamp(-1, 1)
    images = (images + 1) / 2  # [-1, 1] → [0, 1]

    # Gradio Gallery wants a list of (numpy HWC, caption) tuples or PILs.
    out = []
    for i in range(n):
        arr = images[i].permute(1, 2, 0).cpu().numpy()
        out.append(arr)
    return out


with gr.Blocks(title="Lab 3.2 — Pokemon text-to-image DiT") as demo:
    gr.Markdown(
        "# Lab 3.2 — Pokemon text-to-image DiT\n"
        "Type a prompt; the trained latent DiT generates images via "
        "flow-matching Euler integration in SD-VAE's latent space, then "
        "decodes back to pixels. The model was trained on Pokemon "
        "GPT-4-captions, so all prompts get drawn in that style — "
        "feature words like *green*, *with horns*, *with wings*, *fire*, "
        "*water* nudge the silhouette and palette."
    )
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt",
                value="a cute green pokemon with red eyes and a curled tail",
                lines=3,
            )
            n_images = gr.Slider(1, 8, value=4, step=1,
                                 label="Number of images")
            steps = gr.Slider(1, 50, value=30, step=1,
                              label="Euler steps  (4–8 already plausible; 30+ for cleaner samples)")
            cfg_scale = gr.Slider(0.0, 10.0, value=4.0, step=0.5,
                                  label="CFG scale  (1=natural, 3–7=production sweet spot, 0=ignore prompt)")
            seed = gr.Number(value=0, label="Seed", precision=0)
            go = gr.Button("Generate", variant="primary")
        with gr.Column(scale=1):
            gallery = gr.Gallery(label="Generated images",
                                 columns=2, rows=2, height="auto")

    go.click(generate, [prompt, n_images, steps, cfg_scale, seed], gallery)


if __name__ == "__main__":
    demo.launch()
