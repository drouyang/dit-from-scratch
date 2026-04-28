"""Gradio demo for nanoGPT.

Type a prompt and sample completions with adjustable temperature and top-k.
Useful for getting an intuition for how those knobs change the output —
low temperature is repetitive and confident, high temperature is erratic
and creative.

Run:
    cd demo
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python app.py
"""

import sys
from pathlib import Path

import gradio as gr
import torch

# Import the GPT class from the lab1.4 directory above this demo subdir.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpt import GPT  # noqa: E402


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


CKPT_PATH = Path(__file__).resolve().parent.parent / "gpt.pt"

if not CKPT_PATH.exists():
    raise FileNotFoundError(
        f"checkpoint not found at {CKPT_PATH}. "
        f"Run `python train.py` from lab1.4/ first."
    )

device = get_device()
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
cfg = ckpt["config"]
stoi = ckpt["stoi"]
itos = {i: c for c, i in stoi.items()}

model = GPT(**cfg).to(device)
model.load_state_dict(ckpt["state_dict"])
model.eval()


def encode(s):
    return [stoi.get(c, 0) for c in s]


def decode(ids):
    return "".join(itos[i] for i in ids)


def generate(prompt, max_tokens, temperature, top_k, seed):
    torch.manual_seed(int(seed))
    if not prompt:
        prompt = "\n"
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=int(max_tokens),
        temperature=float(temperature),
        top_k=int(top_k) if top_k > 0 else None,
    )
    return decode(out[0].cpu().tolist())


with gr.Blocks(title="nanoGPT — TinyShakespeare") as demo:
    gr.Markdown(
        "# nanoGPT — TinyShakespeare\n"
        "Char-level GPT trained on TinyShakespeare. "
        "Type a prompt; the model continues it character-by-character."
    )
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", value="ROMEO:", lines=4)
            max_tokens = gr.Slider(50, 1000, value=300, step=50,
                                   label="Max new tokens")
            temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.05,
                                    label="Temperature  (0=argmax, 1=raw, >1=spicier)")
            top_k = gr.Slider(0, 100, value=40, step=5,
                              label="Top-k  (0 = disabled)")
            seed = gr.Number(value=42, label="Seed", precision=0)
            go = gr.Button("Generate", variant="primary")
        with gr.Column():
            output = gr.Textbox(label="Generated text", lines=20)

    go.click(generate, [prompt, max_tokens, temperature, top_k, seed], output)


if __name__ == "__main__":
    demo.launch()
