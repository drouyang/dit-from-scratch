"""Gradio webapp: draw a digit, get the MLP's prediction.

Setup (from `lab1.1/demo/`):
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python app.py            # opens http://127.0.0.1:7860

Expects a trained checkpoint at ../mlp.pt. Run `python train.py`
from the lab1.1 directory first if you don't have one.
"""

import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Import the MLP class from the parent lab1.1 directory.
LAB_DIR = Path(__file__).resolve().parent.parent  # points at lab1.1/
sys.path.insert(0, str(LAB_DIR))
from mlp import MLP  # noqa: E402

CKPT = LAB_DIR / "mlp.pt"
MEAN, STD = 0.1307, 0.3081  # same constants train.py normalized with


# ---------- Model (loaded once at startup) ----------
model = MLP()
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval()


# ---------- Preprocessing: canvas -> (1,1,28,28) tensor ----------
def _to_grayscale_array(canvas):
    """Gradio's Sketchpad/ImageEditor can return either a dict (composite/background/
    layers) or a raw numpy array depending on version. Normalize to a 2D uint8 array."""
    if canvas is None:
        return None
    img = canvas["composite"] if isinstance(canvas, dict) else canvas
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.ndim == 3:
        # Drop alpha if present, then ITU-R 601 luma → single channel grayscale.
        arr = arr[..., :3].mean(axis=-1)
    return arr.astype(np.uint8)


def preprocess(canvas):
    arr = _to_grayscale_array(canvas)
    if arr is None:
        return None

    # Browser canvases draw dark strokes on a light background. MNIST is the opposite
    # (bright digit on black). Invert so strokes become bright.
    arr = 255 - arr

    # Crop to the tight bounding box of the drawing, then pad with a small margin.
    # MNIST digits are centered and consistently sized — doing this massively boosts
    # accuracy vs feeding in a 28×28 resize of the full canvas.
    ys, xs = np.where(arr > 32)  # threshold to ignore antialiasing noise
    if len(xs) == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    pad = max(2, int(0.15 * max(x1 - x0, y1 - y0)))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(arr.shape[1] - 1, x1 + pad)
    y1 = min(arr.shape[0] - 1, y1 + pad)
    cropped = Image.fromarray(arr[y0:y1 + 1, x0:x1 + 1])

    # Resize to 28×28 with a high-quality filter.
    cropped = cropped.resize((28, 28), Image.LANCZOS)

    # To float tensor in [0,1], normalize with training stats, add batch + channel dims.
    # np.array(...) (instead of np.asarray) forces a writable copy, which torch.from_numpy needs.
    t = torch.from_numpy(np.array(cropped)).float() / 255.0
    t = (t - MEAN) / STD
    return t.unsqueeze(0).unsqueeze(0)  # shape (1, 1, 28, 28)


# ---------- Inference ----------
def predict(canvas):
    x = preprocess(canvas)
    if x is None:
        return {str(i): 0.0 for i in range(10)}
    with torch.no_grad():
        logits = model(x)                                # (1, 10)
        probs = F.softmax(logits, dim=1).squeeze(0).tolist()
    return {str(i): float(p) for i, p in enumerate(probs)}


# ---------- UI ----------
with gr.Blocks(title="MNIST MLP demo") as demo:
    gr.Markdown(
        "# Draw a digit (0–9)\n"
        "The drawing is inverted, cropped to its bounding box, resized to 28×28, "
        "and normalized before going to the 3-layer MLP trained in `lab1.1/`."
    )
    with gr.Row():
        with gr.Column():
            canvas = gr.Sketchpad(label="draw here", type="numpy")
            with gr.Row():
                submit = gr.Button("predict", variant="primary")
                clear = gr.Button("clear")
        with gr.Column():
            out = gr.Label(label="prediction", num_top_classes=3)

    submit.click(predict, inputs=canvas, outputs=out)
    canvas.change(predict, inputs=canvas, outputs=out)  # live updates as you draw
    clear.click(lambda: None, outputs=canvas)


if __name__ == "__main__":
    demo.launch()
