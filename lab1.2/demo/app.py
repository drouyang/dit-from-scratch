"""Gradio webapp: click a CIFAR-10 test image, get the CNN classifier's prediction.

Setup (from `lab1.2/demo/`):
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python app.py            # opens http://127.0.0.1:7860

Expects a trained checkpoint at ../cnn_classifier.pt.
Run `python train_classifier.py` from the lab1.2 directory first.

The grid shows 100 test images — 10 per class, arranged by class row by row:
  row 0: airplane × 10
  row 1: automobile × 10
  ...
  row 9: truck × 10
"""

import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image

LAB_DIR = Path(__file__).resolve().parent.parent  # points at lab1.2/
sys.path.insert(0, str(LAB_DIR))
from cnn import Classifier  # noqa: E402

CKPT = LAB_DIR / "cnn_classifier.pt"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
NORMALIZE = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
DISPLAY_SIZE = 96  # px — upscale 32×32 so the gallery is readable


# ---------- Load model ----------
device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available()
          else "cpu")

model = Classifier(latent_dim=256)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()
model.to(device)


# ---------- Load 10 test images per class (100 total) ----------
def _load_grid():
    """Return (raw_tensors, true_labels, pil_images) for the 10×10 grid."""
    dataset = torchvision.datasets.CIFAR10(
        LAB_DIR / "data", train=False, download=True,
        transform=transforms.ToTensor(),
    )
    buckets: dict[int, list] = {i: [] for i in range(10)}
    for img, label in dataset:
        if len(buckets[label]) < 10:
            buckets[label].append((img, label))
        if all(len(v) == 10 for v in buckets.values()):
            break

    tensors, labels, pils = [], [], []
    for cls in range(10):
        for img, label in buckets[cls]:
            tensors.append(img)
            labels.append(label)
            arr = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pils.append(Image.fromarray(arr).resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.NEAREST))
    return tensors, labels, pils


raw_tensors, true_labels, gallery_pils = _load_grid()


# ---------- Inference ----------
def classify(evt: gr.SelectData):
    idx = evt.index
    img_tensor = raw_tensors[idx]
    true_label = true_labels[idx]

    x = NORMALIZE(img_tensor).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    pred = int(np.argmax(probs))
    correct = pred == true_label
    mark = "✓" if correct else "✗"

    summary = (
        f"### {mark} Predicted: **{CLASSES[pred]}**\n"
        f"True label: **{CLASSES[true_label]}**\n\n"
        f"Confidence: **{probs[pred]*100:.1f}%**"
    )
    label_dict = {cls: float(p) for cls, p in zip(CLASSES, probs)}
    return summary, label_dict


# ---------- UI ----------
with gr.Blocks(title="CIFAR-10 Classifier") as demo:
    gr.Markdown(
        "# CIFAR-10 CNN Classifier\n"
        "Click any image to classify it. "
        "Grid is 10 rows × 10 columns — one row per class "
        "(airplane → automobile → bird → cat → deer → dog → frog → horse → ship → truck)."
    )
    gallery = gr.Gallery(
        value=gallery_pils,
        columns=10,
        rows=10,
        height=700,
        show_label=False,
        object_fit="contain",
    )
    with gr.Row():
        summary_md = gr.Markdown("*Click an image above.*")
        prob_label = gr.Label(label="Class probabilities", num_top_classes=10)

    gallery.select(classify, outputs=[summary_md, prob_label])


if __name__ == "__main__":
    demo.launch()
