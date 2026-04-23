"""Gradio webapp combining the CIFAR-10 classifier and autoencoder visualizer.

Two tabs:
  Classifier  — click a test image → predicted class + probability bars
  Autoencoder — click a test image → original vs reconstruction + MSE

Setup (from `lab1.2/demo/`):
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python app.py            # opens http://127.0.0.1:7860

Expects checkpoints in lab1.2/:
  cnn_classifier.pt   (from: python train_classifier.py)
  cnn_autoencoder.pt  (from: python train_autoencoder.py)
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

LAB_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_DIR))
from cnn import Autoencoder, Classifier  # noqa: E402

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
NORMALIZE = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
DISPLAY_SIZE = 96   # upscale 32×32 for readability in the gallery
LARGE_SIZE   = 192  # size for the side-by-side original/reconstruction view

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available()
          else "cpu")


# ---------- Load models (None if checkpoint is missing) ----------
def _load(cls, path, **kwargs):
    if not Path(path).exists():
        return None
    m = cls(**kwargs)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m.to(device)

classifier  = _load(Classifier,  LAB_DIR / "cnn_classifier.pt",  latent_dim=256)
autoencoder = _load(Autoencoder, LAB_DIR / "cnn_autoencoder.pt", latent_dim=256)


# ---------- Load 10 test images per class (100 total) ----------
def _load_grid():
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


def _to_pil(tensor, size):
    arr = (tensor.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr).resize((size, size), Image.NEAREST)


raw_tensors, true_labels, gallery_pils = _load_grid()


# ---------- Inference callbacks ----------
def classify(evt: gr.SelectData):
    if classifier is None:
        return "**No checkpoint found.** Run `python train_classifier.py` first.", {}
    idx = evt.index
    img_tensor  = raw_tensors[idx]
    true_label  = true_labels[idx]

    x = NORMALIZE(img_tensor).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(classifier(x), dim=1).squeeze(0).cpu().tolist()

    pred    = int(np.argmax(probs))
    correct = pred == true_label
    mark    = "✓" if correct else "✗"

    summary = (
        f"### {mark} Predicted: **{CLASSES[pred]}**\n"
        f"True label: **{CLASSES[true_label]}**\n\n"
        f"Confidence: **{probs[pred]*100:.1f}%**"
    )
    label_dict = {cls: float(p) for cls, p in zip(CLASSES, probs)}
    return summary, label_dict


def reconstruct(evt: gr.SelectData):
    if autoencoder is None:
        return None, None, "**No checkpoint found.** Run `python train_autoencoder.py` first."
    idx = evt.index
    img_tensor = raw_tensors[idx]

    x = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        recon, _ = autoencoder(x)

    mse = F.mse_loss(recon, x).item()
    original_pil = _to_pil(img_tensor,            LARGE_SIZE)
    recon_pil    = _to_pil(recon.squeeze(0).cpu(), LARGE_SIZE)
    summary      = f"Per-pixel MSE: **{mse:.5f}**  (pixel error ≈ {mse**0.5:.3f})"
    return original_pil, recon_pil, summary


# ---------- UI ----------
with gr.Blocks(title="CIFAR-10 CNN Demo") as demo:
    gr.Markdown(
        "# CIFAR-10 CNN Demo\n"
        "Grid: 10 rows × 10 columns — one row per class "
        "(airplane → automobile → bird → cat → deer → dog → frog → horse → ship → truck)."
    )

    with gr.Tabs():

        with gr.Tab("Classifier"):
            clf_gallery = gr.Gallery(
                value=gallery_pils, columns=10, rows=10,
                height=700, show_label=False, object_fit="contain", allow_preview=False,
            )
            with gr.Row():
                clf_summary = gr.Markdown("*Click an image above.*")
                clf_probs   = gr.Label(label="Class probabilities", num_top_classes=10)
            clf_gallery.select(classify, outputs=[clf_summary, clf_probs])

        with gr.Tab("Autoencoder"):
            ae_gallery = gr.Gallery(
                value=gallery_pils, columns=10, rows=10,
                height=700, show_label=False, object_fit="contain", allow_preview=False,
            )
            ae_summary = gr.Markdown("*Click an image above.*")
            with gr.Row():
                ae_original = gr.Image(label="Original")
                ae_recon    = gr.Image(label="Reconstruction")
            ae_gallery.select(reconstruct, outputs=[ae_original, ae_recon, ae_summary])


if __name__ == "__main__":
    demo.launch()
