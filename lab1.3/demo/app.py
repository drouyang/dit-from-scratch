"""Gradio webapp for the attention lab.

Three tabs:
  Reverse demo  — enter a sequence, see the model's prediction and live
                  per-head attention heatmaps. Anti-diagonal = learned the task.
  Attention 101 — interactive scaled dot-product attention with user-editable
                  query/key/value vectors, showing scores and softmax weights.
  Parity check  — runs verify.py's checks live against torch.nn.MultiheadAttention.

Setup (from `lab1.3/demo/`):
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python app.py            # opens http://127.0.0.1:7860

Expects the checkpoint in lab1.3/:
    attention.pt  (from: python train.py)
"""

import io
import math
import sys
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

LAB_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_DIR))
from attention import AttentionModel, MultiHeadAttention, causal_mask, scaled_dot_product_attention  # noqa: E402


device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")


# ---------- Load model (None if checkpoint missing) ----------
def _load_model():
    path = LAB_DIR / "attention.pt"
    if not path.exists():
        return None, None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = AttentionModel(**cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


model, cfg = _load_model()


# ---------- Helpers ----------
def _heatmap_image(matrix, title, xlabel="key pos (input)", ylabel="query pos (output)",
                   annotate=False, figsize=(3.2, 3.0)):
    """Render an (L, K) matrix as a heatmap PIL image."""
    fig, ax = plt.subplots(figsize=figsize, dpi=140)
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=max(1.0, float(matrix.max())))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.tick_params(labelsize=7)
    if annotate:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                v = matrix[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def _heads_grid(attn, title_prefix=""):
    """Render a (H, L, L) attention tensor as a row of per-head heatmaps."""
    H = attn.shape[0]
    fig, axes = plt.subplots(1, H, figsize=(2.4 * H, 2.6), dpi=140, squeeze=False)
    for h in range(H):
        ax = axes[0, h]
        im = ax.imshow(attn[h], cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"{title_prefix}head {h}", fontsize=10)
        ax.set_xlabel("key pos", fontsize=8)
        if h == 0:
            ax.set_ylabel("query pos", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def _parse_sequence(text, vocab_size, seq_len):
    """Parse whitespace/comma-separated ints. Returns (tokens list, error str)."""
    if text is None:
        return None, "Empty input."
    raw = text.replace(",", " ").split()
    if not raw:
        return None, "Empty input."
    try:
        tokens = [int(t) for t in raw]
    except ValueError:
        return None, f"Could not parse as integers: {text!r}"
    if len(tokens) != seq_len:
        return None, f"Need exactly {seq_len} tokens, got {len(tokens)}."
    for t in tokens:
        if not (0 <= t < vocab_size):
            return None, f"Token {t} out of range [0, {vocab_size})."
    return tokens, None


# ---------- Tab 1: Reverse demo ----------
def reverse_predict(text):
    if model is None:
        blank = _heatmap_image(np.zeros((1, 1)), "no checkpoint")
        return "**No checkpoint found.** Run `python train.py` first.", blank

    tokens, err = _parse_sequence(text, cfg["vocab_size"], cfg["seq_len"])
    if err:
        blank = _heatmap_image(np.zeros((1, 1)), "error")
        return f"**Error:** {err}", blank

    x = torch.tensor([tokens], device=device)
    with torch.no_grad():
        logits, attn = model(x, return_attn=True)   # attn: (1, H, L, L)
    preds = logits.argmax(dim=-1).squeeze(0).cpu().tolist()
    expected = list(reversed(tokens))
    correct = preds == expected

    mark = "✓" if correct else "✗"
    summary = (
        f"### {mark} Input → Prediction\n\n"
        f"- input:    `{tokens}`\n"
        f"- expected: `{expected}` (reversed)\n"
        f"- predicted: `{preds}`\n\n"
        f"Accuracy: **{sum(int(a == b) for a, b in zip(preds, expected))}/{len(expected)}** "
        f"positions correct."
    )

    heatmap = _heads_grid(attn.squeeze(0).cpu().numpy())
    return summary, heatmap


def random_sequence():
    if cfg is None:
        return ""
    xs = np.random.randint(0, cfg["vocab_size"], size=cfg["seq_len"])
    return " ".join(str(x) for x in xs)


# ---------- Tab 2: Attention 101 ----------
# A tiny educational calculator. User sets 3 query vectors and 3 key/value vectors
# in 2D, and we show the scaled scores, softmax weights, and output.

EDUCATION_DEFAULT = (
    "1.0 0.0\n"
    "0.0 1.0\n"
    "1.0 1.0"
)


def _parse_matrix(text, expected_rows, expected_cols, label):
    try:
        rows = [line.strip() for line in text.strip().splitlines() if line.strip()]
        if len(rows) != expected_rows:
            return None, f"{label}: need {expected_rows} rows, got {len(rows)}."
        mat = []
        for r in rows:
            nums = [float(x) for x in r.replace(",", " ").split()]
            if len(nums) != expected_cols:
                return None, f"{label}: each row needs {expected_cols} numbers."
            mat.append(nums)
        return np.array(mat, dtype=np.float32), None
    except ValueError:
        return None, f"{label}: could not parse numbers."


def attention_calc(q_text, k_text, v_text):
    q, err = _parse_matrix(q_text, 3, 2, "queries")
    if err:
        blank = _heatmap_image(np.zeros((1, 1)), "error")
        return f"**Error:** {err}", blank, blank, ""
    k, err = _parse_matrix(k_text, 3, 2, "keys")
    if err:
        blank = _heatmap_image(np.zeros((1, 1)), "error")
        return f"**Error:** {err}", blank, blank, ""
    v, err = _parse_matrix(v_text, 3, 2, "values")
    if err:
        blank = _heatmap_image(np.zeros((1, 1)), "error")
        return f"**Error:** {err}", blank, blank, ""

    qt = torch.tensor(q); kt = torch.tensor(k); vt = torch.tensor(v)
    out, attn = scaled_dot_product_attention(qt, kt, vt)
    scores = qt @ kt.T / math.sqrt(qt.shape[-1])

    scores_img = _heatmap_image(scores.numpy(), "scores = QKᵀ / √d",
                                xlabel="key j", ylabel="query i", annotate=True)
    attn_img   = _heatmap_image(attn.numpy(), "attn = softmax(scores)",
                                xlabel="key j", ylabel="query i", annotate=True)

    out_lines = "\n".join(
        f"- out[{i}] = " + " + ".join(
            f"{attn[i, j]:.2f}·v[{j}]" for j in range(3)
        ) + f" = [{out[i, 0]:.3f}, {out[i, 1]:.3f}]"
        for i in range(3)
    )
    summary = (
        "### Output\n\n"
        f"Each output row is a weighted sum of value rows, with weights from the "
        f"softmax above.\n\n{out_lines}"
    )
    return summary, scores_img, attn_img, ""


# ---------- Tab 3: Parity check ----------
def run_parity():
    torch.manual_seed(0)
    B, L, D, H = 4, 16, 64, 8

    mine = MultiHeadAttention(embed_dim=D, num_heads=H)
    ref = torch.nn.MultiheadAttention(embed_dim=D, num_heads=H, batch_first=True)
    mine.in_proj_weight.data.copy_(ref.in_proj_weight.data)
    mine.in_proj_bias.data.copy_(ref.in_proj_bias.data)
    mine.out_proj.weight.data.copy_(ref.out_proj.weight.data)
    mine.out_proj.bias.data.copy_(ref.out_proj.bias.data)

    x = torch.randn(B, L, D)

    rows = []

    def add(name, a, b, atol=1e-6):
        diff = (a - b).abs().max().item()
        ok = diff < atol
        rows.append(f"| {'✓' if ok else '✗'} | {name} | {diff:.2e} | {atol:.0e} |")
        return ok

    mine_out, mine_attn = mine(x, return_attn=True)
    ref_out, ref_attn = ref(x, x, x, need_weights=True, average_attn_weights=False)
    all_ok = True
    all_ok &= add("outputs match (unmasked)", mine_out, ref_out)
    all_ok &= add("attention weights match (unmasked)", mine_attn, ref_attn)

    mask = causal_mask(L)
    mine_out_c, mine_attn_c = mine(x, attn_mask=mask, return_attn=True)
    ref_out_c, ref_attn_c = ref(x, x, x, attn_mask=mask, need_weights=True,
                                average_attn_weights=False)
    all_ok &= add("outputs match (causal)", mine_out_c, ref_out_c)
    all_ok &= add("attention weights match (causal)", mine_attn_c, ref_attn_c)

    x1 = torch.randn(B, L, D, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    mine(x1)[0].sum().backward()
    ref(x2, x2, x2, need_weights=False)[0].sum().backward()
    all_ok &= add("input gradients match (backward)", x1.grad, x2.grad, atol=1e-5)

    header = "| | check | max \\|Δ\\| | tolerance |\n|---|---|---|---|\n"
    verdict = "### ALL CHECKS PASSED ✓" if all_ok else "### SOMETHING FAILED ✗"
    return verdict + "\n\n" + header + "\n".join(rows)


# ---------- UI ----------
with gr.Blocks(title="Attention Lab Demo") as demo:
    gr.Markdown(
        "# Module 1.3 — Attention Demo\n"
        "Three tabs: **Reverse** (trained model + attention heatmaps), "
        "**Attention 101** (tweak Q/K/V and watch softmax), "
        "**Parity** (our implementation vs. `torch.nn.MultiheadAttention`)."
    )

    with gr.Tabs():

        with gr.Tab("Reverse"):
            if model is None:
                gr.Markdown(
                    "**No checkpoint found at `lab1.3/attention.pt`.**\n"
                    "Run `python train.py` from `lab1.3/` first."
                )
            else:
                gr.Markdown(
                    f"Enter exactly **{cfg['seq_len']}** space-separated integers in "
                    f"`[0, {cfg['vocab_size']})`. The model should output the reversed "
                    "sequence, and each attention head's heatmap should look like an "
                    "anti-diagonal."
                )
                with gr.Row():
                    seq_input = gr.Textbox(
                        label=f"Input sequence ({cfg['seq_len']} tokens)",
                        value=random_sequence(),
                    )
                    rand_btn = gr.Button("Randomize", scale=0)
                predict_btn = gr.Button("Predict", variant="primary")
                rev_summary = gr.Markdown()
                rev_heatmap = gr.Image(label="Attention weights per head", type="pil")
                rand_btn.click(random_sequence, outputs=seq_input)
                predict_btn.click(reverse_predict, inputs=seq_input,
                                  outputs=[rev_summary, rev_heatmap])
                seq_input.submit(reverse_predict, inputs=seq_input,
                                 outputs=[rev_summary, rev_heatmap])

        with gr.Tab("Attention 101"):
            gr.Markdown(
                "Play with the raw scaled dot-product attention kernel. "
                "Each matrix has 3 rows (3 tokens) and 2 columns (`d_k = 2`). "
                "Watch how raising one query's alignment with a key concentrates "
                "the softmax weight on that key, and how the output is that key's "
                "value vector."
            )
            with gr.Row():
                q_box = gr.Textbox(label="Queries (3 × 2)", value=EDUCATION_DEFAULT,
                                   lines=3)
                k_box = gr.Textbox(label="Keys (3 × 2)", value=EDUCATION_DEFAULT,
                                   lines=3)
                v_box = gr.Textbox(label="Values (3 × 2)", value="1.0 0.0\n2.0 0.0\n3.0 0.0",
                                   lines=3)
            calc_btn = gr.Button("Compute", variant="primary")
            calc_summary = gr.Markdown()
            with gr.Row():
                scores_img = gr.Image(label="Scores", type="pil")
                attn_img   = gr.Image(label="Softmax weights", type="pil")
            _hidden = gr.Textbox(visible=False)
            calc_btn.click(attention_calc, inputs=[q_box, k_box, v_box],
                           outputs=[calc_summary, scores_img, attn_img, _hidden])

        with gr.Tab("Parity"):
            gr.Markdown(
                "Runs the same checks as `verify.py`: copies weights from "
                "`torch.nn.MultiheadAttention` into our implementation and "
                "compares outputs, attention weights, and gradients."
            )
            parity_btn = gr.Button("Run parity check", variant="primary")
            parity_out = gr.Markdown()
            parity_btn.click(run_parity, outputs=parity_out)


if __name__ == "__main__":
    demo.launch()
