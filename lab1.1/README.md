# Module 1.1 — MLP (Warm-up)

> Part 1 — Building Blocks · First module of [DiT from Scratch](../README.md)

**Goal**: lock in the PyTorch training loop muscle memory.

**Why this matters for DiT**: the FFN inside every transformer block is an MLP, AdaLN modulation is an MLP from the time embedding, and the sinusoidal timestep embedding is fed through an MLP. You will write this pattern dozens of times.

**Deliverable**: `train.py` hitting 98%+ on MNIST.

## What the model learns

**MNIST** is 70,000 28×28 grayscale images of handwritten digits 0–9 (60k train / 10k test). Each image is labeled with the digit it depicts.

![MNIST digit samples — 16 examples per row, one row per digit class 0–9](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

*Sample grid from [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:MnistExamples.png) (public domain). Each row shows 16 random examples of a single digit class, illustrating the within-class handwriting variation the model has to look past.*

The task is **classification**: given the 784 pixel values, output a probability distribution over the 10 digit classes. The MLP learns a function `f: ℝ⁷⁸⁴ → ℝ¹⁰` (logits). Training minimizes **cross-entropy** between the softmax of those logits and the one-hot true label — equivalent to maximizing the log-probability the model assigns to the correct digit.

Concretely, what gets learned is a stack of matrices (`784→512`, `512→256`, `256→10`) that together transform raw pixels into features discriminative enough to separate the ten digits. 98%+ accuracy on the test set means the model generalizes that mapping to handwriting it never saw during training.

## Files

| File | What it is |
| --- | --- |
| `mlp.py` | 3-layer MLP: `Flatten → Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear` |
| `train.py` | Downloads MNIST, trains, evaluates each epoch, saves weights |
| `visualize.py` | Loads a checkpoint and plots first-layer weights reshaped to 28×28 |
| `demo/` | Gradio webapp — draw a digit in the browser and see the model's prediction |

## Instructions

### 1. Set up

Python 3.9+. From `lab1.1/`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Training auto-selects device: CUDA → Apple MPS → CPU. MNIST downloads to `./data/` on first run (~55 MB).

### 2. Train with defaults — hit 98%+

```bash
python train.py
```

Default: Adam, lr=1e-3, 10 epochs. Reaches ~98.2% test acc. Per-epoch output:

```
epoch  1 |   4.8s | train_loss 0.3412 | test_loss 0.1488 | test_acc 95.67%
...
epoch 10 |   4.6s | train_loss 0.0612 | test_loss 0.0708 | test_acc 98.24%
```

Watch `train_loss` drift below `test_loss` after epoch 5 — that's overfitting.

**Dropout** randomly zeros a fraction `p` of activations during `train()` mode (and rescales the rest by `1/(1-p)` so the expected sum stays the same). At `eval()` it's a no-op. The effect: the network can't rely on any single unit, so it learns more distributed, robust features — a cheap regularizer. Re-run with more dropout and watch the train/test loss gap shrink (at the cost of slower training-loss descent):

```bash
python train.py --dropout 0.4
```

### 3. Swap optimizers

`--lr` is the **learning rate** — the step size per gradient update (`w ← w − lr · ∇w`). The three optimizers differ in how they compute that step:

- **SGD** (Stochastic Gradient Descent): `w ← w − lr · g`. Here `g` is the raw gradient (with momentum in our setup, so it's a running average of recent gradients). Simple, but needs a large `lr` (~`0.05`) because gradients can be small.
- **Adam**: divides each parameter's step by a running estimate of its gradient magnitude. This auto-scales per-parameter → robust to gradient scale, converges fast. Sweet spot ~`1e-3`.
- **AdamW**: Adam + *decoupled* weight decay. Plain Adam couples L2 regularization into the adaptive step (which effectively cancels it); AdamW applies decay as a separate shrink-toward-zero term. Default choice for transformers today.

Run all three at the *same* `--lr` to see the raw difference:

```bash
python train.py --optimizer sgd   --lr 1e-3    # barely learns — lr too small for raw SGD
python train.py --optimizer adam  --lr 1e-3    # fast convergence
python train.py --optimizer adamw --lr 1e-3    # similar to Adam here (no weight decay tuning)
```

Then give SGD its sweet spot — it can match Adam, just needs more epochs:

```bash
python train.py --optimizer sgd --lr 0.05 --epochs 15
```

### 4. Visualize first-layer weights

The full model is three `Linear` layers with ReLU + Dropout between them:

```
[28×28 image] → Flatten(784) → Linear(784→512) → ReLU → Dropout
                             → Linear(512→256) → ReLU → Dropout
                             → Linear(256→10)  → logits
```

**Why only the first layer?** Its weight matrix has shape `(512, 784)` — each of the 512 output units has a 784-dim weight vector that lives in the *same space as the input pixels*. Reshape that vector to 28×28 and you get a picture of the pattern that unit is looking for (its dot product with the image is maximal when the image matches). Layers 2 and 3 operate on hidden activations (512-dim, 256-dim), which have no spatial meaning — you can't reshape them back to an image.

Run the visualizer:

```bash
python visualize.py --ckpt mlp.pt --save first_layer_weights.png
```

You get an 8×8 grid (64 of the 512 units) reshaped to 28×28. Bright/dark pixels in a tile = positive/negative weights at that input location. What to look for:

- **Localized blobs and oriented strokes** — units that respond to a specific pen-stroke position and angle. These are the "edge-detector"-like features deep nets are known for. The network re-uses them across all ten digit classes.
- **Noisy / salt-and-pepper tiles** — dead or underutilized units. Expected; with 512 hidden units, MNIST doesn't need all of them.
- **Higher-dropout checkpoints** tend to look cleaner and more structured: dropout forces each unit to be individually useful, which pushes weights away from the noisy regime.

If you trained with non-default hidden sizes, pass them so the checkpoint loads correctly:

```bash
python visualize.py --ckpt mlp.pt --hidden 256 128 --save first_layer_weights.png
```

### 5. Play with the model in the browser

The `demo/` directory has a small Gradio webapp that loads `mlp.pt`, lets you draw a digit on a canvas, and shows the top-3 predicted classes live as you draw. It's a quick way to see how the model responds to your own handwriting (and how different it is from the clean MNIST distribution).

Requires **Python 3.10 or newer** (Gradio 5 dropped 3.9 support). macOS's default `python3` is 3.9 — install a newer one via Homebrew if needed:

```bash
brew install python3
```

Then, from `lab1.1/demo/`:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Opens at http://127.0.0.1:7860. The preprocessing pipeline matches training: invert colors (MNIST is white-on-black), crop to the drawing's bounding box with a small margin (MNIST digits are centered and normalized in size), resize to 28×28, and normalize with the same `mean=0.1307`, `std=0.3081`. If your first attempts misclassify, try drawing in the center with moderate thickness and see how the probabilities shift.

