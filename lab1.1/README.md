# Module 1.1 — MLP (Warm-up)

> Part 1 — Building Blocks · First module of [DiT from Scratch](../README.md)

**Goal**: lock in the PyTorch training loop muscle memory.

**Why this matters for DiT**: the FFN inside every transformer block is an MLP, AdaLN modulation is an MLP from the time embedding, and the sinusoidal timestep embedding is fed through an MLP. You will write this pattern dozens of times.

**Deliverable**: `train.py` hitting 98%+ on MNIST, plus a one-paragraph "what clicked" note.

## Files

| File | What it is |
| --- | --- |
| `mlp.py` | 3-layer MLP: `Flatten → Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear` |
| `train.py` | Downloads MNIST, trains, evaluates each epoch, saves weights |
| `visualize.py` | Loads a checkpoint and plots first-layer weights reshaped to 28×28 |

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

**Dropout** randomly zeros a fraction `p` of activations during `train()` mode (and rescales the rest by `1/(1-p)` so the expected sum stays the same). At `eval()` it's a no-op. The effect: the network can't rely on any single unit, so it learns more distributed, robust features — a cheap regularizer. Re-run with `--dropout 0.4` and watch the train/test loss gap shrink (at the cost of slower training-loss descent).

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

Then give SGD its sweet spot: `--optimizer sgd --lr 0.05 --epochs 15`. It can match Adam — just needs more epochs.

### 4. Visualize first-layer weights

```bash
python visualize.py --ckpt mlp.pt --save first_layer_weights.png
```

You get an 8×8 grid of 28×28 filters. Many should look like localized pen-stroke / edge detectors; some will look noisy — both expected. Higher-dropout checkpoints tend to look cleaner.

(If you trained with non-default hidden sizes, pass them too: `--hidden 256 128`.)

### 5. Break train/eval mode

Comment out `model.train()` and `model.eval()` in `train.py` and re-run. Dropout stays active at test time → test accuracy drops and gets noisy. (BatchNorm would break the same way via running-stats vs batch-stats.)

### 6. Write your "what clicked" note

One paragraph. What surprised you — e.g. how brittle SGD is without the right `lr`, how much dropout cleans up the filters, how the five-line training loop is really the whole game.

## Flags

| Flag | Default | Notes |
| --- | --- | --- |
| `--optimizer {sgd,adam,adamw}` | `adam` | |
| `--lr` | `1e-3` | SGD wants ~`0.05` |
| `--epochs` | `10` | |
| `--batch-size` | `128` | |
| `--hidden H1 H2` | `512 256` | e.g. `--hidden 256 128` for a smaller net |
| `--dropout` | `0.2` | |
| `--save-path` | `mlp.pt` | |
| `--seed` | `0` | |
