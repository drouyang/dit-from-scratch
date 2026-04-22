# Module 1.1 — MLP (Warm-up)

**Goal**: lock in the PyTorch training loop muscle memory.

**Why this matters for DiT**: the FFN inside every transformer block is an MLP, AdaLN modulation is an MLP from the time embedding, and the sinusoidal timestep embedding is fed through an MLP. You will write this pattern dozens of times.

## Core intuitions

- All of deep learning is elaborations on `Linear → nonlinearity → Linear → ...`
- The five-line training loop: `zero_grad → forward → loss → backward → step`
- `train()` vs `eval()` mode changes Dropout/BatchNorm behavior, not gradient flow

## Exercise

- [ ] Train a 3-layer MLP on MNIST to 98%+ test accuracy
- [ ] Swap optimizers (SGD, Adam, AdamW) and observe convergence differences
- [ ] Visualize learned first-layer weights as 28×28 images — do they look like edge detectors?

## References

- PyTorch tutorials: "Learn the Basics"
- Karpathy's "A Recipe for Training Neural Networks"

---

## Files

| File | What it is |
| --- | --- |
| `mlp.py` | The model: `Flatten → Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear`. Three `Linear` layers = "3-layer MLP". |
| `train.py` | Downloads MNIST, trains, evaluates each epoch, saves weights. |
| `visualize.py` | Loads a checkpoint and plots first-layer weights reshaped to 28×28. |
| `requirements.txt` | torch, torchvision, matplotlib. |

## Setup

Python 3.9+. From `lab1.1/`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Training auto-selects device: CUDA → Apple MPS → CPU.

## Usage

### Train

```bash
# default: Adam, lr=1e-3, 10 epochs. Reaches ~98.2% test acc.
python train.py

# SGD (needs higher lr + momentum is already 0.9)
python train.py --optimizer sgd --lr 0.05 --epochs 15

# AdamW (Adam + decoupled weight decay)
python train.py --optimizer adamw --lr 1e-3
```

Flags:

| Flag | Default | Notes |
| --- | --- | --- |
| `--optimizer {sgd,adam,adamw}` | `adam` | |
| `--lr` | `1e-3` | SGD usually wants ~`0.05` |
| `--epochs` | `10` | |
| `--batch-size` | `128` | |
| `--hidden H1 H2` | `512 256` | e.g. `--hidden 256 128` for a smaller net |
| `--dropout` | `0.2` | |
| `--save-path` | `mlp.pt` | |
| `--seed` | `0` | |

MNIST downloads to `./data/` on first run (~55 MB).

Expected output each epoch:

```
epoch  1 |   4.8s | train_loss 0.3412 | test_loss 0.1488 | test_acc 95.67%
...
epoch 10 |   4.6s | train_loss 0.0612 | test_loss 0.0708 | test_acc 98.24%
```

### Visualize first-layer weights

```bash
python visualize.py --ckpt mlp.pt --save first_layer_weights.png
```

If you trained with non-default hidden sizes, pass them too: `--hidden 256 128`.

Produces an 8×8 grid of units from the first `Linear` layer, each reshaped to 28×28. Many units should look like localized pen-stroke / edge detectors; some will look noisy — both are expected.

## Suggested experiments

1. **Hit 98%+.** The default `python train.py` should get there. Notice the gap between `train_loss` and `test_loss` widening after epoch 5 — that's overfitting. Try raising `--dropout` to `0.4`.
2. **Optimizer comparison.** Run each of `sgd / adam / adamw` with the *same* `--lr 1e-3`. SGD will barely learn. Now give each its sweet spot (`sgd --lr 0.05`, `adam --lr 1e-3`, `adamw --lr 1e-3`). SGD + momentum can match Adam in accuracy but takes more epochs.
3. **Train vs eval mode.** In `train.py`, comment out `model.train()` and `model.eval()`. Dropout will now be active at test time → test accuracy drops and becomes noisy. BatchNorm would behave similarly (running stats vs batch stats).
4. **Weight visualization.** Re-run `visualize.py` on checkpoints from different optimizers / dropout levels. Higher dropout tends to produce cleaner, more structured first-layer filters.
5. **Parameter count vs accuracy.** Try `--hidden 64 32`, `--hidden 128 64`, `--hidden 512 256`, `--hidden 1024 512`. Plot params vs best test acc. You'll see diminishing returns well before 98% becomes unreachable.

## What "muscle memory" you should walk away with

```python
for epoch in range(epochs):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()       # 1. clear grads from last step
        logits = model(x)           # 2. forward
        loss = criterion(logits, y) # 3. loss
        loss.backward()             # 4. backward (fills .grad)
        optimizer.step()            # 5. update params
    evaluate(model, test_loader)    # model.eval() + torch.no_grad()
```

That's the whole game. Every subsequent lab swaps out the model, the loss, or the data — the loop stays the same.
