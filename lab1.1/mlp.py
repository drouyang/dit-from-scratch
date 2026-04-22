# Minimal 3-layer MLP for MNIST digit classification.
#
# This file defines WHAT the network computes.
# train.py defines HOW it is trained (loss, optimizer, loop).
#
# PyTorch convention for any model:
#   1. Subclass `nn.Module` — you inherit parameter tracking, .to(device),
#      .train()/.eval(), .state_dict() for saving, etc.
#   2. Register layers as attributes in __init__. Any nn.Module assigned
#      to `self.X` is auto-discovered; its parameters show up in
#      self.parameters() and will be optimized.
#   3. Implement forward(self, x). PyTorch computes the backward pass
#      automatically (autograd) as long as forward() uses differentiable ops.

import torch.nn as nn  # all layer types (Linear, ReLU, Dropout, ...) live here


class MLP(nn.Module):
    """3-layer MLP: pixels → logits over 10 digit classes.

    "3-layer" counts the Linear layers (the ones with learnable weights).
    ReLU and Dropout don't count — they're stateless transformations
    sandwiched between the Linears.
    """

    def __init__(self, in_dim=784, hidden=(512, 256), out_dim=10, dropout=0.2):
        # in_dim  = 28 * 28 = 784   (pixels in a flattened MNIST image)
        # hidden  = (h1, h2)        (widths of the two hidden layers)
        # out_dim = 10              (one logit per digit class)
        # dropout = p               (fraction of activations to zero during training)
        super().__init__()  # required — sets up the nn.Module bookkeeping
        h1, h2 = hidden

        # nn.Sequential chains modules: output of each feeds straight into
        # the next. Equivalent to writing it out in forward() step by step,
        # but more compact.
        #
        # Shape of the tensor at each stage (B = batch size, implicit & preserved):
        #
        #   input (from DataLoader): (B, 1, 28, 28)   raw MNIST, one gray channel
        #   Flatten():               (B, 784)        collapse everything after batch dim
        #   Linear(784, h1):         (B, h1)         y = x W^T + b, learnable W and b
        #   ReLU():                  (B, h1)         max(0, x) elementwise — the nonlinearity
        #   Dropout(p):              (B, h1)         zero p fraction in train(); no-op in eval()
        #   Linear(h1, h2):          (B, h2)
        #   ReLU():                  (B, h2)
        #   Dropout(p):              (B, h2)
        #   Linear(h2, 10):          (B, 10)         "logits" — raw, unnormalized class scores
        #
        # Note: NO softmax at the end. nn.CrossEntropyLoss (in train.py) expects raw
        # logits and applies log_softmax internally, which is more numerically stable
        # than doing softmax here and then log().
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, out_dim),
        )

    def forward(self, x):
        # x: (B, 1, 28, 28) batch of images.  Returns: (B, 10) logits.
        #
        # You almost never call forward() directly. In user code, do `model(x)` —
        # that routes through nn.Module.__call__, which runs pre-/post-hooks and
        # then calls forward(). Same result, correct plumbing.
        return self.net(x)
