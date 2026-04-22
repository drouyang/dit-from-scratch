import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden=(512, 256), out_dim=10, dropout=0.2):
        super().__init__()
        h1, h2 = hidden
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
        return self.net(x)
