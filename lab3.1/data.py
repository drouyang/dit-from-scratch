"""MNIST loader for DiT.

Same dataset as lab 1.1 (classifier) and lab 2.1 (VAE), but normalized for
generative modeling rather than classification:

    - lab 1.1: standardize with MNIST's mean/std    (helps a discriminative MLP)
    - lab 2.1: scale to [0, 1]                       (matches BCE reconstruction)
    - lab 3.1: scale to [-1, 1]                      (matches the unit-variance
                                                      noise sampled at t=1)

The [-1, 1] choice is what Stable Diffusion / DDPM use, and it pairs cleanly
with flow matching's noise prior `x_1 ~ N(0, I)`: the data and the noise then
have the same scale, so the velocity `v = noise - x_0` has well-behaved
magnitude across the whole range of t.
"""

import torch
from torchvision import datasets, transforms

NUM_CLASSES = 10
IMAGE_SIZE = 28
IN_CHANNELS = 1


_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),                   # uint8 [0, 255] -> float [0, 1]
    transforms.Normalize((0.5,), (0.5,)),    # -> [-1, 1]
])


def get_dataset(root="./data", train=True, download=True):
    return datasets.MNIST(root, train=train, download=download, transform=_TRANSFORM)


def get_loader(batch_size, train=True, num_workers=0, pin_memory=True, root="./data"):
    ds = get_dataset(root, train=train)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=train, drop_last=train,
        num_workers=num_workers, pin_memory=pin_memory,
    )


def denormalize(x):
    """Map [-1, 1] back to [0, 1] for visualization."""
    return (x + 1.0) / 2.0
