"""Tiny COCO subset loader.

We use a small slice of MS-COCO captions (~5K image-caption pairs) at 64×64
resolution. Just enough that the DiT has real text-image pairs to learn from,
and the captions are diverse enough that the text encoder is doing real work
(unlike CIFAR-10 where you'd only have 10 distinct prompts).

Data flow:
    - HuggingFace `datasets` library streams a small subset
    - Each image is center-cropped + resized to 64×64
    - Pixels normalized to [-1, 1] to match SD-VAE's expected input range
    - Each image has 5 human captions; we pick one randomly per epoch
    - The caption is left as a raw string here — tokenization happens inside
      `CLIPTextEncoder.encode` so the text encoder owns its tokenizer
"""

import random
from io import BytesIO

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TinyCOCO(Dataset):
    """A small COCO captions subset. Lazy-streams from HuggingFace `datasets`.

    Args:
        n_samples: cap on the number of images. ~5K is a reasonable laptop scale.
        image_size: resolution to resize/crop to (default 64).
        split: "train" or "validation".
    """

    def __init__(self, n_samples=5000, image_size=64, split="train"):
        from datasets import load_dataset
        # HuggingFace mirror of MS-COCO 2017 with captions. Streaming avoids
        # downloading the full ~20GB dataset.
        ds = load_dataset("yerevann/coco-karpathy", split=split, streaming=True)
        self.samples = []
        for i, ex in enumerate(ds):
            if i >= n_samples:
                break
            self.samples.append(ex)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),                       # [0, 1]
            transforms.Normalize((0.5,) * 3, (0.5,) * 3) # [-1, 1] for SD-VAE
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ex = self.samples[idx]
        # Image: bytes or PIL depending on dataset
        img = ex["image"]
        if isinstance(img, dict) and "bytes" in img:
            img = Image.open(BytesIO(img["bytes"]))
        if img.mode != "RGB":
            img = img.convert("RGB")
        x = self.transform(img)

        # Captions: pick one of the 5 randomly
        captions = ex.get("captions") or ex.get("sentences") or [ex.get("caption", "")]
        if isinstance(captions, list) and isinstance(captions[0], dict):
            captions = [c.get("raw") or c.get("text") for c in captions]
        caption = random.choice(captions) if captions else ""

        return x, caption


def collate(batch):
    """Stack images, keep captions as a list of strings (CLIP tokenizer
    expects strings, not pre-tokenized tensors)."""
    xs = torch.stack([b[0] for b in batch])
    captions = [b[1] for b in batch]
    return xs, captions
