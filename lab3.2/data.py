"""Tiny image-caption loader for text-to-image training.

We use `lambdalabs/pokemon-blip-captions` — 833 Pokemon images with
BLIP-generated captions like *"a drawing of a green pokemon with red eyes"*.
Tiny (~85 MB), parquet-formatted (fast to load on current `datasets`
versions), and the captions are diverse enough that the text encoder is
doing real work — the model has to learn how individual color, shape, and
feature words map to image regions.

Why not raw MS-COCO: the canonical COCO captions datasets on HuggingFace
use *script-based* loaders that the current `datasets` library refuses
("Dataset scripts are no longer supported"). Pokemon-blip-captions uses
native parquet, so it works without library version juggling.

Authentication note: `lambdalabs/pokemon-blip-captions` is a *gated* dataset
on HuggingFace. To download it the first time you need to:
    1. Create an HF token at https://huggingface.co/settings/tokens
    2. Visit https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions
       and accept the dataset's access terms.
    3. Authenticate locally: either run `huggingface-cli login` (interactive),
       or export `HF_TOKEN=hf_...` in your shell before running `python train.py`.

Pedagogically the same as COCO at this scale: real natural-language captions,
the model learns color/shape/feature words → image-region mappings.

Data flow:
    - HuggingFace `datasets` downloads the parquet once (cached)
    - Each image is center-cropped + resized to 64×64
    - Pixels normalized to [-1, 1] to match SD-VAE's expected input range
    - Captions are passed as raw strings — tokenization happens inside
      CLIPTextEncoder.encode (the text encoder owns its tokenizer)
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class TinyDataset(Dataset):
    """Pokemon BLIP captions — small image-caption set for text-to-image.

    Args:
        n_samples: cap on the number of images. Source has 833 total.
        image_size: resolution to resize/crop to (default 64).
    """

    def __init__(self, n_samples=833, image_size=64):
        from datasets import load_dataset
        ds = load_dataset("lambdalabs/pokemon-blip-captions", split="train")
        n = min(n_samples, len(ds))
        self.samples = [ds[i] for i in range(n)]

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
        img = ex["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        x = self.transform(img)
        caption = ex["text"]   # single BLIP-generated caption per image
        return x, caption


def collate(batch):
    """Stack images, keep captions as a list of strings (CLIP tokenizer
    expects strings, not pre-tokenized tensors)."""
    xs = torch.stack([b[0] for b in batch])
    captions = [b[1] for b in batch]
    return xs, captions


# Backward-compat alias so train.py's `from data import TinyCOCO` still works.
TinyCOCO = TinyDataset
