"""Tiny image-caption loader for text-to-image training.

We use `diffusers/pokemon-gpt4-captions` — 833 Pokemon images with
GPT-4-generated captions like *"a cute drawing of a green and pink pokemon
with large eyes and a curled tail"*. Tiny (~85 MB), parquet-formatted (no
script-loader issues with current `datasets`), captions are richer and
more attribute-dense than BLIP's, so cross-attention has more word-level
signal to learn from.

Why not raw MS-COCO: the canonical COCO captions datasets on HuggingFace
use *script-based* loaders that the current `datasets` library refuses
("Dataset scripts are no longer supported"). The diffusers mirror uses
native parquet, so it works without library version juggling.

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


def _extract_caption(ex):
    """Find the caption field in a sample, regardless of which name the
    dataset uses (`caption` / `text` / `prompt` / `en` are all common)."""
    for key in ("caption", "text", "prompt", "en"):
        v = ex.get(key)
        if isinstance(v, str) and v:
            return v
        if isinstance(v, list) and v:
            return v[0] if isinstance(v[0], str) else (v[0].get("raw") or v[0].get("text", ""))
    return ""


class TinyDataset(Dataset):
    """Pokemon GPT-4 captions — small image-caption set for text-to-image.

    Args:
        n_samples: cap on the number of images. Source has 833 total.
        image_size: resolution to resize/crop to (default 64).
    """

    def __init__(self, n_samples=833, image_size=64):
        from datasets import load_dataset
        ds = load_dataset("diffusers/pokemon-gpt4-captions", split="train")
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
        caption = _extract_caption(ex)
        return x, caption


def collate(batch):
    """Stack images, keep captions as a list of strings (CLIP tokenizer
    expects strings, not pre-tokenized tensors)."""
    xs = torch.stack([b[0] for b in batch])
    captions = [b[1] for b in batch]
    return xs, captions


# Backward-compat alias so train.py's `from data import TinyCOCO` still works.
TinyCOCO = TinyDataset
