"""Tiny video-caption dataset for WAN LoRA fine-tuning.

Expects a directory layout:

    data/
    ├── captions.json    # [{"video": "clips/cat_01.mp4", "caption": "..."}, ...]
    └── clips/
        ├── cat_01.mp4
        ├── cat_02.mp4
        └── ...

Each video is sampled to `n_frames` frames at `fps`, center-cropped + resized
to (height, width). Pixels normalized to [-1, 1] to match Wan-VAE's expected
input range (same convention as SD-VAE).

For style/character LoRAs, 20–100 short clips of the target concept is
plenty. For broader fine-tunes, hundreds to a few thousand. Past that,
LoRA stops being the right tool — you're back in full-rank fine-tune
territory, which the README's survey section discusses.
"""

import json
from pathlib import Path

import decord
import numpy as np
import torch
from torch.utils.data import Dataset


class VideoCaptionDataset(Dataset):
    def __init__(self, root, n_frames=33, fps=16, height=480, width=832):
        self.root = Path(root)
        with (self.root / "captions.json").open() as f:
            self.entries = json.load(f)
        self.n_frames = n_frames
        self.fps = fps
        self.height = height
        self.width = width

    def __len__(self):
        return len(self.entries)

    def _load_clip(self, path):
        """(n_frames, H, W, 3) uint8 -> (3, n_frames, H, W) float32 in [-1, 1]."""
        vr = decord.VideoReader(str(path), height=self.height, width=self.width)
        # Sample n_frames evenly across the clip (or stride-1 from start if too short).
        n = len(vr)
        if n >= self.n_frames:
            idx = np.linspace(0, n - 1, self.n_frames).astype(int)
        else:
            idx = list(range(n)) + [n - 1] * (self.n_frames - n)
        frames = vr.get_batch(idx).asnumpy()                  # (T, H, W, 3) uint8
        frames = torch.from_numpy(frames).float() / 127.5 - 1 # [-1, 1]
        return frames.permute(3, 0, 1, 2).contiguous()        # (3, T, H, W)

    def __getitem__(self, idx):
        ent = self.entries[idx]
        video = self._load_clip(self.root / ent["video"])
        caption = ent["caption"]
        return video, caption


def collate(batch):
    """Stack videos along batch dim, keep captions as a list of strings."""
    videos = torch.stack([b[0] for b in batch])
    captions = [b[1] for b in batch]
    return videos, captions
