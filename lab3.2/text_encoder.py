"""Frozen CLIP text encoder for prompt embedding.

Production text-to-image models all condition on text via a frozen pretrained
text encoder. We use CLIP's text encoder (`openai/clip-vit-base-patch32`) — small
enough to run on M3, well-known to the open-source ecosystem.

The DiT consumes two things produced here:
    1. `text_tokens`: per-token CLIP outputs, shape (B, L, D_text). Used by
       cross-attention inside each DiT block — the model attends to *individual
       words* in the prompt.
    2. `text_pooled`: a single pooled vector summarizing the whole prompt,
       shape (B, D_text). Used to drive AdaLN-Zero modulation alongside the
       time embedding.

This split mirrors how SD3 / FLUX consume CLIP: token-level for cross-attention,
pooled for modulation.

Frozen means: no gradient flow into CLIP, no parameter updates. The DiT learns
to *use* CLIP's outputs; CLIP itself is treated as a fixed feature extractor.
"""

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer


class CLIPTextEncoder(nn.Module):
    """Wrapper around HuggingFace CLIPTextModel for prompt embedding."""

    def __init__(self, model_name="openai/clip-vit-base-patch32", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.max_length = max_length
        self.hidden_size = self.model.config.hidden_size  # 512 for base/32

    @torch.no_grad()
    def encode(self, prompts, device="cpu"):
        """Encode a list of prompts.

        Args:
            prompts: list of strings, length B
            device: where to move the output tensors

        Returns:
            tokens: (B, L, D_text) per-token embeddings (last hidden state)
            pooled: (B, D_text) single pooled summary per prompt
            attention_mask: (B, L) 1=real token, 0=padding
        """
        toks = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        out = self.model(**toks)
        # `last_hidden_state`: (B, L, D) — per-token contextualized embeddings
        # `pooler_output`:   (B, D)    — CLS-like summary, what CLIP's text-image
        #                                 contrastive loss uses
        return out.last_hidden_state, out.pooler_output, toks["attention_mask"]
