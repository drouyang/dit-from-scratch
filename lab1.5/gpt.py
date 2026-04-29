# Decoder-only transformer (GPT) — same architecture as lab 1.4, with a
# `from_pretrained(model_type)` classmethod that loads OpenAI's GPT-2 weights
# from HuggingFace into our class. Demonstrates that the same `gpt.py` you
# trained on TinyShakespeare runs at GPT-2 scale: only n_layer / n_embd /
# n_head and the pretrained weights change.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention via F.scaled_dot_product_attention."""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        self.resid_dropout = nn.Dropout(dropout)
        self.dropout_p = dropout

    def forward(self, x):
        B, L, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        q, k, v = self.c_attn(x).split(D, dim=-1)
        q = q.view(B, L, H, Dh).transpose(1, 2)
        k = k.view(B, L, H, Dh).transpose(1, 2)
        v = v.view(B, L, H, Dh).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.resid_dropout(self.c_proj(out))


class MLP(nn.Module):
    """Position-wise feed-forward: Linear → GELU → Linear → Dropout.

    Uses the tanh-approximation GELU to match HF GPT-2's `gelu_new` activation.
    PyTorch's default F.gelu is the exact erf-based form; for inference parity
    with the official OpenAI weights we want the approximation.
    """

    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x), approximate="tanh")))


class Block(nn.Module):
    """One transformer block: pre-norm attention + pre-norm MLP, both residual.

        x ─┬─► LN ─► MHA ─┐ ┌─► LN ─► MLP ─┐
           │              ▼ │              ▼
           └─────────────►⊕─┴─────────────►⊕─► out
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# GPT-2 family configs. All four share head_dim=64, block_size=1024, vocab=50257.
GPT2_CONFIGS = {
    "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),    # 124M
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),   # 355M
    "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280),   # 774M
    "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600),   # 1.5B
}


class GPT(nn.Module):
    """Decoder-only transformer. Predicts the next token from the past."""

    def __init__(self, vocab_size, block_size, n_layer=12, n_head=12,
                 n_embd=768, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.token_embed = nn.Embedding(vocab_size, n_embd)
        self.pos_embed = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, dropout=dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying — head.weight and token_embed.weight share parameters.
        self.head.weight = self.token_embed.weight

        self.apply(self._init_weights)
        for p_name, p in self.named_parameters():
            if p_name.endswith("c_proj.weight") or p_name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        """Total parameter count. Matches the conventional "GPT-2 124M" figure
        — `parameters()` already deduplicates tied weights (head ↔ token_embed),
        so each shared tensor is counted exactly once.
        """
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, L = idx.shape
        assert L <= self.block_size, f"sequence length {L} exceeds block_size {self.block_size}"

        pos = torch.arange(L, device=idx.device)
        x = self.token_embed(idx) + self.pos_embed(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
        )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx

    @classmethod
    def from_pretrained(cls, model_type):
        """Load OpenAI's official GPT-2 weights from HuggingFace into our class.

        HF GPT-2's parameter layout differs from ours in two ways:

        1. Naming.  HF uses `transformer.wte / wpe / h.{i}.{...} / ln_f` and
           `lm_head`; ours uses `token_embed / pos_embed / blocks.{i}.{...} /
           ln_f / head`.  Inside each block HF uses `ln_1 / ln_2`, `mlp.c_fc /
           mlp.c_proj`; ours uses `ln1 / ln2`, `mlp.fc1 / mlp.fc2`.

        2. Conv1D vs Linear.  HF stores attention/MLP linears as `Conv1D`,
           which means weight is shape `(in, out)` instead of `(out, in)`.
           So four weight tensors per block need to be transposed.

        Everything else (the math, the activation choice once we use
        approximate='tanh', the residual+pre-norm wiring) matches.
        """
        from transformers import GPT2LMHeadModel

        assert model_type in GPT2_CONFIGS, f"unknown model_type: {model_type}"
        cfg = dict(GPT2_CONFIGS[model_type])
        cfg.update(vocab_size=50257, block_size=1024, dropout=0.0)

        model = cls(**cfg)
        sd = model.state_dict()

        print(f"Downloading {model_type} from HuggingFace …")
        hf_model = GPT2LMHeadModel.from_pretrained(model_type)
        hf_sd = hf_model.state_dict()

        # Conv1D-stored weights that need a transpose.
        transposed = (
            "attn.c_attn.weight", "attn.c_proj.weight",
            "mlp.c_fc.weight",    "mlp.c_proj.weight",
        )
        # Buffers we don't need: HF stores a materialized causal mask and a
        # masked_bias scalar. We use F.sdpa with is_causal=True instead.
        skip_suffixes = (".attn.bias", ".attn.masked_bias")

        def hf_to_ours(name):
            if name.startswith("transformer."):
                name = name[len("transformer."):]
            name = name.replace("wte.", "token_embed.")
            name = name.replace("wpe.", "pos_embed.")
            name = name.replace("h.", "blocks.", 1)
            name = name.replace("ln_1.", "ln1.")
            name = name.replace("ln_2.", "ln2.")
            name = name.replace("mlp.c_fc.", "mlp.fc1.")
            name = name.replace("mlp.c_proj.", "mlp.fc2.")
            return name

        copied = 0
        for hf_name, hf_param in hf_sd.items():
            if any(hf_name.endswith(s) for s in skip_suffixes):
                continue
            # `lm_head.weight` is tied to `wte.weight` in HF; we tie head to
            # token_embed in __init__, so loading wte is enough.
            if hf_name == "lm_head.weight":
                continue

            our_name = hf_to_ours(hf_name)
            is_transposed = any(hf_name.endswith(t) for t in transposed)
            with torch.no_grad():
                target = hf_param.t() if is_transposed else hf_param
                assert sd[our_name].shape == target.shape, (
                    f"shape mismatch for {our_name}: "
                    f"ours={tuple(sd[our_name].shape)}, hf={tuple(target.shape)}"
                )
                sd[our_name].copy_(target)
            copied += 1

        print(f"  copied {copied} tensors  →  {model.num_params() / 1e6:.1f} M params")
        return model
