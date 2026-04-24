"""Numerical parity check against torch.nn.MultiheadAttention.

Because MultiHeadAttention in attention.py uses the exact same parameter
layout as PyTorch's built-in, we can copy weights 1-to-1 and check that the
outputs are bit-for-bit identical (within float32 rounding).

If this passes, the scaled dot-product kernel, the Q/K/V split, the head
reshape, and the output projection are all implemented correctly.

Run: `python verify.py`
"""

import torch

from attention import MultiHeadAttention, causal_mask


def copy_weights(mine: MultiHeadAttention, ref: torch.nn.MultiheadAttention):
    """Copy weights from PyTorch's MHA into ours. The parameter names line up
    exactly — this is why we matched the layout."""
    mine.in_proj_weight.data.copy_(ref.in_proj_weight.data)
    mine.in_proj_bias.data.copy_(ref.in_proj_bias.data)
    mine.out_proj.weight.data.copy_(ref.out_proj.weight.data)
    mine.out_proj.bias.data.copy_(ref.out_proj.bias.data)


def check(name, a, b, atol=1e-6):
    diff = (a - b).abs().max().item()
    ok = diff < atol
    mark = "✓" if ok else "✗"
    print(f"  {mark} {name:32s}  max |Δ| = {diff:.2e}")
    return ok


def main():
    torch.manual_seed(0)
    B, L, D, H = 4, 16, 64, 8

    mine = MultiHeadAttention(embed_dim=D, num_heads=H)
    # batch_first=True so PyTorch takes (B, L, D) like ours does.
    ref = torch.nn.MultiheadAttention(embed_dim=D, num_heads=H, batch_first=True)
    copy_weights(mine, ref)

    x = torch.randn(B, L, D)

    print("unmasked self-attention")
    mine_out, mine_attn = mine(x, return_attn=True)
    # PyTorch's MHA takes separate Q, K, V inputs; for self-attention pass x thrice.
    # average_attn_weights=False returns per-head weights so we can compare them too.
    ref_out, ref_attn = ref(x, x, x, need_weights=True, average_attn_weights=False)
    all_ok = True
    all_ok &= check("outputs match",            mine_out, ref_out)
    all_ok &= check("attention weights match",  mine_attn, ref_attn)

    print("\ncausal self-attention")
    mask = causal_mask(L)  # (L, L) bool, True above diagonal = blocked
    mine_out_c, mine_attn_c = mine(x, attn_mask=mask, return_attn=True)
    # PyTorch's attn_mask expects the same bool convention ("True = not allowed").
    ref_out_c, ref_attn_c = ref(x, x, x, attn_mask=mask, need_weights=True,
                                average_attn_weights=False)
    all_ok &= check("outputs match (causal)",           mine_out_c, ref_out_c)
    all_ok &= check("attention weights match (causal)", mine_attn_c, ref_attn_c)

    # Additional sanity: with a causal mask, the attention matrix must be
    # lower-triangular (each row's upper-triangular entries are exactly 0).
    upper_tri_sum = mine_attn_c.masked_select(mask).abs().sum().item()
    all_ok &= check("causal attn upper-tri is zero",
                    torch.tensor(upper_tri_sum), torch.tensor(0.0), atol=1e-6)

    print("\nbackward pass parity (gradients w.r.t. input)")
    x1 = torch.randn(B, L, D, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    mine(x1)[0].sum().backward()
    ref(x2, x2, x2, need_weights=False)[0].sum().backward()
    all_ok &= check("input gradients match", x1.grad, x2.grad, atol=1e-5)

    print()
    print("ALL CHECKS PASSED ✓" if all_ok else "SOMETHING FAILED ✗")


if __name__ == "__main__":
    main()
