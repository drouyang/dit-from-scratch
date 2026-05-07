# Module 4.1 — WAN 2.1 inference + code tour

**Goal**: load **WAN 2.1 T2V-1.3B**, generate a 3-second video from a text prompt, and trace every component of the running system back to a lab from Parts 1–3. By the end you can open `Wan-Video/Wan2.1`'s source and recognize patchify, AdaLN-Zero, RoPE, cross-attention, flow matching, CFG — exactly the building blocks you wrote yourself, just extended to one more dimension.

WAN 2.1 T2V-1.3B is the smallest official WAN checkpoint (~1.3B params); inference fits comfortably on a single 4090 (24 GB), with VRAM to spare.

## Run inference

### Path A: diffusers (recommended for reading)

From the repo root with the shared venv activated:

```bash
pip install -r lab4.1/requirements.txt
python lab4.1/inference_diffusers.py --prompt "a fluffy red panda eating bamboo on a tree branch"
```

`WanPipeline` for the WAN 2.1 family first run downloads the T2V-1.3B checkpoint (~5 GB total: VAE + umT5 + transformer) into the HF cache.

Output: `out.mp4` — 49 frames at 832×480, 16 fps (≈3 seconds). Generation takes ~3–5 minutes on a 4090 at default settings.

### Path B: official repo (recommended for understanding production code)

```bash
git clone https://github.com/Wan-Video/Wan2.1.git
cd Wan2.1
pip install -r requirements.txt
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B

python generate.py --task t2v-1.3B --size 832*480 \
    --ckpt_dir ./Wan2.1-T2V-1.3B \
    --prompt "a fluffy red panda eating bamboo on a tree branch"
```

T2V-1.3B doesn't need the `--offload_model` / `--t5_cpu` flags that the bigger WAN 2.2 variants require — it just fits.

## Concept-organized code tour

The lab's main deliverable. Every production component → call path → file:function in `Wan-Video/Wan2.1`, mapped to which lab introduced it.

### Entry call path

```
generate.py  (--task t2v-1.3B)
   └─► wan/text2video.py :: WanT2V.generate()
          ├─► wan/modules/t5.py :: T5EncoderModel(...)              ← text encoding
          │      (tokenizer in wan/modules/tokenizers.py)
          ├─► torch.randn(...)                                       ← initial latent z_T
          ├─► sampling loop (FlowUniPCMultistepScheduler):
          │      └─► wan/modules/model.py :: WanModel(z_t, t, c)    ← DiT forward
          │             ├─► WanAttentionBlock                         ← block (×N)
          │             │      ├─► WanSelfAttention + RoPE-3D         ← image self-attn
          │             │      └─► WanCrossAttention                  ← attends to text
          │             └─► (returns predicted velocity)
          └─► wan/modules/vae.py :: WanVAE.decode(z_0)               ← video output
```

This is structurally identical to lab 3.2's `sample.py`. Only the modalities, resolutions, and parameter counts changed.

### Key steps in `WanT2V.generate()` (`wan/text2video.py`)

The tree above is the mental model; here's the actual code with line numbers, walked top-to-bottom.

**1. Text encoding** — encodes the prompt **and** the negative prompt; both pass through the model in the sampling loop for CFG (lab 2.2).

```python
# wan/text2video.py, lines 129–137
context      = self.text_encoder([input_prompt], self.device)
context_null = self.text_encoder([n_prompt],     self.device)
```

`n_prompt` defaults to `self.sample_neg_prompt` — a fixed string the WAN team picked for video generation (artifact-suppressing words like *"blurry, low quality, distorted, ..."*).

**2. Noise initialization** — one Gaussian noise tensor with target latent shape `(C, T, H, W)`.

```python
# wan/text2video.py, lines 139–147
noise = [torch.randn(
    target_shape[0], target_shape[1],
    target_shape[2], target_shape[3],
    dtype=torch.float32, device=self.device, generator=seed_g)]
```

`seed_g` is a `torch.Generator` seeded with `args.base_seed` — what makes a run reproducible at the same seed.

**3. Scheduler construction** — two solver branches, both producing a list of `timesteps` to iterate over.

```python
# wan/text2video.py, lines 156–179
if sample_solver == 'unipc':
    sample_scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=self.num_train_timesteps,
        shift=1, use_dynamic_shifting=False)
    sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
elif sample_solver == 'dpm++':
    sample_scheduler = FlowDPMSolverMultistepScheduler(
        num_train_timesteps=self.num_train_timesteps,
        shift=1, use_dynamic_shifting=False)
    sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
    timesteps, _ = retrieve_timesteps(sample_scheduler, device=self.device, sigmas=sampling_sigmas)
```

Both replace lab 2.2's plain Euler integrator with a **higher-order multistep solver** — same flow-matching velocity field `v_θ(x_t, t)`, smarter integration of the ODE so you get the same quality at fewer steps:

- **UniPC** ([Zhao et al. 2023](https://arxiv.org/abs/2302.04867)) — predictor-corrector, up to 3rd-order. Tends to win at very low step counts (5–10). Default in WAN.
- **DPM++** ([Lu et al. 2022](https://arxiv.org/abs/2211.01095)) — 2nd-order multistep. Older, well-trodden; what most SD1.x/SDXL ComfyUI workflows ship.

The `Flow` prefix means the scheduler is adapted to flow matching's velocity parameterization (`v = noise − x_0`) rather than DDPM's noise-prediction parameterization. The integrator math is the same; only the variable being integrated differs. `shift` is the rectified-flow timestep-shifting parameter that pushes more sample budget toward the noise side of the trajectory.

**4. Sampling loop with CFG** — each step runs the DiT *twice* (once with the prompt, once with the negative prompt) and extrapolates.

```python
# wan/text2video.py, lines 175–199
for _, t in enumerate(tqdm(timesteps)):
    timestep = torch.stack([t])
    noise_pred_cond   = self.model(latent_model_input, t=timestep, **arg_c   )[0]
    noise_pred_uncond = self.model(latent_model_input, t=timestep, **arg_null)[0]
    noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
    # scheduler.step(noise_pred, t, latents) → updates latents
```

Two forward passes per step is exactly the CFG cost lab 2.2 calls out — `arg_c` carries the prompt's text embeddings, `arg_null` the negative prompt's. The CFG extrapolation is the same `v_uncond + s · (v_cond − v_uncond)` formula you wrote in lab 2.2 / lab 3.2.

`tqdm(timesteps)` is just a progress-bar wrapper — it iterates the same items in the same order and prints `100%|████| 50/50 [01:48<00:00, 2.17s/it]` to the console. No effect on the math; production inference servers (lab 4.2's SGLang-Diffusion) often strip it.

**5. Offload the DiT, free the cache** — runs *after* the sampling loop, *before* the VAE decode. This is exactly the mechanism behind the `--offload_model True` flag we discussed.

```python
# wan/text2video.py
if offload_model:
    self.model.cpu()
    torch.cuda.empty_cache()
```

Two distinct things happen:

- **`self.model.cpu()`** — `self.model` is the WAN DiT (`WanModel` from `wan/modules/model.py`, ~1.3B params, ~2.6 GB at bf16). Calling `.cpu()` on a `nn.Module` walks all its parameters, gradients, and buffers and copies them from GPU memory to CPU RAM. After this line returns, those tensors no longer occupy any GPU memory — they're just regular CPU tensors. (`.cpu()` and `.cuda()` both do this: a memcpy plus a re-binding of the parameter to the new device.)
- **`torch.cuda.empty_cache()`** — this is the part that actually frees the GPU memory back to NVIDIA's CUDA allocator. Critical detail: PyTorch maintains its own caching memory allocator on top of CUDA. When you "free" a tensor (e.g., by moving it to CPU), PyTorch keeps the underlying GPU memory in its cache so it can reuse it for the next allocation without paying a `cudaMalloc` round-trip. `nvidia-smi` will still show that memory as in-use by the process, even though no live tensor references it. `torch.cuda.empty_cache()` flushes that cache — releases the memory back to CUDA's allocator (and to `nvidia-smi`'s view of the world), so the next allocation (the VAE) has room.

Without the `empty_cache` call, the DiT's freed memory would stay reserved-but-unused in PyTorch's cache, and the VAE decode below would still OOM despite the DiT being "offloaded." Both lines together are what make the offload actually work.

**6. VAE decode** — single call after the sampling loop, on rank 0 only.

```python
# wan/text2video.py, line 208
if self.rank == 0:
    videos = self.vae.decode(x0)
```

The `if self.rank == 0` guard matters under sequence parallelism: only rank 0 holds the final assembled `x0` and writes the output file; other ranks have already done their share of the sampling work and skip the decode.

### Inside `WanModel.forward()` — the transformer

Step 4 of the sampling loop calls `self.model(latent_model_input, t=timestep, **arg_c)`. PyTorch's `nn.Module.__call__` dispatches that to **`WanModel.forward()`** (after running pre/post hooks; for inference there usually aren't any). So the body of one DiT step lives in `wan/modules/model.py`, in the `WanModel` class.

**`WanModel.forward()`** (`wan/modules/model.py`, lines 347–415) — the orchestrator for one step. Same five phases as lab 3.1's `DiT.forward()`, just with 3D extensions.

```python
# wan/modules/model.py, lines 347–415 (abridged)
def forward(self, x, t, context, seq_len, ...):
    # 1. Patchify (3D: time + space)
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]   # Conv3d
    grid_sizes = torch.stack([torch.tensor(u.shape[2:], ...) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]           # → (B, T·H·W, hidden)

    # 2. Time embedding → 6-tensor conditioning vector e0
    e  = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
    e0 = self.time_projection(e).unflatten(1, (6, self.dim))

    # 3. Text embedding (umT5 features → DiT hidden)
    context = self.text_embedding(...)

    # 4. The DiT block stack
    for block in self.blocks:
        x = block(x, e=e0, ..., context=context, ...)

    # 5. Head + unpatchify
    x = self.head(x, e)
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]
```

What's identical to lab 3.1 / 3.2: the five-phase shape, the 6 modulation tensors per block, the cross-attention to text features. What's new:

- `self.patch_embedding` is a **`Conv3d`** (lab 3.1 was `Conv2d`) with patch shape `(p_t, p_h, p_w)` — tokenizes time *and* space at once.
- `grid_sizes` carries each sample's `(T', H', W')` so unpatchify and 3D RoPE can reconstruct the spatial layout.
- `time_projection` produces `e0` of shape `(B, 6, hidden)` — the same `(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)` you wrote in lab 3.1's `adaLN_modulation`, just unflattened.

**`WanAttentionBlock.forward()`** (lines 188–276) — one block of `self.blocks`. AdaLN-Zero + self-attention + cross-attention + FFN, mirroring lab 3.2's three-sublayer block.

```python
# wan/modules/model.py, lines 210–242 (abridged)
def forward(self, x, e, ..., freqs, context, context_lens):
    e = (self.modulation + e).chunk(6, dim=1)            # 6 modulation tensors

    # self-attention sublayer (modulated + gated)
    y = self.self_attn(self.norm1(x) * (1 + e[1]) + e[0], ..., freqs)
    x = x + y * e[2]                                     # gated residual

    # cross-attention sublayer (unmodulated, like lab 3.2)
    x = x + self.cross_attn(self.norm3(x), context, context_lens)

    # FFN sublayer (modulated + gated)
    y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
    x = x + y * e[5]
    return x
```

The pattern `norm(x) * (1 + scale) + shift` is exactly lab 3.1's `modulate(LN(x), shift, scale)`. The trailing `* e[2]` / `* e[5]` is lab 3.1's gate. Cross-attention sits unmodulated between the two — same convention as lab 3.2.

**`WanSelfAttention.forward()`** (lines 68–148) — standard MHA with one twist: **RoPE-3D applied to Q and K** before the attention call.

```python
# wan/modules/model.py, lines 88–107 (abridged)
def forward(self, x, seq_lens, grid_sizes, freqs):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

    q = self.norm_q(self.q(x)).view(b, s, n, d)
    k = self.norm_k(self.k(x)).view(b, s, n, d)
    v = self.v(x).view(b, s, n, d)

    x = flash_attention(
        q=rope_apply(q, grid_sizes, freqs),              # ← RoPE-3D
        k=rope_apply(k, grid_sizes, freqs),              # ← RoPE-3D
        v=v,
        k_lens=seq_lens, window_size=self.window_size)
```

Three things worth noting:

- **`norm_q`, `norm_k`** — RMSNorm on Q and K *before* RoPE. This is the QK-norm stabilization trick that lab 4.2's SGLang-Diffusion deep dive identifies as a fusable kernel ("JIT QK-norm").
- **`rope_apply`** — the production version of lab 3.1's `apply_rope`. It splits the head dim into **three** frequency bands (one each for `t`, `h`, `w`) instead of two. The split is `c − 2·⌊c/3⌋` for `t` and `⌊c/3⌋` each for `h`, `w`, applied as a rotation per band.
- **`flash_attention`** — same `Q · Kᵀ → softmax → · V` math as lab 1.3, just dispatched to the FlashAttention kernel for tiled softmax (lab 4.2's first technique row).

**`WanT2VCrossAttention.forward()`** (lines 151–176) — structurally identical to lab 3.2's `CrossAttention`. Q from image tokens, K and V from the text `context`.

```python
# wan/modules/model.py, lines 153–176
def forward(self, x, context, context_lens):
    b, n, d = x.size(0), self.num_heads, self.head_dim

    q = self.norm_q(self.q(x)).view(b, -1, n, d)         # image queries
    k = self.norm_k(self.k(context)).view(b, -1, n, d)   # text keys
    v = self.v(context).view(b, -1, n, d)                # text values

    x = flash_attention(q, k, v, k_lens=context_lens)
    return self.o(x.flatten(2))
```

Two telling differences from self-attention: K and V come from `context` (the umT5 features), and **`rope_apply` is absent**. RoPE is for *spatial* positions; text tokens are sequential and the encoder already baked their positions into the embeddings. Same design as lab 3.2.

**`rope_params` and `rope_apply` — the 3D extension** (lines 30–65):

```python
# wan/modules/model.py, lines 31–40
def rope_params(max_seq_len, dim, theta=10000):
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs
```

`rope_params` builds a per-axis frequency table identical to lab 3.1's `rope_freqs`. The 3D extension lives in `rope_apply`, which splits the head dim into three bands (t, h, w) and rotates each by its axis position. After `Q · K`, the dot product factors cleanly into three independent cosines:

```
logit  ≈  cos(Δt·θ_t) · ⟨q_t, k_t⟩
        + cos(Δh·θ_h) · ⟨q_h, k_h⟩
        + cos(Δw·θ_w) · ⟨q_w, k_w⟩
        + (smaller sin cross-terms)
```

Same trick lab 3.1 walked through with the 4×4 worked example, just with three axes instead of two.

### Component map

| WAN component | File:function | Lab where you built it | Same as the lab | What's new |
|---|---|---|---|---|
| `WanAttentionBlock` | `wan/modules/model.py` | lab 3.1 (`DiTBlock`), lab 3.2 (added cross-attn) | LN → attn → +res structure, AdaLN-Zero modulation, gate, MLP sublayer | RoPE-3D applied inside attn |
| `WanSelfAttention` (with RoPE) | `wan/modules/model.py` | lab 3.1 (`Attention`) | Multi-head self-attention with RoPE on Q/K | RoPE has 3 axes (t, h, w) instead of 2 |
| `WanCrossAttention` | `wan/modules/model.py` | lab 3.2 (`CrossAttention`) | Image queries attend to text K/V | Larger text encoder (umT5-XXL); same op |
| `rope_params()`, `rope_apply()` | `wan/modules/model.py` | lab 3.1 (`rope_freqs`, `apply_rope`) | Same rotation-by-position trick | 3 frequency bands (t, h, w), concatenated |
| 3D patchify | `wan/modules/model.py` | lab 3.1 (Conv2d patchify) | Conv-based tokenization | `Conv3d` with patch shape `(p_t, p_h, p_w)` |
| `unpatchify()` | `wan/modules/model.py` | lab 3.1 (`unpatchify`) | Reverse of patchify | 3D rearrangement |
| `WanVAE.encode/decode` | `wan/modules/vae.py` | lab 3.2 (SD-VAE) | Latent diffusion: encode pixels → latent | 3D causal: compresses time + space; latent shape `(C, T, H, W)` |
| `T5EncoderModel` | `wan/modules/t5.py` | lab 3.2 (CLIP text encoder) | Pretrained, frozen, returns per-token features | umT5-XXL is ~100× larger; richer language understanding |
| `FlowUniPCMultistepScheduler` / `FlowDPMSolverMultistepScheduler` | imported into `text2video.py` | lab 2.2 (Euler ODE) | Flow-matching sampler | Higher-order multistep solvers (`--sample_solver unipc` is default; `dpm++` available); same `v_θ` field, fewer steps for equivalent quality |
| AdaLN-Zero modulation | `wan/modules/model.py` | lab 3.1 (`adaLN_modulation`) | `c → SiLU → Linear` decoded into shift/scale/gate per block | Identical mechanism |
| CFG | `wan/text2video.py` | lab 2.2 / 3.2 | `v_uncond + s · (v_cond − v_uncond)` | Identical formula |

## Self-check

After reading the tour:

- [ ] Open `wan/modules/model.py` and find `WanAttentionBlock`. Confirm you can map every line of its forward pass to either lab 3.1 (self-attn / AdaLN / MLP) or lab 3.2 (cross-attn).
- [ ] Open `wan/modules/vae.py` and identify the `Conv3d` layers — that's the 3D causal extension over SD-VAE's 2D `Conv2d`.
- [ ] Find `rope_apply()` and confirm three concatenated rotation bands. Compare against lab 3.1's `apply_rope()` (which has two: row + column).

If you can do these three things, you've completed the lab.
