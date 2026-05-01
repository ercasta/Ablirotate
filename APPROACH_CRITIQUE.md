# Critique of the Ablirotate Approach

This document is a technical review of the three design notes in this repository:

- `APPROACH.md` — gradient-free post-training pruning + defragmentation.
- `INTERLEAVED_TRAINING.md` — cyclic reorder/freeze/grow training loop.
- `AUTONOMOUS_AGENT.md` — unattended sandbox loop with periodic micro-training.

The brief in `critique.md` asks for (a) reinforcement of strong ideas, (b)
identification of weak points, and (c) specific answers to three questions:

1. How does neuron swapping and compaction in few layers actually help
   computation?
2. What exactly is swapped? Does it require altering the LLM's structure or
   is it pure reshuffling?
3. What harness is needed to control the "temperature" that activates /
   deactivates rarely used neuron groups, track activations, and produce a
   *dynamic-size* model with layers that can live on disk and be hot-loaded?
   Can the technique sit on top of existing frameworks (PyTorch, agentic
   harnesses) or does it require a redesign?

Throughout, I anchor the analysis in two concrete architectures already
referenced by the project: **Qwen2.5-Coder-32B** (dense SwiGLU + GQA) and
**Gemma-3-27B** (Mixture-of-Experts). Numbers come from the published model
configs.

---

## 1. Executive summary

The approach has a clear, defensible nucleus and several weak spots that
deserve attention before scale-up.

**What is genuinely strong**
- The choice of *structured* (neuron-level) sparsity over unstructured
  weight-level sparsity is the right call for commodity hardware and is
  argued correctly in `APPROACH.md §3`.
- Defragmentation as the bridge between *logical* sparsity (mask) and
  *physical* speed-up (smaller GEMM) is the most important and correct
  insight in the repo.
- Coordinated permutation across `gate_proj`, `up_proj`, and `down_proj`
  for SwiGLU is a non-trivial requirement and is handled correctly in the
  code (numerically equivalent up to truncation).
- The four-way classification (`common / keep_specific / drop_specific /
  neutral`) is a more principled reformulation of "abliteration" that
  generalises beyond the single-direction refusal-vector trick.

**What is shaky**
- "Activation rate" is a binary, threshold-based signal that throws away
  magnitude and sign — much weaker than Wanda's `|w|·‖x‖` or a small Taylor
  proxy. The repo presents this as a feature (gradient-free) but the cost
  is rarely quantified.
- "Compacting in few layers" (the user's phrasing) is misleading: the
  number of layers does not change. Only the *width* of each layer's MLP
  intermediate dimension and (potentially) attention head count change.
  Compacting *across* layers would break the residual stream and is not
  what the code does.
- The "Cold mode" thermal-reactivation story (negative bias suppresses a
  neuron, sampling temperature can re-enable it) does not work as
  described: sampling temperature scales the output logits, not internal
  SwiGLU gates. A real temperature-conditional activation requires a
  separate mechanism (see §4.2).
- Numerical-equivalence claims silently break under quantisation (the
  permutation reorders per-channel scales, which is *not* a no-op).
- The autonomous-loop assumes a frozen `universal_boundary.pt` from the
  reference model can guide an agent whose neurons are diverging from the
  reference. After a few cycles the boundary is stale and the agent's
  neuron i and the reference's neuron i refer to different features.
- Several of the harder pieces of `INTERLEAVED_TRAINING.md` (mid-run
  optimizer-state surgery, layer growth, hot-swap inference) are
  underspecified and will require more than a handful of utility functions.

The remainder of this document expands each point.

---

## 2. The target architectures (numbers used below)

To avoid hand-waving, the rest of the document refers to these concrete
shapes.

**Qwen2.5-Coder-32B** (dense, SwiGLU + GQA):

| Field | Value |
|---|---|
| `num_hidden_layers` | 64 |
| `hidden_size` (d_model) | 5 120 |
| `intermediate_size` (d_ffn) | 27 648 |
| `num_attention_heads` (Q) | 40 |
| `num_key_value_heads` (KV) | 8 |
| `head_dim` | 128 |
| MLP | SwiGLU: `down_proj(SiLU(gate_proj(x)) * up_proj(x))` |
| Total params (approx.) | 32.5 B |

Per-layer MLP cost per token:

```
gate_proj : 5 120 × 27 648  =  141.6 M MACs
up_proj   : 5 120 × 27 648  =  141.6 M MACs
down_proj : 27 648 × 5 120  =  141.6 M MACs
                              ───────────
                              424.7 M MACs / layer
```

× 64 layers ≈ **27.2 G MACs / token** for MLP alone.

Per-layer attention cost per token (excluding KV-cache reads):

```
q_proj : 5 120 × 5 120     = 26.2 M
k_proj : 5 120 × 1 024     =  5.2 M     (GQA: 8 × 128)
v_proj : 5 120 × 1 024     =  5.2 M
o_proj : 5 120 × 5 120     = 26.2 M
                            ──────
                            62.8 M MACs / layer
```

× 64 layers ≈ **4.0 G MACs / token** for attention projections (the dot
products and softmax are extra and depend on context length).

So MLP dominates by ~7×, which is why MLP-side compaction is where most of
the speed-up has to come from. This is consistent with what the code
optimises.

**Gemma-3-27B** is structured very differently (MoE with sparse
routing). Numbers and consequences for that model are discussed in §6.

---

## 3. Question 1 — how does compaction actually help computation?

### 3.1 The argument is correct, but only for the part it controls

Modern GPUs and tensor cores execute dense GEMM at full throughput
regardless of the values stored in the matrix. A weight tensor that is 50 %
zero is no faster than a fully dense one unless either

- the hardware has dedicated structured-sparse kernels (Ampere+ 2:4
  sparse Tensor Cores, ~1.6× speed-up at best), or
- the zeros are removed *physically*, shrinking the matrix.

`APPROACH.md` correctly chooses the second route and applies it at
neuron granularity. With `keep_fraction = 0.80` on Qwen Coder 32B, the
intermediate dimension drops from **27 648 → 22 118**. Per-layer MLP MACs
drop from 424.7 M → 339.7 M, a **20 % reduction**.

### 3.2 But "20 % MLP reduction" ≠ "20 % wall-clock speed-up"

The doc's `§4.5` reports "roughly 20 % reduction in MLP FLOPs with minimal
impact on code-generation quality". That is the *MLP-only* number. End-to-end
inference cost is dominated by MLP only at moderate context length.
Approximate breakdown for Qwen Coder 32B at 4 K context, batch 1, decode
phase:

| Component | Share of compute | Touched by Ablirotate? |
|---|---:|---|
| MLP projections | ~65 % | Yes (intermediate dim shrink) |
| Attention QKVO projections | ~10 % | Possible (head pruning) but not in the SwiGLU defragmenter today |
| Attention dot product + softmax + KV cache I/O | ~20 % | No |
| LM head, RMSNorm, rotary, residual adds | ~5 % | No |

A 20 % reduction in the MLP slice (65 %) yields ≈ **13 % wall-clock
speed-up** for compute-bound regimes, less for memory-bound ones. That
is still a real win, but the README/APPROACH sections currently invite
the reader to interpret it as ~20 % overall, which it isn't.

For batch-1 autoregressive decoding the regime is **memory-bandwidth
bound**: the limiting factor is reading parameters from HBM. The relevant
saving is then the *parameter count* reduction. With Qwen Coder 32B the
MLP intermediate dim contributes:

```
3 × hidden_size × intermediate_size × num_layers × dtype_bytes
= 3 × 5 120 × 27 648 × 64 × 2
≈ 54.4 GB out of ~65 GB total in bf16
```

A 20 % truncation removes ≈ 10.9 GB of bandwidth-bound weight reads per
forward step, roughly a **~17 % bandwidth reduction**. So bandwidth-bound
batch-1 decode does see a wall-clock benefit close to the MLP-FLOP figure.
Compute-bound prefill (long prompt) sees less.

It would strengthen `APPROACH.md` to (a) separate the prefill vs. decode
case and (b) state a wall-clock figure rather than a FLOP figure when
making throughput claims.

### 3.3 "Compacting in few layers" — clarifying the user's question

The user asks how compacting "in few layers" helps. The accurate answer is
that **Ablirotate does not compact across layers**. It compacts *inside*
each of the 64 transformer blocks independently: each block's MLP
intermediate dim shrinks, each block's attention head count may shrink,
but the residual stream's `hidden_size = 5 120` and the layer count
`num_hidden_layers = 64` are invariant.

Compacting across layers (e.g. moving "useful" neurons from layer 17 into
layer 12 and dropping layer 17) is **not possible** under this scheme
because:

- Each transformer block's `down_proj` writes back into a residual stream
  whose i-th coordinate has a fixed semantic meaning across all layers
  (any permutation must be applied identically to every layer's `down_proj`
  output, every layer's `q/k/v/o_proj` input, every LayerNorm parameter,
  every rotary frequency, the embedding and the LM head — at which point
  you have a global rotation, à la SliceGPT, not a local permutation).
- Removing a whole layer is a different operation (depth pruning, e.g.
  ShortGPT, LayerSkip) and is *not* what defragmentation does.

The doc would benefit from making this explicit, because the user's question
is a natural reading of "compacting" and indicates the boundary is not as
clear as it should be.

### 3.4 Where the speed-up is over-claimed

A few specific items to tighten:

- `§4.5` states "20 % reduction in MLP FLOPs". This is true per layer for
  the MLP; it is not the whole-model speed-up.
- The expected **dead fraction** ("10–25 % of neurons … rate < 5 %") is
  cited from "Llama-2-70B, Mistral-7B" but no measurement on Qwen Coder
  32B is provided. SwiGLU dead-fraction depends heavily on the calibration
  distribution — code-only calibration will look denser than mixed.
- For attention, head pruning has a comparable arithmetic effect but is
  *not yet implemented* in the Qwen-specific defragmenter (`Limitations §3`),
  so the 20 % MLP figure is the only contribution today.

---

## 4. Question 2 — what is actually swapped?

### 4.1 The two distinct operations

There are **two operations**, and they have very different compatibility
implications. The repo sometimes blurs them.

**Operation A — pure permutation (function-preserving)**

Inside one MLP block, a permutation P over the intermediate axis is applied
simultaneously to:

- rows of `gate_proj`        (shape `[d_ffn, d_model]`)
- rows of `up_proj`          (shape `[d_ffn, d_model]`)
- columns of `down_proj`     (shape `[d_model, d_ffn]`)

The forward computation is

```
y = down_proj · (SiLU(gate_proj · x) ⊙ up_proj · x)
```

After permutation,

```
y' = (down_proj · Pᵀ) · (SiLU(P · gate_proj · x) ⊙ (P · up_proj · x))
   = down_proj · Pᵀ · P · (SiLU(gate_proj · x) ⊙ (up_proj · x))      (∵ ⊙ is index-wise)
   = y                                                                (∵ Pᵀ·P = I)
```

So Operation A is **bit-exact under exact arithmetic** and effectively
exact under bf16/fp32. The model file's tensor shapes are unchanged.
**Any inference engine that accepts the original checkpoint will accept the
permuted checkpoint with no code or config changes.**

For attention (GQA on Qwen Coder 32B), the analogous permutation is over
*query heads* in `q_proj` and `o_proj` (groups of 128 columns / rows).
Permuting Q heads while keeping K/V grouping intact is fine because the
GQA group assignment `q_head → kv_head` is determined by integer division
(`kv_head = q_head // (n_q / n_kv)`). If the permutation crosses GQA
group boundaries, the K/V mapping must be recomputed — easy if you
preserve the boundary, harder if you intermix.

**Operation B — truncation (structure-changing)**

After permutation, drop the last `(1 - keep_fraction) × d_ffn` rows /
columns. The intermediate dimension now changes from 27 648 to ~22 118.

This **does** change the model's structure:

- `config.json`'s `intermediate_size` must be updated.
- The checkpoint's tensor shapes change.
- HuggingFace `from_pretrained` will load the modified checkpoint without
  code changes, *but only if* the config is consistent with the weight
  shapes. The Transformers `Qwen2MLP` does `gate = nn.Linear(d_model,
  intermediate_size)` etc.; as long as the config matches, the layer
  builds correctly.
- **Tensor-parallel inference** (vLLM, TGI, TensorRT-LLM, Megatron):
  these usually require `intermediate_size % TP_size == 0`, often with an
  additional alignment to 64 or 128 for cuBLAS. Truncating to `22 118`
  works for TP=2 (22 118 % 2 = 0) but not 4, 8, 16 cleanly. The keep
  fraction must therefore be *quantised* to a TP-friendly multiple.
  Currently the doc does not enforce this.
- **CUDA Graphs / `torch.compile` caches**: shape change invalidates the
  compile, requiring a recompile on first inference after truncation.

So the answer to the user's question is:

> **Operation A is "just a reshuffle" and is fully compatible with the LLM
> structure.** The serialised checkpoint is structurally identical; existing
> tooling loads it transparently.
>
> **Operation B is a structural change** in the same sense as any other
> structural pruning method (SliceGPT, LLM-Pruner): one config field
> changes, one or more tensors shrink, and TP / quantisation kernels
> require re-alignment.

### 4.2 What is *not* swapped (and why)

To avoid future confusion, it's worth listing what is **not** permuted /
modified:

- `hidden_size = 5 120` is invariant. No permutation may be applied to the
  residual stream without applying the same permutation to every layer
  *and* to the embedding, LM head, RoPE table, RMSNorm weights, and any
  cached states — at which point Ablirotate becomes SliceGPT.
- `num_hidden_layers = 64` is invariant. Layer-level removal would be a
  separate "depth pruning" step; it is mentioned in `§5.3 Option C` of the
  interleaved spec but not implemented.
- LayerNorm / RMSNorm parameters along the hidden dim are not touched.
- Rotary embeddings are not touched.
- KV cache layout is not touched (good — vLLM-friendly).

### 4.3 Pitfall: permutation under quantisation is *not* free

`APPROACH.md` and `AUTONOMOUS_AGENT.md` both casually mix "numerically
equivalent" with "compatible with 4-bit quantisation". They are not.

In int8 / int4 per-channel quantisation, each output channel of a Linear
layer carries a per-channel scale `s_i` (and zero-point `z_i`). When you
permute output channels of `gate_proj` (rows), the per-channel scales
travel with the rows — fine, *if* you also permute the corresponding
scale tensor. The pruner code must therefore touch:

- `gate_proj.weight` (or `gate_proj.qweight`)
- `gate_proj.scales` and `gate_proj.zeros` (if AWQ/GPTQ)
- For symmetric int8: `gate_proj.scale` per-row tensor.

For *inputs* of `down_proj` (columns), the per-input-channel scales are
typically **not** carried because the scale lives with the *output*
channels of the previous op. This means re-quantisation along the
permuted axis is required when you permute `down_proj`'s columns —
which is precisely what Ablirotate does. The repo's `MatrixDefragmenter`
should be audited for whether it handles GPTQ/AWQ-quantised modules; if
not, the "QLoRA-style 4-bit base + bf16 specific tail" path in
`AUTONOMOUS_AGENT.md §6.1` will silently produce a degraded model.

---

## 5. Question 3 — the harness for dynamic-size models

This is the most ambitious and the most under-specified part of the
proposal. I'll separate it into the four sub-questions implicit in the
brief.

### 5.1 What does "temperature" mean here?

The repo uses "temperature" in two distinct senses, and conflates them.

**(a) Sampling temperature** (the LM-head softmax divisor). This affects
which token is emitted given the model's final logits. It does **not**
affect any internal weight or activation pathway.

**(b) Cold-mode bias** (`§2.2` of `APPROACH.md`). A "cold" neuron has a
large negative bias added to its gate, so under typical inputs
`SiLU(gate − λ)·up` ≈ 0 and the neuron is silent.

The doc says cold neurons "allow re-activation at elevated sampling
temperatures". As written, this is **incorrect**: the SwiGLU gate's bias
is a static weight, not a quantity that the sampler scales. There is no
mechanism by which a higher sampling temperature feeds back into the
gate's bias.

What the doc *probably* means is the related but distinct idea:

- A neuron that is statistically silent on the calibration set may still
  fire on rare/atypical inputs that lie far from the calibration mean.
- Increasing the sampling temperature increases the variance of the
  emitted-token distribution, which over many turns drives the model into
  more atypical regions of input space, where cold neurons may again
  exceed their bias.

This is plausible but is a second-order effect, not a direct mechanism. To
obtain a *direct* "harness-controlled" gating mechanism, the architecture
must be augmented with a per-neuron-group activation knob `α_i ∈ [0, 1]`
that the harness can set at runtime, e.g.:

```python
intermediate = SiLU(gate_proj(x)) * up_proj(x) * alpha           # broadcast over batch
```

with `alpha` stored as a non-trainable buffer. The harness now has a
thermostat: write zeros into the indices it wants to silence, ones into
the indices it wants live. This is cheap (one extra elementwise multiply
per MLP, ~0.1 % overhead) and it gives true runtime control without
weight surgery.

The same mechanism naturally implements **graduated cold-mode**:

| Forget age | α |
|---|---|
| 0 | 1.0 |
| 1–2 (soft) | 0.5 |
| 3–5 (cold) | 0.05 |
| ≥ 6 (hard) | weight tensor truncated |

This is the recommended re-formalisation: "Cold mode" should be `α`-based,
not bias-based, so that the harness — not the sampler — controls the knob.

### 5.2 Activation tracking — the cheap part

Activation tracking is the *easiest* part of the harness:

- Forward hooks on `down_proj` (PyTorch
  `register_forward_pre_hook`) capture the input tensor of shape
  `(batch, seq, d_ffn)` per layer.
- Threshold and accumulate into an `int32[d_ffn]` counter per layer.
- Memory: 64 layers × 27 648 × 4 bytes ≈ 7 MB total. Negligible.
- For multi-category tracking (differential abliteration), C categories
  multiply this by C. Still in the tens of MB.

Throughput cost is the real concern, not memory:

- A naive Python hook adds ~50–200 µs of CPU/GPU sync per layer per call.
  Over 64 layers and a 1 K-token sequence this is ~5–15 ms — a 5 %–15 %
  inference penalty. For development this is fine; for an
  *always-on* tracker on a production agent it is not.
- Production-grade tracking should fuse the threshold-and-count into the
  Linear kernel itself (a Triton kernel that, for free, accumulates a
  per-channel count of "did this row's pre-activation exceed τ?"). This
  is straightforward to write but is not in the repo.
- An intermediate compromise: `torch.compile` the MLP block with the hook
  inlined, eliminating the Python dispatch overhead.

Recommendation: add a `low_overhead=True` mode to `ActivationTracker`
that uses a fused kernel and quantises the count to int8 per window
(saturating at 255). Per-layer overhead drops to <1 % and the tracker
becomes always-on-cheap.

### 5.3 Dynamic size — adding / removing neurons mid-run

This is the hard part. PyTorch was not designed for hot-resizable
modules, and the repo glosses over what changes when you add or remove
neurons inside a running training loop.

The state that has to be kept consistent:

1. **`nn.Module` parameters.** Replacing `gate_proj.weight` with a larger
   tensor is fine *if* you also re-register it with the optimizer.
   `optimizer.add_param_group(...)` works for new tensors but does not
   re-key existing tensors. Practical pattern: rebuild the optimizer from
   scratch after each grow event, copying state for the persisting
   parameters and zero-init'ing for the new rows.
2. **Adam moments (`m`, `v`).** Must be expanded to match the new shape.
   Zero-init for new rows; copy old moments for existing rows. `Adafactor`'s
   factored state has the same requirement at row + column granularity.
3. **Gradient hooks.** `register_hook` is keyed on the parameter tensor
   id, so replacing the tensor invalidates the hook. Re-register after
   every shape change.
4. **`torch.compile` cache / CUDA Graphs.** Both invalidate on shape
   change. A grow event therefore costs one recompile (1–30 s for an
   8B–32B model). This rules out grow events at every step but is fine
   for the proposed cycle (every few hours).
5. **Distributed sharding.** FSDP / DeepSpeed ZeRO compute their shard
   plans at init. Mid-run resharding requires a checkpoint-and-reload
   cycle. Single-GPU runs are unaffected.
6. **KV cache.** Untouched by MLP grow/shrink. Untouched by attention
   *head* grow only if the head dim is fixed (it is in Qwen2 / Gemma).

So the dynamic-size loop **is** implementable in stock PyTorch on a
single GPU, with ~tens of seconds of pause per grow / shrink event for
recompilation and optimizer rebuild. It is *not* implementable as a hot
operation on a multi-GPU FSDP / vLLM / TGI deployment without
substantial extra engineering (or a process restart, which the repo
mentions as an option — that's the right pragmatic answer for v1).

### 5.4 Disk-resident layers — bandwidth reality check

The brief asks about "layers stored on disk and dynamically added or
removed". Two things have to be separated.

**(a) Disk-resident at *load* time, fully resident during execution.**
This is just lazy / staged loading: keep the layer file mmap'd, page it
into GPU when the model is rebuilt, and evict on swap-out. HuggingFace
`accelerate` already supports this via `device_map="auto"` with `offload_folder`.
DeepSpeed ZeRO-Infinity does the same for training. No fundamental
problem.

**(b) Disk-resident at *execution* time** (layer not in GPU memory when
the forward pass through it begins). This is the eyebrow-raising
variant. Numbers for Qwen Coder 32B:

- One transformer block (MLP + attention) ≈ 510 MB in bf16.
- NVMe sequential read: ~5 GB/s on a good consumer SSD, ~12 GB/s on
  PCIe 5 server-grade.
- Therefore disk → GPU transfer time per layer: 100 ms (consumer) /
  40 ms (server).
- Per-token forward through 64 layers, sequential disk paging: 6.4 s
  (consumer) / 2.6 s (server) **per token**.

Throughput-wise this is **catastrophic** — three orders of magnitude
slower than GPU-resident inference (which produces tokens in 30–100 ms).
The only regime where execution-time disk residency makes sense is
**ultra-long-tail layers**: layers that are needed in <1 % of forward
passes can sit on disk if the harness can predict their absence quickly.
That requires a *router* — i.e. you've reinvented MoE, but with a much
slower expert-fetch path than a Switch Transformer's HBM-resident
expert.

A more productive interpretation of the brief: **"layers can be evicted
to CPU / disk between cycles, and re-loaded on demand at cycle
boundaries (not per-token)."** This is straightforward and useful:

- After each interleaved-training cycle, `drop_specific` neurons that
  are scheduled for hard truncation can be moved to a "frozen disk
  archive" (preserving them for possible later resurrection if the task
  distribution shifts back).
- A future cycle that requires capacity expansion can then reload the
  archived neurons rather than zero-init'ing fresh ones — a form of
  "structural memory" between training phases.
- The cost is one I/O round-trip per cycle (~6 hours), totally negligible.

This is the regime in which the disk-resident idea is sound. The
per-token regime is not.

### 5.5 Compatibility with existing frameworks

**PyTorch core** — fully compatible. No fork required.
- Forward / pre-forward hooks: ✓
- Custom `nn.Module` subclasses for SwiGLU + α gate: ✓
- Gradient masking via `register_hook`: ✓
- `torch.compile` with shape-stable phases: ✓ (recompile per phase)

**HuggingFace Transformers** — ~95 % compatible.
- Loading permuted weights: transparent (Operation A).
- Loading truncated weights with updated `intermediate_size`: works; the
  generic `Qwen2ForCausalLM` reads the config and builds layers
  accordingly.
- The `Qwen2MLP` forward is a simple `down_proj(act_fn(gate(x))*up(x))`
  expression and is amenable to monkey-patching with the α gate without
  forking the library.

**Stock HF Trainer** — *not* a good fit for the cyclic loop.
- `Trainer.train()` assumes a single training phase with stable shapes
  and stable parameter set.
- The cyclic reorder/freeze/grow loop should be implemented as a custom
  outer loop calling `Trainer` (or raw PyTorch) for the inner training
  phase. This is a few hundred lines of orchestration code, no library
  fork.

**PEFT / LoRA** — orthogonal and complementary.
- A LoRA adapter attached to the *specific tail only* (`weight[U:]`)
  reduces the trainable parameter count further.
- This is the right way to reduce VRAM in §6 of `AUTONOMOUS_AGENT.md`,
  and is more honestly described as "LoRA on the specific tail" than
  "QLoRA-style universal + bf16 specific" (which mixes two axes — bit
  width vs. trainable mask).

**Inference servers (vLLM, TGI, TensorRT-LLM)** — partial compatibility.
- Loading a permuted/truncated checkpoint at server start: works.
- Hot-swapping the model mid-flight: not supported. A drain-and-restart
  is required. This is acceptable for the 6-hour cycle.
- The `intermediate_size` after truncation must be a multiple of the
  TP-size and ideally of 128 for cuBLAS alignment.
- vLLM specifically caches CUDA graphs per shape and per batch size;
  shape changes invalidate the cache and incur a one-time recompile on
  next request.

**Agentic frameworks (LangChain, LangGraph, AutoGen, smolagents)** —
fully compatible at the application level. They invoke the model through
a thin client API; the underlying model swap is invisible to them
provided the API endpoint is stable. The agent harness only needs to
tolerate *latency spikes* during cycle boundaries.

**The honest verdict on Q3**: the technique is testable today on real
models with substantial but routine engineering (a custom orchestrator,
the α-gate monkey-patch, a careful checkpoint pipeline). It does *not*
require a redesign of PyTorch or HuggingFace Transformers. The
"layers on disk, hot-swap mid-token" reading of the brief is not
practical at any throughput a user would tolerate; the "layers on disk
between cycles" reading is straightforward and worth implementing.

---

## 6. Cross-cutting criticisms

### 6.1 The activation-rate signal is weaker than alternatives

A binary threshold on a single tensor magnitude discards three sources
of information that competing methods exploit:

- **Magnitude.** A neuron that fires at 0.5 σ above threshold every
  token is treated identically to one that fires at 50 σ. Wanda's `|w|·‖x‖`
  preserves this.
- **Sign / direction.** A neuron whose output is sometimes +1 and
  sometimes –1 for the same input class may be a coding-relevant
  parity neuron; its rate looks identical to a noisy neuron.
- **Correlation with downstream loss.** Taylor / Fisher-style methods
  (Molchanov, EWC) approximate `∂L/∂a_i · a_i`, capturing "this neuron
  matters because the loss is sensitive to it". Activation rate captures
  only "this neuron fires sometimes".

For purely structural ranking of dead-vs-alive at high rate-thresholds
(< 5 %), the binary signal is fine — dead neurons are dead under any
metric. For the more interesting middle band (5 %–40 % activation
rate), the signal is too crude to separate "important but selective"
from "noisy and disposable". `APPROACH.md §3` acknowledges this in
passing ("at the cost of using a cruder importance proxy") but does not
quantify the cost.

A useful augmentation, still gradient-free: use Wanda's `‖x‖` aggregated
per output channel as a tie-breaker among neurons with similar
activation rates. This fits the existing tracker (the same forward hook
already has access to the activations) and would meaningfully sharpen
the ranking.

### 6.2 The "common" classification is threshold-fragile

In `INTERLEAVED_TRAINING.md §2`, *common* = active in ≥ 1 keep AND ≥ 1
drop category. Two adversarial regimes:

- **Threshold too tight.** Almost no neuron is active in *all* keep
  categories *and* in any drop category. `common` shrinks to a few
  hundred neurons; the freeze segment is small; the loop devolves into
  near-full fine-tuning.
- **Threshold too loose.** Almost every neuron fires somewhere in some
  category; `common` engulfs everything; the freeze segment is large;
  the loop barely trains.

The doc currently treats the activation threshold as "configurable" and
moves on. For the loop to be reliable, it needs:

- A principled threshold-selection method (e.g. fix `common` at a target
  fraction of total neurons, ~30 %, and back out the threshold).
- A monitor that flags when `|common|` changes by more than X % between
  cycles, indicating threshold or distribution drift.
- A grace zone: neurons within ε of the threshold are treated as
  "common" to avoid border thrash between cycles.

### 6.3 Frozen-prefix gradients still propagate through frozen weights

`§5.1` describes selective freezing via gradient hooks zeroing the
prefix grad. This is correct — the *parameter* update on the prefix is
zero. It does **not** mean the prefix is dynamically frozen during
forward / backward.

In particular:

- The frozen prefix's *outputs* still flow into the trainable tail's
  inputs and into the residual stream.
- The trainable tail's optimizer sees inputs that change distributionally
  over training as the tail itself learns. (Internal covariate shift
  inside the layer.) RMSNorm helps but does not eliminate this.
- The residual stream's downstream layers therefore see a
  distributionally drifting input even though their parameters are
  frozen. After enough updates, the frozen layers are operating on
  out-of-distribution residual states — quietly degrading the
  "universal" capabilities the freeze was supposed to preserve.

Mitigations the doc should consider:
- Add an L2 anchor on the *output* of the specific tail of each layer
  (regress against its initial value). Cheap and local.
- Periodically run a validation pass on a non-task distribution and
  abort the cycle if degradation > τ.
- Add a soft EWC-style penalty *only on the layers downstream of where
  the saturation expansion happened* (the most likely source of
  distributional drift).

### 6.4 The autonomous loop's "boundary import" is structurally fragile

`AUTONOMOUS_AGENT.md` proposes that the reference host computes
`universal_boundary.pt` (a `dict[layer_name → int]`) and exports it to
the agent host. Two failure modes:

- **Coordinate drift.** After cycle 1, the agent has *permuted* its
  weight matrices; the agent's neuron index `i` no longer corresponds to
  the reference's neuron index `i`. To use the boundary, the agent
  must keep a **persistent permutation map** `agent_index → reference_index`
  and translate the boundary every cycle. If this map is not maintained
  carefully, every imported boundary is wrong by the size of the latest
  permutation.
- **Capacity drift.** After Option A grow events, the agent has
  neurons that the reference does not have. The boundary file is silent
  about these, so they default to "trainable" — fine — but the
  *reference's* claim that "neuron j is universal" no longer makes
  sense if the agent has grown the layer. The boundary semantics must
  become "for the persistent prefix of size `min(d_agent, d_reference)`
  in shared coordinates, here is the universal cutoff".

Currently the doc does not specify either invariant. Without them the
import is an aliased pointer.

### 6.5 The "Cold mode" temperature story does not work as written

Already covered in §5.1. The fix is to drop the "elevated sampling
temperature" justification and replace it with the runtime-controlled α
gate. The motivation for keeping cold neurons in the model
(reversibility, recoverable capacity) is sound; only the mechanism is
wrong.

### 6.6 MoE compatibility — Gemma-3 27B is an awkward fit

`AUTONOMOUS_AGENT.md §5.1` lists Gemma-3 27B with an `intermediate_size`
of 40 960 and notes its MoE routing. Activation-rate semantics changes
fundamentally for MoE:

- A given expert is selected for only ~`top_k / num_experts` of tokens.
  Its raw activation rate is therefore bounded above by routing
  probability, regardless of the expert's *internal* importance.
- Two natural fixes — neither implemented:
  - **Conditional rate.** Compute `P(neuron active | expert was
    routed)`. This requires per-expert tracker state (mostly trivial
    extension).
  - **Routing-aware ranking.** Importance =
    `P(expert routed) × E[neuron active | routed]`. This is a proper
    estimate of the expected fraction of forward passes touching the
    neuron.
- Truncation of expert intermediate dim must respect MoE-specific
  constraints (Gemma 3's expert layers have block-sparse weight layouts
  in some kernels; not all kernels accept arbitrary `intermediate_size`).
- Differential abliteration across experts is more interesting than
  within experts: a "code expert" may be a `keep_specific` *expert*,
  not a `keep_specific` *neuron* of the dense FFN.

The repo would benefit from explicitly bracketing MoE as out of scope
for v1 of the dense pipeline, or from sketching the routing-aware
extension before claiming Gemma-3 27B as a target.

### 6.7 No quality gate — the elephant in the room

`APPROACH.md §5.1` says "Ablirotate does not measure task accuracy". This
is fine for a post-training tool that ships a pruned checkpoint and
asks the user to evaluate. It is **not** fine for an autonomous loop
that fires every 6 hours and may quietly degrade quality across hundreds
of cycles.

The loop *must* include:

- An automatic regression gate: a fixed eval set (small, fast,
  representative) run after every cycle. Reject the cycle's output if
  the gate fails by more than τ.
- A rollback mechanism: keep the prior cycle's checkpoint, and if the
  gate fails, revert.
- A drift detector: if the agent's calibration distribution has shifted
  significantly from the previous cycle (KS test on activation rates),
  reduce the truncation aggressiveness for that cycle.

This is sketched in `§3` ("Quality referee") of `AUTONOMOUS_AGENT.md`
but is not formalised. Given the autonomy claim, it should be the most
detailed, not the least.

---

## 7. Recommendations

In rough priority order:

1. **Disambiguate the two operations** (permutation vs truncation) in
   `APPROACH.md`. Most of the "is it just a reshuffle" confusion goes
   away once these are presented separately.

2. **Replace bias-based "Cold mode" with an α-gate buffer** in the
   forward path. Fix the temperature explanation accordingly.

3. **Quote wall-clock figures, not FLOP figures**, when describing
   inference speed-up. Distinguish prefill vs decode and bandwidth-
   vs compute-bound.

4. **Add a quantisation-permutation audit** to `MatrixDefragmenter`.
   If a Linear is GPTQ/AWQ-quantised, its scales/zeros tensors must
   be permuted alongside the weight tensor.

5. **Strengthen the activation signal** by mixing in Wanda's `‖x‖` as
   a magnitude-aware tie-breaker. Still gradient-free, much sharper
   ranking.

6. **Specify the boundary-import contract** in `AUTONOMOUS_AGENT.md`:
   persistent permutation map, capacity-drift handling, and a sanity
   check on every import.

7. **Hard-require an automatic eval gate** in the autonomous loop.
   Without it, "unattended" is not safe.

8. **Restate the disk-residency story** in terms of cycle-boundary
   eviction rather than per-token paging, with the bandwidth math made
   explicit so future readers do not chase the per-token interpretation.

9. **Bracket MoE explicitly** as v1.5 work, with a sketch of the
   routing-aware tracker extension. Conflating dense SwiGLU and Gemma-3
   MoE without a routing-aware tracker invites confusion.

10. **Document the threshold-selection method** for activation rate
    and `common` classification. A single configurable knob with no
    selection guidance is the most likely source of irreproducibility
    in any future replication.

---

## 8. Summary — what to take away

The Ablirotate approach is **defensible, structurally correct, and
implementable today on real models** for the post-training and
single-GPU interleaved-training cases. Its core insight (defragmentation
turns logical sparsity into hardware-realised speed-up) is the right
bet, and the differential-abliteration generalisation of refusal-direction
editing is genuinely novel framing.

Its main weaknesses are at the interfaces:

- The interface between activation-rate (a binary, threshold-based
  proxy) and the loss landscape (where importance actually lives).
- The interface between the autonomous loop and the rest of the
  system (no quality gate, fragile boundary import, no rollback).
- The interface between the proposed mechanism and existing
  inference / quantisation tooling (TP alignment, scale permutation,
  CUDA-graph invalidation), where the doc currently glosses over
  details that will bite during integration.
- The interface between the harness's intuitive "temperature knob"
  and what the model can actually be told — currently routed through
  a sampling-temperature claim that does not work, but easily fixable
  via an explicit α-gate.

Fixing the ten items in §7 would turn the design from "promising
research direction" into "well-specified engineering proposal". The
core ideas are worth that investment.
