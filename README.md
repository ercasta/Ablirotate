# Ablirotate
Techniques for shrinking LLMs

Ablirotate is a Python toolkit for creating **sparse**, task-specialised LLMs
from full-size transformer models.  The library is structured around two
complementary levels of operation:

1. **Post-training (gradient-free)** – shrink and specialise a pre-trained
   checkpoint using nothing but forward-pass statistics.  No labelled data,
   no loss function, no optimizer required.
2. **Interleaved training** – extend the post-training step into a cyclic
   training loop that alternates between gradient-free neuron reordering and
   selective gradient-based fine-tuning, keeping the *universal* core of the
   model frozen while the *task-specific* segment specialises.
3. **Agentic sandbox** – deploy a single model instance that self-specialises
   unattended on a target class of tasks (e.g. coding), guided by a frozen
   reference copy of the original model that anchors the universal neurons
   and acts as a quality referee.

### Approach at a glance

| Part | Name | One-line summary |
|------|------|-----------------|
| 1 | **Post-training** | Profile neuron activations with forward-pass hooks, then prune, reorder, and defragment weight matrices — no gradients, no labels, works on any checkpoint. |
| 2 | **Interleaved training** | Cycle between a gradient-free reorder phase (neuron sorting by universality) and a gradient-based fine-tune phase (only the task-specific neuron segment is updated, universal neurons are frozen). |
| 3 | **Agentic sandbox** | A self-contained agent loop where the model specialises autonomously on a task domain while a reference model keeps universal capabilities anchored; suitable for unattended, long-running deployment. |

---

## Part 1 – Post-training approach

### How it works

The post-training pipeline has four mechanisms that can be applied
independently or in sequence:

| Step | Mechanism | What it does |
|------|-----------|-------------|
| 1 | **Activation tracking** | Lightweight forward hooks record which MLP neurons and attention heads fire (and how often) as the model processes representative prompts. |
| 2 | **Activation-based pruning** | Neurons whose activation rate falls below a threshold are zeroed out, scaled down, or cold-gated. |
| 3 | **Matrix defragmentation** | Weight matrices are reordered so that the most active neurons are contiguous, allowing the dead tail to be truncated — giving a real wall-clock speed-up without sparse kernels. |
| 4 | **Differential abliteration** | The model is profiled separately on multiple prompt categories (e.g. Python code, Italian prose, COBOL). Neurons active only in unwanted categories are dropped; neurons universal across desired categories are preserved. |

Because every step is gradient-free, the pipeline can be applied to any
model checkpoint — including 30B+ parameter models — without a GPU cluster or
labelled data.

### Installation

```bash
pip install .          # runtime only
pip install ".[dev]"   # + pytest for running tests
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0, HuggingFace Transformers ≥ 4.35.

### Quick start – activation pruning and defragmentation

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ablirotate import ActivationTracker, ModelPruner, MatrixDefragmenter

model = AutoModelForCausalLM.from_pretrained("my-llm")
tokenizer = AutoTokenizer.from_pretrained("my-llm")

# 1. Track activations over typical usage -----------------------------------
tracker = ActivationTracker(model, threshold=0.0)

for prompt in my_typical_prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        model(**inputs)

# 2. Prune neurons below 10 % activation rate --------------------------------
pruner = ModelPruner(model, tracker.stats, rate_threshold=0.10)
pruned = pruner.prune(mode="soft")   # scales weights; shape unchanged
print(pruned)  # {layer_name: n_pruned, ...}

# 3. Defragment: sort neurons by activity and truncate dead tail -------------
defrag = MatrixDefragmenter(model, tracker.stats, keep_fraction=0.80)
result = defrag.defragment()
print(result)  # {layer_name: (original_size, new_size), ...}

tracker.remove_hooks()
```

### Quick start – differential abliteration

Identify neurons that are **universal** across desired tasks and drop those
that are specific only to unwanted ones.

```python
from ablirotate import ActivationTracker, ModelPruner, DifferentialAbliterator

tracker = ActivationTracker(model)
abliterator = DifferentialAbliterator(tracker, keep_rate_threshold=0.10)

# Profile each category, then call .record()
for lang in ["python", "javascript", "rust", "go", "java", "cpp", "csharp"]:
    for nl in ["english", "italian"]:
        for prompt in prompts[lang][nl]:
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                model(**inputs)
    abliterator.record(f"{lang}_{nl}")

# Keep neurons universally active in python+italian; drop cobol-only ones
keep_mask = abliterator.compute_keep_mask(
    keep_categories=["python_english", "python_italian"],
    drop_categories=["cobol_english"],
)

pruner = ModelPruner(model, tracker.stats, rate_threshold=0.0)
pruner.prune_to_mask(keep_mask, mode="hard")

tracker.remove_hooks()
```

`DifferentialAbliterator` classifies every neuron into four groups:

| Group | Condition | Treatment |
|-------|-----------|-----------|
| **common** | Active in ≥ 1 keep category **and** ≥ 1 drop category | Always preserved — central neurons used across all tasks. |
| **keep_specific** | Active only in keep categories | Preserved — reinforce desired capabilities. |
| **drop_specific** | Active only in drop categories | Removed — exclusively responsible for unwanted capabilities. |
| **neutral** | Inactive in all observed categories | Preserved. |

### Pruning modes

| Mode   | Effect |
|--------|--------|
| `soft` | Scale inactive neuron weights by `high_temp_scale` (default 0 → zero-out). Matrix **shape unchanged**. |
| `hard` | Permanently zero-out weights of inactive neurons. |
| `cold` | Zero-out weights AND inject a large negative bias so the neuron can only re-activate at high sampling temperatures. |

---

## Part 2 – Interleaved training

The post-training approach produces a fast, specialised model in a single
pass, but has no mechanism to acquire new knowledge.  The interleaved training
approach extends it into a **cyclic loop** that alternates between a
gradient-free reorder phase and a selective gradient-based training phase.

### Core idea

Every neuron in every layer is continuously classified as either *universal*
(active across all observed task distributions — the general reasoning core)
or *specific* (active only in keep or drop categories, or dormant).

```
universal  =  common
specific   =  keep_specific ∪ drop_specific ∪ neutral
```

The loop alternates between two phases:

```
╔══════════════════════════════════════════════════════════════════╗
║  REORDER PHASE (gradient-free)                                   ║
║  1. Profile the current model with DifferentialAbliterator.      ║
║  2. Sort neurons: common → keep_specific → neutral → drop_spec.  ║
║  3. Apply permutation via MatrixDefragmenter.                    ║
║  4. Optionally truncate the drop_specific tail.                  ║
║  5. Record boundary index U (last universal neuron per layer).   ║
╚══════════════════════════════════════════════════════════════════╝
         │
         ▼
╔══════════════════════════════════════════════════════════════════╗
║  TRAINING PHASE (gradient-based fine-tuning)                     ║
║  1. Freeze weight rows/columns [:U] (universal segment).         ║
║  2. Fine-tune only rows/columns [U:] (specific segment).         ║
║  3. If the specific segment saturates, append new capacity       ║
║     (extra FFN neurons, a new attention head, or a new block).   ║
║  4. Train for N steps on target task data.                       ║
╚══════════════════════════════════════════════════════════════════╝
         │
         ▼ (re-collect activation statistics)
         └──────────────────────── back to REORDER PHASE ──────────
```

The loop runs until task performance plateaus or a compute budget is
exhausted.

### Why it works

* **Stability** — universal neurons encode cross-task general knowledge;
  freezing them prevents catastrophic forgetting more cleanly than a
  regularisation penalty.
* **Efficiency** — only the specific segment (a small fraction of parameters)
  is updated, matching the convergence benefits observed with LoRA while
  keeping the separation *inside* the existing weight matrices with no
  inference overhead from extra modules.
* **Adaptability** — the universal/specific boundary is recomputed at every
  cycle, so neurons that generalise over time migrate to the frozen core
  automatically.

### Selective freezing via gradient hooks

```python
def freeze_universal_hook(grad, U):
    grad = grad.clone()
    grad[:U] = 0.0
    return grad

for name, param in model.named_parameters():
    if name in universal_boundary:
        U = universal_boundary[name]
        param.register_hook(lambda g, U=U: freeze_universal_hook(g, U))
```

The parameter shape is preserved across the reorder phase because the
permutation is numerically equivalent — no checkpoint surgery required.

### Capacity expansion

When the specific segment saturates (most neurons are already assigned to
an observed category and validation loss stops improving), new capacity can
be added:

| Option | What is added | When to use |
|--------|---------------|-------------|
| **A – grow FFN width** | New rows in `gate_proj` / `up_proj`, new columns in `down_proj`, initialised to near-zero. | First choice; least disruptive. |
| **B – add attention head** | New Q head appended (zero-initialised). | When saturation is concentrated in attention layers. |
| **C – insert transformer block** | New near-identity block inserted after the most-saturated layer. | Last resort; analogous to Net2Net progressive growing. |

### Relationship to the post-training toolkit

| Existing component | Role in the interleaved loop |
|--------------------|------------------------------|
| `ActivationTracker` | Collects statistics for reorder and expansion decisions. |
| `DifferentialAbliterator.classify_neurons` | Defines the universal / specific boundary. |
| `DifferentialAbliterator.prioritized_indices` | Produces the permutation for the reorder phase. |
| `MatrixDefragmenter` | Applies the permutation; tail truncation removes drop_specific neurons. |
| `ModelPruner` | Optionally hard-prunes the drop_specific tail. |

The **reorder phase is fully supported today** by the existing toolkit.  The
training phase additionally requires gradient masking hooks, a saturation
metric, expansion utilities, and a loop orchestrator (see
`INTERLEAVED_TRAINING.md` for full algorithm details).

---

## Part 3 – Autonomous sandbox agent

A practical extension of the interleaved loop where a **single model instance
runs unattended** on a specific class of tasks (e.g. Python + Git coding)
while a separate, unmodified copy of the original model handles everything
else.  The original model provides the *universal neuron* reference — its
activation profile on unrelated prompts defines which neurons must remain
frozen in the agent — and acts as a quality referee.

The autonomous loop follows the same four-phase structure as the interleaved
training loop, driven by a timer rather than a human operator:

| Phase | What happens |
|---|---|
| **Observation** | Agent handles tasks; `ActivationTracker` accumulates stats silently. Reference model profiles unrelated prompts. |
| **Reorder** | `DifferentialAbliterator` + `MatrixDefragmenter` permute weight matrices; universal neurons migrate to the frozen prefix `[:U]`. |
| **Graduated pruning** | Drop-specific neurons are soft-pruned → cold-gated → hard-pruned over successive cycles according to a `NeuronForgetSchedule`. |
| **Micro-training** | Only the specific segment `[U:]` is trained on a rolling replay buffer of recent agent tasks. |

### Hardware quick-reference

| Model | Minimum VRAM (bfloat16) | Minimum VRAM (4-bit base) |
|---|---|---|
| Gemma 4 26–27B MoE | ~40 GB (single A100 80 GB or 2× A100 40 GB) | ~20–24 GB (single A100 40 GB) |
| 8B (Llama-3, Gemma-3, Qwen2.5) | ~24 GB (single 3090) | ~12 GB (single 3080 Ti / 4070) |
| 2B (Gemma-3-2B, Phi-3-mini) | ~8–10 GB (single 3080 Ti) | ~6 GB (single 3070 Ti) |

### Minimising VRAM at the cost of training speed

1. **4-bit quantise the universal prefix** — halves VRAM for the frozen part.
2. **Neuron micro-batch training** — unfreeze and update `M` neurons at a time;
   optimizer state shrinks from `O(N_specific)` to `O(M)`.
3. **Layer-serial training** — train one layer at a time; allocates optimizer
   state only for that layer.
4. **Gradient checkpointing** on the universal segment — recompute activations
   during backward, saving ~40% activation memory.

→ See [`AUTONOMOUS_AGENT.md`](AUTONOMOUS_AGENT.md) for the full design,
detailed hardware tables, RAM-minimisation techniques, and a list of new
components required to implement the approach.

---

## API reference

### `ActivationTracker(model, threshold, track_mlp, track_attention, mlp_submodule_name, attn_submodule_name)`

Registers forward hooks on every sub-module whose name ends with
`mlp_submodule_name` (default `"act"`) or contains `attn_submodule_name`
(default `"attn"`).

- **`.stats`** – `dict[str, ActivationStats]`
- **`.reset()`** – clear all counters
- **`.remove_hooks()`** – detach all hooks (also called on `__exit__`)
- **`.aggregate_activation_mask(rate_threshold)`** – `dict[str, BoolTensor]`
- **`.summary()`** – `dict[str, float]` mean activation rate per layer

### `ActivationStats`

- **`.activation_rate`** – `Tensor` of shape `(n_units,)`, values in [0, 1]
- **`.least_active_indices(top_k)`** / **`.most_active_indices(top_k)`**

### `ModelPruner(model, stats, rate_threshold, high_temp_scale)`

- **`.prune(mode)`** → `dict[str, int]` neurons pruned per layer
- **`.prune_to_mask(keep_mask, mode)`** → same, using external mask

### `MatrixDefragmenter(model, stats, keep_fraction)`

- **`.defragment()`** → `dict[str, (orig, new)]`
- **`.partition(rate_boundary)`** → `dict[str, {'hot': Tensor, 'cold': Tensor}]`
- **`.compute_permutation(layer_name)`** → sorted index tensor

### `DifferentialAbliterator(tracker, keep_rate_threshold, drop_rate_threshold)`

- **`.record(category)`** – snapshot current tracker stats; reset tracker
- **`.classify_neurons(keep_categories, drop_categories)`** → `dict[str, dict[str, BoolTensor]]`
- **`.compute_keep_mask(keep_categories, drop_categories)`** → `dict[str, BoolTensor]`
- **`.prioritized_indices(keep_categories, drop_categories)`** → `dict[str, LongTensor]`
- **`.activation_overlap(cat_a, cat_b)`** → Jaccard overlap per layer
- **`.summary()`** → mean activation rate per category per layer

---

## Further reading

- [`APPROACH.md`](APPROACH.md) – detailed theory, mechanism descriptions, and
  comparison with magnitude pruning, SparseGPT, Wanda, LoRA, and other methods.
- [`INTERLEAVED_TRAINING.md`](INTERLEAVED_TRAINING.md) – full algorithm
  specification for the interleaved reordering–freezing training loop,
  convergence analysis, and comparison with EWC, PackNet, and MoE approaches.
- [`AUTONOMOUS_AGENT.md`](AUTONOMOUS_AGENT.md) – design for an unattended
  sandbox agent that self-specialises through periodic retraining, including
  hardware requirements for Gemma 4 27B, 8B, and 2B models and a catalogue
  of VRAM-minimisation techniques.

---

## Running tests

```bash
pytest
```

