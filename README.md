# Ablirotate
Techniques for shrinking LLMs

Ablirotate is a Python toolkit for creating **sparse**, task-specialised LLMs
from full-size transformer models.  It does this through three complementary
mechanisms:

1. **Activation tracking** – lightweight forward hooks record which MLP
   neurons and attention heads fire (and how often) as the model is used.
2. **Activation-based pruning** – neurons that rarely activate are removed or
   "cold-gated" (requiring high temperature to re-engage).
3. **Matrix defragmentation** – weight matrices are reordered so that the most
   active neurons are grouped together, enabling the dead tail to be sliced off
   cleanly without sacrificing speed.
4. **Differential abliteration** – run the model on multiple prompt categories
   (e.g. Python code, JavaScript code, Italian prose, Japanese prose), keep
   neurons that fire universally across the desired categories, and drop those
   that are specific only to unwanted ones.

---

## Installation

```bash
pip install .          # runtime only
pip install ".[dev]"   # + pytest for running tests
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0, HuggingFace Transformers ≥ 4.35.

---

## Quick start

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

---

## Differential abliteration

Identify neurons that are **universal** across desired tasks and drop those
that are specific to unwanted ones.

```python
from ablirotate import ActivationTracker, ModelPruner, DifferentialAbliterator

tracker = ActivationTracker(model)
abliterator = DifferentialAbliterator(tracker, keep_rate_threshold=0.10)

# Run inference for each category, then call .record()
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

### Pruning modes

| Mode   | Effect |
|--------|--------|
| `soft` | Scale inactive neuron weights by `high_temp_scale` (default 0 → zero-out). Matrix **shape unchanged**. |
| `hard` | Permanently zero-out weights of inactive neurons. |
| `cold` | Zero-out weights AND inject a large negative bias so the neuron can only re-activate at high sampling temperatures. |

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
- **`.compute_keep_mask(keep_categories, drop_categories)`** → `dict[str, BoolTensor]`
- **`.activation_overlap(cat_a, cat_b)`** → Jaccard overlap per layer
- **`.summary()`** → mean activation rate per category per layer

---

## Running tests

```bash
pytest
```

