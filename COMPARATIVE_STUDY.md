# Comparative Study: Interleaved Training vs. Baseline Fine-Tuning Methods

## Purpose

This document describes a practical, fully automated study that compares the
**Ablirotate interleaved training approach** against several baseline fine-tuning
and post-training methods on a Python + Git coding task, using
[SWE-bench](https://swe-bench.github.io) as the evaluation harness.

The study targets **small models** (2B and 8B parameter tier), requires no
human intervention once started, and is designed to run to completion on a
single consumer GPU (24 GB VRAM for the 8B tier; ≤ 12 GB for the 2B tier).

---

## 1. Research Questions

| # | Question |
|---|---|
| RQ1 | Does interleaved training achieve a better SWE-bench resolve rate than LoRA and full-SFT at the **same final model size**? |
| RQ2 | Does post-training pruning alone (no gradient steps) improve or degrade resolve rate compared to the unmodified baseline? |
| RQ3 | How does the resolve-rate vs. model-size Pareto front differ across the five conditions? |
| RQ4 | Does the 2B tier replicate the ranking order observed for the 8B tier? |

---

## 2. Experimental Conditions

Five conditions are applied to each base model.  Each condition produces a
checkpoint plus metadata (parameter count, file size, inference latency).

| ID | Condition | Description |
|----|-----------|-------------|
| **C0** | **Baseline** | Unmodified pre-trained model; no fine-tuning, no pruning. |
| **C1** | **Post-training only** | Ablirotate pruning + matrix defragmentation; no gradient step. |
| **C2** | **Full supervised fine-tuning (SFT)** | All parameters updated with next-token prediction loss on coding data. |
| **C3** | **LoRA SFT** | LoRA adapters (rank 64) applied to all linear projections; base weights frozen. |
| **C4** | **Interleaved training** | Ablirotate cyclic loop: reorder → selective freeze fine-tune → repeat. |

**C1** is the control that measures the benefit of structural pruning alone.
**C2** and **C3** are the practical fine-tuning baselines most practitioners
use today.  **C4** is the proposed approach.

All gradient-based conditions (**C2**, **C3**, **C4**) use the same training
dataset and the same total compute budget (measured in GPU-hours), so
differences in resolve rate are attributable to the training *method*, not to
data or compute.

---

## 3. Models

| Tier | Model | Parameters | HuggingFace ID |
|------|-------|-----------|----------------|
| 8B | Llama-3.1-8B-Instruct | 8.0B | `meta-llama/Llama-3.1-8B-Instruct` |
| 2B | Gemma-3-2B-IT | 2.0B | `google/gemma-3-2b-it` |

Both models are instruction-tuned to simplify prompt engineering in the
SWE-bench agent harness.  The experiment can be extended to other checkpoints
(e.g. Qwen2.5-Coder-7B-Instruct) by updating the model IDs in the run config.

---

## 4. Datasets

### 4.1 Fine-Tuning / Profiling Data

| Dataset | Use | Source |
|---------|-----|--------|
| `SWE-bench/SWE-bench_train` | Training split for C2, C3, C4 | HuggingFace Datasets |
| CodeSearchNet Python (`code_search_net`, `python`) | Activation profiling calibration set for C1 and the reorder phases of C4 | HuggingFace Datasets |
| Git-operation synthetic prompts (generated offline) | Git-specific prompts for the `"code_git"` profiling category | Included in this repo (see `data/git_prompts.jsonl`) |

**SWE-bench train split** contains approximately 19,000 historical GitHub
issues paired with their ground-truth patches.  Each record has a `problem_statement`
field (natural language issue description) and a `patch` field (unified diff).
The fine-tuning objective is next-token prediction on the concatenated
`[problem_statement] → [patch]` sequence.

### 4.2 Evaluation Data

[SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)
(500 verified instances) is the primary evaluation set.  The full SWE-bench
test split (2,294 instances) is used for a secondary run if compute budget
permits.

---

## 5. Metrics

| Metric | Definition | Computed by |
|--------|-----------|-------------|
| **Resolve rate** (primary) | % of SWE-bench instances where the generated patch passes all tests | SWE-bench evaluation harness |
| **Model file size** | Checkpoint size on disk (GB, bfloat16 or 4-bit) | `os.path.getsize` |
| **Active parameter count** | Non-zero parameter count after pruning | `sum(p.count_nonzero() for p in model.parameters())` |
| **Median inference latency** | Tokens / second at batch size 1 | Measured during evaluation loop |
| **Peak training VRAM** | Maximum GPU memory during the training phase | `torch.cuda.max_memory_allocated()` |

All metrics are written to `results/{model_tier}/{condition}/metrics.json` for
later aggregation.

---

## 6. Automated Pipeline Overview

The entire study runs without human intervention by invoking a single entry
point:

```
python study/run_study.py --config study/config.yaml
```

The orchestrator performs the following top-level steps for every
`(model, condition)` combination in the config:

```
1.  Download / verify model checkpoint
2.  Prepare fine-tuning data (tokenise, split)
3.  Run condition-specific training / post-training pipeline
4.  Save checkpoint + log metrics
5.  Run SWE-bench evaluation on the saved checkpoint
6.  Append results to results/summary.csv
```

Steps are cached: if a checkpoint for a given `(model, condition)` already
exists, the training step is skipped and the evaluation resumes from the
saved checkpoint.

---

## 7. Step-by-Step Implementation

### 7.1 Activation Profiling (C1 and C4)

The profiling step is shared between post-training (C1) and the reorder phases
of interleaved training (C4).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from ablirotate import ActivationTracker, DifferentialAbliterator
import torch, json

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tracker = ActivationTracker(model)
abliterator = DifferentialAbliterator(tracker)

# Profile: Python + Git code (keep category)
for prompt in code_prompts:          # ~512 samples from CodeSearchNet + git_prompts.jsonl
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**inputs)
abliterator.record("code_python_git")

# Profile: non-code prose (drop category — only needed for differential pruning)
for prompt in prose_prompts:         # ~128 samples from a held-out English text set
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**inputs)
abliterator.record("prose_en")

tracker.remove_hooks()
```

### 7.2 Condition C1 — Post-Training Only

```python
from ablirotate import MatrixDefragmenter, ModelPruner

# Hard-prune drop_specific neurons (≥ 1 observed prose-only category)
keep_mask = abliterator.compute_keep_mask(
    keep_categories=["code_python_git"],
    drop_categories=["prose_en"],
)
pruner = ModelPruner(model, tracker.stats, rate_threshold=0.0)
pruner.prune_to_mask(keep_mask, mode="hard")

# Defragment: reorder so active neurons are contiguous; keep 85 % of width
priority_order = abliterator.prioritized_indices(
    keep_categories=["code_python_git"],
    drop_categories=["prose_en"],
)
defrag = MatrixDefragmenter(model, tracker.stats, keep_fraction=0.85)
defrag.defragment()  # applies priority_order permutation internally

model.save_pretrained(f"checkpoints/c1_{model_tier}")
```

### 7.3 Condition C2 — Full SFT

Standard next-token-prediction fine-tuning with all parameters trainable.

```python
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

training_args = TrainingArguments(
    output_dir=f"checkpoints/c2_{model_tier}",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,          # effective batch size 16
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=50,
    save_strategy="epoch",
    # compute_budget_hours enforced by an early-stopping callback
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=swe_bench_train_tokenised,
    data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
)
trainer.train()
```

### 7.4 Condition C3 — LoRA SFT

LoRA adapters are applied to all Q, K, V, O, gate, up, and down projections
using [PEFT](https://github.com/huggingface/peft).

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # typically ~2–3 % of total

# Same TrainingArguments as C2; same training loop
# After training, merge adapters for fair size comparison:
model = model.merge_and_unload()
model.save_pretrained(f"checkpoints/c3_{model_tier}")
```

> **Note on fair comparison:** adapter merging produces a dense checkpoint with
> the same architecture as the base model.  For the model-size metric, both the
> merged checkpoint *and* the adapter-only checkpoint (base + adapter files) are
> recorded, since deployed systems often keep them separate.

### 7.5 Condition C4 — Interleaved Training

The interleaved loop runs for a fixed number of cycles (default **4 cycles**,
configurable).  Each cycle consists of:

1. **Reorder phase** — profiling + permutation (Section 7.1 + 7.2 steps 1-2).
2. **Training phase** — selective fine-tuning with universal neurons frozen.

The same total number of gradient steps is used as in C2 and C3, distributed
evenly across the 4 training phases.

```python
from ablirotate import ActivationTracker, DifferentialAbliterator, MatrixDefragmenter

N_CYCLES     = 4
STEPS_TOTAL  = total_steps_c2          # same budget as C2
STEPS_CYCLE  = STEPS_TOTAL // N_CYCLES

for cycle in range(N_CYCLES):

    # ── Reorder phase ─────────────────────────────────────────────────────
    tracker  = ActivationTracker(model)
    abliterator = DifferentialAbliterator(tracker)
    profile_model(model, tokenizer, code_prompts, "code_python_git", abliterator)
    profile_model(model, tokenizer, prose_prompts, "prose_en", abliterator)

    priority_order = abliterator.prioritized_indices(
        keep_categories=["code_python_git"],
        drop_categories=["prose_en"],
    )
    defrag = MatrixDefragmenter(model, tracker.stats, keep_fraction=0.85)
    defrag.defragment()

    # Record universal boundary per layer (index U of last universal neuron)
    universal_boundary = compute_universal_boundary(abliterator)
    tracker.remove_hooks()

    # ── Training phase ─────────────────────────────────────────────────────
    # Attach gradient hooks that zero the universal prefix
    hooks = []
    for name, param in model.named_parameters():
        if name in universal_boundary:
            U = universal_boundary[name]
            h = param.register_hook(
                lambda g, U=U: zero_prefix(g, U)
            )
            hooks.append(h)

    run_training(model, tokenizer, swe_bench_train_tokenised,
                 steps=STEPS_CYCLE, lr=2e-5 * (0.7 ** cycle))

    for h in hooks:
        h.remove()

model.save_pretrained(f"checkpoints/c4_{model_tier}")
```

Helper `zero_prefix`:

```python
def zero_prefix(grad, U):
    g = grad.clone()
    g[:U] = 0.0
    return g
```

Helper `compute_universal_boundary`:

```python
def compute_universal_boundary(abliterator):
    groups = abliterator.classify_neurons(
        keep_categories=["code_python_git"],
        drop_categories=["prose_en"],
    )
    boundary = {}
    for layer, g in groups.items():
        # Universal neurons = common; these are placed at position [:U] after defrag
        boundary[layer] = int(g["common"].sum().item())
    return boundary
```

---

## 8. SWE-Bench Evaluation Harness

### 8.1 Why SWE-bench

SWE-bench tests an agent's ability to resolve real GitHub issues — a task that
requires understanding a repository's existing code, writing a plausible patch,
and passing the repository's existing test suite.  It measures coding capability
holistically rather than scoring completion of isolated snippets, and its
automated test runner removes the need for human evaluators.

The [Verified subset](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)
(500 instances) was filtered by human annotators to remove ambiguous or
under-specified issues, giving more reliable signal from a smaller evaluation
set.

### 8.2 Agent Setup

Each condition's checkpoint is wrapped with
[SWE-agent](https://github.com/SWE-agent/SWE-agent) using the
`AstroContext` scaffold.  The agent is given:

- the full repository at the commit preceding the issue, mounted as a Docker
  volume,
- the issue text from `problem_statement`,
- a 15-minute wall-clock timeout per instance.

The agent uses **greedy decoding** (`temperature=0`, `do_sample=False`) so
that results are deterministic and comparable across conditions without
multiple sampling runs.

### 8.3 Running the Evaluation

```bash
# Example for condition C4, 8B tier
python -m sweagent.run \
    --agent.model.name  checkpoints/c4_8b \
    --agent.model.args  '{"temperature": 0, "max_new_tokens": 4096}' \
    --env.repo_path     /repo \
    --eval.dataset      princeton-nlp/SWE-bench_Verified \
    --eval.split        test \
    --output_dir        results/8b/c4
```

The evaluation harness writes one JSON file per instance to `output_dir` and
a `result.jsonl` summary with `resolved: true/false` per instance.

### 8.4 Aggregation

```python
import json, glob, pandas as pd

rows = []
for path in glob.glob("results/*/*/result.jsonl"):
    tier, cond = path.split("/")[1], path.split("/")[2]
    with open(path) as f:
        records = [json.loads(l) for l in f]
    resolved = sum(r["resolved"] for r in records)
    rows.append({
        "tier": tier, "condition": cond,
        "n_instances": len(records),
        "n_resolved": resolved,
        "resolve_rate": resolved / len(records),
    })

df = pd.DataFrame(rows)
df.to_csv("results/summary.csv", index=False)
print(df.sort_values(["tier", "resolve_rate"], ascending=[True, False]).to_string())
```

---

## 9. Directory Structure

```
study/
├── config.yaml                  # Master config (model IDs, dataset names, hyperparams)
├── run_study.py                 # Orchestrator: iterates over (model, condition) pairs
├── conditions/
│   ├── c0_baseline.py           # No-op: just save the checkpoint and evaluate
│   ├── c1_post_training.py      # Profiling + pruning + defrag
│   ├── c2_full_sft.py           # Full supervised fine-tuning
│   ├── c3_lora_sft.py           # LoRA fine-tuning + merge
│   └── c4_interleaved.py        # Cyclic reorder + selective SFT loop
├── shared/
│   ├── data_prep.py             # Download + tokenise SWE-bench train + CodeSearchNet
│   ├── profiling.py             # profile_model() helper
│   ├── metrics.py               # compute_metrics() and log_metrics()
│   └── evaluation.py            # Wrapper around sweagent CLI
data/
├── git_prompts.jsonl            # ~256 Git-operation prompts for profiling
└── prose_prompts.jsonl          # ~128 English-prose prompts (drop-category profiling)
checkpoints/                     # One sub-directory per (condition, tier)
results/
├── 8b/
│   └── {c0..c4}/
│       ├── metrics.json         # Size, latency, VRAM
│       └── result.jsonl         # Per-instance SWE-bench outcomes
├── 2b/
│   └── {c0..c4}/
└── summary.csv                  # Aggregated resolve rates + model sizes
```

---

## 10. Configuration File

```yaml
# study/config.yaml

models:
  - id: meta-llama/Llama-3.1-8B-Instruct
    tier: 8b
    dtype: bfloat16
  - id: google/gemma-3-2b-it
    tier: 2b
    dtype: bfloat16

conditions: [c0, c1, c2, c3, c4]

data:
  swe_bench_dataset: princeton-nlp/SWE-bench
  swe_bench_split: train
  eval_dataset: princeton-nlp/SWE-bench_Verified
  eval_split: test
  codesearchnet_subset: python
  n_code_profiling_samples: 512
  n_prose_profiling_samples: 128

training:
  total_steps: 3000               # same budget for C2, C3, C4
  per_device_batch_size: 2
  gradient_accumulation_steps: 8  # effective batch 16
  learning_rate: 2.0e-5
  lr_scheduler: cosine
  warmup_ratio: 0.05
  lora_rank: 64
  lora_alpha: 128
  n_interleaved_cycles: 4         # C4 only; steps split evenly across cycles

ablirotate:
  keep_fraction: 0.85
  pruning_mode: hard
  keep_categories: [code_python_git]
  drop_categories: [prose_en]
  saturation_threshold: 0.85      # triggers expansion in C4 if needed
  expansion_option: A             # grow FFN width first

evaluation:
  timeout_minutes: 15
  decoding_temperature: 0
  max_new_tokens: 4096
  docker_image: sweagent/swe-agent:latest

hardware:
  8b_gpu_memory_gb: 24            # single RTX 3090 / 4090
  2b_gpu_memory_gb: 12            # single RTX 3080 Ti or 4070
```

---

## 11. Hardware Requirements and Estimated Runtime

| Tier | GPU | Training VRAM | Eval VRAM | Estimated wall-clock |
|------|-----|---------------|-----------|----------------------|
| 8B | RTX 3090 / 4090 (24 GB) | 22 GB | 18 GB | ~20 h total (all 5 conditions) |
| 2B | RTX 3080 Ti / 4070 (12 GB) | 10 GB | 6 GB | ~6 h total (all 5 conditions) |

Both the agent model and reference model (used only in the reorder phase of C4)
fit simultaneously in bfloat16 for the 8B tier on a 24 GB card by loading the
reference model CPU-side during the training phase and moving it to GPU only for
the profiling pass.

For tighter budgets, enabling 4-bit quantisation on the universal prefix
(Section 7.5 note) reduces peak VRAM by ≈ 40 % with negligible impact on
resolve rate.

---

## 12. Expected Results and Interpretation

The hypotheses driving the study, in order from strongest to most speculative:

| Hypothesis | Rationale |
|------------|-----------|
| **C4 ≥ C3 ≥ C2** (resolve rate at same model size) | Selective freezing in C4 prevents forgetting general reasoning; C3 is less prone to catastrophic forgetting than C2 because the base is frozen. |
| **C1 close to C0** (resolve rate) but smaller model | Post-training pruning removes prose-specific neurons without adding coding knowledge; resolve rate should not improve much, but model size decreases. |
| **C4 achieves C2-level resolve rate at C1-level model size** | This is the main hypothesis: interleaved training acquires task knowledge while pruning unused capacity, hitting a better point on the resolve-rate vs. size curve. |

If C4 does not outperform C3, the most likely explanations are:

1. **Profiling budget too small** — 512 calibration prompts may not cleanly
   separate universal from specific neurons for coding tasks.  Increase to
   2,000+ samples.
2. **Too few interleaved cycles** — 4 cycles over 3,000 steps is aggressive; the
   universal/specific boundary may not have stabilised.  Try 8 cycles with 6,000
   total steps.
3. **Saturation not reached** — the specific segment of a 2B or 8B model may
   have sufficient capacity that the expansion criterion never triggers, reducing
   the advantage of the dynamic reordering.

If C4 outperforms C2 but not C3, it still validates the main benefit (better
forgetting protection vs. full SFT) while revealing that LoRA is a strong
competitor at this scale.

---

## 13. Ablation Extensions

The following optional extensions can be run after the main study to isolate
individual components of the interleaved loop:

| Ablation | What it tests |
|----------|--------------|
| **C4-noprune** | C4 without the defrag / tail-truncation step; measures the contribution of structural pruning to resolve rate. |
| **C4-nofreeze** | C4 without gradient hooks (all neurons trainable); measures the contribution of selective freezing. |
| **C4-1cycle** | C4 with a single reorder + training cycle; measures the benefit of multiple reorder iterations. |
| **C3+C1** | LoRA fine-tuning followed by post-training pruning; measures whether post-hoc pruning can recover the size advantage of C4. |

---

## 14. Reproducing the Study

```bash
# 1. Install dependencies
pip install ".[dev]" peft transformers datasets accelerate
pip install git+https://github.com/SWE-agent/SWE-agent.git

# 2. Set HuggingFace token (for gated models such as Llama)
export HF_TOKEN=<your_token>

# 3. Run the full study
python study/run_study.py --config study/config.yaml

# 4. Inspect results
cat results/summary.csv
```

Checkpoints and per-instance evaluation logs are written to the `checkpoints/`
and `results/` directories respectively.  Intermediate checkpoints are saved
every 500 gradient steps so a run can be resumed after interruption.

---

## 15. Relationship to Existing Documentation

| Document | Relationship |
|----------|-------------|
| [`README.md`](README.md) | Describes all three parts of the Ablirotate toolkit used in this study. |
| [`APPROACH.md`](APPROACH.md) | Theory and literature background for the pruning and differential abliteration mechanisms used in C1 and the reorder phase of C4. |
| [`INTERLEAVED_TRAINING.md`](INTERLEAVED_TRAINING.md) | Full algorithm specification for C4 including the selective-freezing gradient hooks, saturation criterion, and capacity expansion. |
| [`AUTONOMOUS_AGENT.md`](AUTONOMOUS_AGENT.md) | Design for an unattended long-running deployment that extends C4 to a production sandbox.  The orchestrator in this study is a minimal, batch-mode variant of that design. |
