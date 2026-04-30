# Autonomous Sandbox Agent — Agentic Specialisation with Periodic Retraining

This document describes a design for leaving a single language-model agent running
**unattended in a sandbox**, handling a specific class of tasks (e.g. Python + Git
coding), while a separate, unmodified copy of the original model continues to
handle all other prompts.  Periodic micro-training cycles gradually specialise the
agent model without touching the original.

---

## 1. Motivation

The [interleaved training loop](INTERLEAVED_TRAINING.md) assumes a human
operator who decides when to run each reorder/training cycle and who supplies
labelled data.  In an autonomous agent setting the operator is absent.  The
agent must:

* keep handling tasks without downtime between training cycles,
* identify its own *universal neurons* (shared with the original model) without
  any human labelling,
* decide when to retrain and for how long given a fixed hardware budget, and
* avoid catastrophic forgetting of general capabilities while acquiring
  task-specific ones.

---

## 2. System Architecture

```
┌─────────────────────────────────┐   ┌──────────────────────────────────┐
│  AGENT SANDBOX                  │   │  REFERENCE HOST (original model) │
│                                 │   │                                  │
│  Task queue (Python + Git)      │   │  Accepts all task types          │
│        │                        │   │  ActivationTracker always on     │
│        ▼                        │   │        │                         │
│  Specialised model              │   │        ▼                         │
│  + ActivationTracker            │   │  Category snapshots:             │
│  + NeuronForgetSchedule         │◄──┤   "code_python", "prose_en",     │
│                                 │   │   "maths", "cobol", …            │
│  Orchestrator (timer-driven)    │   │        │                         │
│   ├─ Observation window flush   │   │  Exports: universal_boundary.pt  │
│   ├─ Reorder phase              │◄──┘  (broadcast to agent sandbox)    │
│   ├─ Graduated pruning          │
│   └─ Micro-training session     │
└─────────────────────────────────┘
```

The reference model's `DifferentialAbliterator` exports a
`universal_boundary.pt` file — a `dict[layer_name → int]` giving the index of
the last universal neuron after permutation — and broadcasts it to the agent
sandbox at the start of each reorder phase.  Weight tensors are never shared.

---

## 3. Dual-Model Design — Why Keep the Original

The original model runs in inference-only mode alongside the agent.  It plays
three roles:

| Role | Mechanism |
|---|---|
| **Universal neuron identification** | Its activation profile on *non-coding* prompts defines which neurons are truly general-purpose and must stay frozen in the agent model. |
| **Quality referee** | Reference outputs for the same coding prompts are used to detect quality regression (embedding similarity or test-pass rate). |
| **Safety net** | If agent specialisation degrades coding quality beyond a configurable threshold, the reference model is promoted back as the primary until the next cycle. |

---

## 4. The Four-Phase Autonomous Loop

The orchestrator fires on a configurable clock (default every 6 hours or every
N agent turns, whichever comes first).

### Phase 1 — Observation (4–8 hours)

The agent handles tasks normally.
`Gemma4ActivationTracker` (or the generic `ActivationTracker` for smaller
models) accumulates statistics for the category `"code_python_git"`.  The
reference model simultaneously accumulates statistics for `"prose_en"`,
`"maths"`, `"foreign_lang"`, etc.  No weight changes happen; the only overhead
is a few megabytes of integer activation counters.

### Phase 2 — Reorder (minutes, gradient-free)

1. Import the latest `universal_boundary.pt` from the reference host.
2. Run `DifferentialAbliterator.prioritized_indices()` on the agent tracker
   stats, with `keep_categories=["code_python_git"]` and `drop_categories`
   derived from the reference export.
3. Apply `MatrixDefragmenter` — permute weight matrices to put universal
   neurons at positions `[:U]` and drop-specific neurons in the tail.  No
   truncation yet.
4. Bump `NeuronForgetSchedule` ages for neurons currently in `drop_specific`
   or `neutral`.

### Phase 3 — Graduated Pruning (seconds)

For each neuron in the forget schedule:

| Age (observation windows) | Treatment |
|---|---|
| 1–2 | `soft` prune (`high_temp_scale=0.5`), reversible |
| 3–5 | `cold` mode — zero-out weights, inject negative bias |
| ≥ 6 | `hard` prune; eligible for tail truncation at next reorder |

Any neuron that re-appears as `keep_specific` or `common` has its age reset
and all suppression removed.

### Phase 4 — Micro-Training (10–60 minutes, gradient-based)

1. Freeze rows `[:U]` in each layer via gradient hooks (see §Selective
   Freezing below).
2. Replay a rolling buffer of recent agent task inputs (next-token prediction
   or execution-outcome reward if an evaluator is available).
3. Run 100–500 gradient steps, depending on hardware budget (see §Hardware).
4. If `saturation > 0.85` and `val_loss_delta < τ`: grow FFN width (Option A
   from the interleaved training spec).
5. Remove gradient hooks; model is ready for the next observation window.

**Downtime:** the model is unavailable during micro-training.  A
*shadow-training* pattern can eliminate this: keep the active model frozen
while a second in-memory copy of the specific segment trains asynchronously,
then hot-swap the specific-segment weights on completion.

### Scheduling summary

```
Every 6 h (or N agent turns, whichever first):
  1. [~2 min]  Flush observation window → DifferentialAbliterator.record("code_python_git")
  2. [~2 min]  Receive updated universal_boundary.pt from reference host
  3. [~5 min]  Reorder phase (MatrixDefragmenter, no truncation)
  4. [~1 min]  Graduated pruning (age bump + soft/cold/hard per tier)
  5. [20–60 min] Micro-training (100–500 steps on task replay buffer)
  6. [~1 min]  Checkpoint model + forget schedule to disk
  7. [~1 min]  Reference evaluation (optional: 10 coding prompts vs. reference)
```

---

## 5. Hardware Requirements

### 5.1 Gemma 4 26B / 27B MoE

Gemma 4 27B has 62 layers, hidden size 5120, intermediate size 40960, and uses
Mixture-of-Experts routing.  Only the active experts' weights are used per
token, so the *active parameter* count is much smaller than the total.

| Phase | VRAM (bfloat16) | Notes |
|---|---|---|
| Inference (observation) | ~20–24 GB | 8× A100/H100 with tensor parallelism, or 2× with 4-bit quantisation |
| Reorder phase | +2–4 GB peak | Process one layer at a time and free immediately |
| Micro-training, full specific segment | +8–16 GB for optimizer state | Adam moments only for the `[U:]` tail |
| Micro-training, neuron micro-batch variant | +1–3 GB per layer | See §RAM minimisation — fits in a single 40 GB A100 |

**Practical recommendation:** two A100 80 GB GPUs or one H100 80 GB are
sufficient when the specific segment is ≤ 30% of neurons.  With 4-bit
quantisation of the universal prefix (QLoRA-style base + bfloat16 specific
tail) the entire loop fits on a single A100 40 GB.

### 5.2 8B Models (Llama-3-8B, Gemma-3-8B, Qwen2.5-8B)

| Phase | VRAM (bfloat16) | VRAM (4-bit) |
|---|---|---|
| Inference | ~16 GB | ~8 GB |
| Reorder phase | +1 GB peak | +1 GB peak |
| Micro-training, full specific segment | +4–6 GB | +1–2 GB |
| Agent + reference simultaneously | ~32 GB bfloat16 | ~16 GB 4-bit |

**Practical recommendation:** a single NVIDIA 3090 (24 GB) handles the full
loop in bfloat16.  With 4-bit base + bfloat16 specific tail, both agent and
reference model fit simultaneously on a 3090.

### 5.3 2B Models (Gemma-3-2B, Phi-3-mini, SmolLM2-2B)

| Phase | VRAM (bfloat16) | Notes |
|---|---|---|
| Inference | ~4 GB | |
| Micro-training, full specific segment | +1–2 GB | |
| Agent + reference simultaneously | ~8–10 GB | Fits on a single 12 GB GPU |

**Practical recommendation:** the entire pipeline — agent model, reference
model, and micro-training — runs on a single consumer GPU (e.g. RTX 3080 Ti
or 4070).  This tier is recommended for development and experimentation.

---

## 6. RAM-Minimisation Techniques

Listed from highest to lowest impact.  Each technique trades training speed
for VRAM; they can be combined.

### 6.1 4-bit Quantisation of the Universal Segment (largest impact)

Load the `[:U]` prefix of each layer in NF4/GPTQ and keep the `[U:]`
specific tail in bfloat16.  The forward pass uses the quantised prefix; only
the bfloat16 tail participates in backpropagation.  This roughly halves VRAM
for the dominant (universal) portion of every layer with minimal quality
loss.

### 6.2 Neuron Micro-Batch Training

Rather than computing gradients for all specific-segment neurons
simultaneously, cycle through blocks of `M` neurons.  At each step:

1. Unfreeze block `[i·M : (i+1)·M]`.
2. Compute gradients and optimizer step.
3. Re-freeze.
4. Advance to the next block.

Adam moments then cover only `M` neurons at a time instead of `N_specific`
neurons.  Speed cost: approximately `N_specific / M` times more steps to cover
all neurons once per epoch.

**Example (Gemma 4 27B, M = 512):**
Optimizer state drops from ~1.5 GB to ~3 MB.

### 6.3 Layer-Serial Training (maximum memory saving)

Unfreeze one transformer layer at a time.  Gradients and optimizer state for
all other layers are never allocated.

**Example (Gemma 4 27B, one MLP layer):**
One MLP layer ≈ 3 × 40960 × 5120 × 2 bytes ≈ 1.2 GB.
Adam state for that layer ≈ 2.4 GB → total well within a 4 GB budget per pass.
Full cycle loops through all 62 layers sequentially.

### 6.4 Gradient Checkpointing on the Universal Segment

During the backward pass, recompute universal-segment activations on-the-fly
rather than storing them.  Trades ~30% training throughput for ~40% activation
memory saving.  Useful when VRAM is slightly over-budget rather than
critically constrained.

### 6.5 Alternative Optimizers

Adam keeps two moment tensors per parameter.  Alternatives:

| Optimizer | Memory vs. Adam | Quality trade-off |
|---|---|---|
| **Adafactor** | ~10× lower (factored row + column approx.) | Mild — tune learning rate |
| **SGD + momentum** | 2× lower (one moment tensor) | May need more steps to converge |
| **SOAP / Lion** | ~1× (same or lower) | Competitive with Adam on LLM fine-tuning |

---

## 7. New Components Required

The following components are not yet in the Ablirotate codebase and would need
to be implemented to realise this design.

| Component | Description |
|---|---|
| `AgentSandboxOrchestrator` | Timer-driven loop controller; coordinates all four phases and handles the shadow-training swap. |
| `NeuronForgetSchedule` | Per-neuron age tracking; serialisable to disk between cycles. |
| `NeuronMicroBatchTrainer` | Gradient hooks + Adam micro-batch cycling logic (§6.2). |
| `SaturationMonitor` | Computes `saturation(l)` and `val_loss_delta` to trigger capacity expansion. |
| `ReplayBuffer` | Rolling buffer of recent agent task inputs for micro-training replay. |
| `UniversalBoundaryExporter` | Serialises `universal_boundary.pt` from the reference host and broadcasts it to the agent sandbox. |

### Relationship to Existing Components

| Existing component | Role in the autonomous loop |
|---|---|
| `Gemma4ActivationTracker` | Primary tracker on the agent model; hooks MLP + attention per coding task. |
| `ActivationTracker` (generic) | Used on the reference model for 8B/2B architectures. |
| `DifferentialAbliterator` | Classifies neurons as universal/specific; produces the permutation and `universal_boundary.pt`. |
| `MatrixDefragmenter` | Reorder phase; tail truncation applied only for neurons at age ≥ 6. |
| `ModelPruner` / `Gemma4MlpPruner` | Graduated pruning — soft/cold/hard per age tier. |

---

## 8. Selective Freezing

The same gradient-hook pattern described in `INTERLEAVED_TRAINING.md` applies
here, keyed on `universal_boundary.pt`:

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

---

## 9. Further Reading

- [`APPROACH.md`](APPROACH.md) — detailed theory, mechanism descriptions, and
  comparison with related methods.
- [`INTERLEAVED_TRAINING.md`](INTERLEAVED_TRAINING.md) — full algorithm
  specification for the interleaved reordering–freezing training loop,
  convergence analysis, and comparison with EWC, PackNet, and MoE approaches.
