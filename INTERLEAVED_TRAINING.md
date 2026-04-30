# Interleaved Reordering–Freezing Training

## 1. Motivation

Ablirotate currently operates entirely in the **post-training, gradient-free** regime: it
observes activation statistics, reorders neurons, and prunes dead tails — all without
touching the loss function.  The approach documented here extends this into a
**cyclic training loop** that alternates between two distinct phases:

1. **Reorder phase (post-training / gradient-free):** use current activation statistics to
   reclassify neurons and physically reorder each weight matrix, placing "universal" neurons
   at the front and "task-specific" ones at the back.
2. **Training phase (gradient-based fine-tuning):** freeze the universal portion of each
   layer and fine-tune only the specific portion, optionally with a criterion to grow new
   capacity (extra heads or an additional FFN width segment) when existing specific slots
   saturate.

The central hypothesis is that *universal* neurons encode general linguistic and reasoning
capabilities that should be preserved and stabilised across all updates, while *specific*
neurons encode task-particular features that can specialise, be replaced, or be expanded
without disturbing the universal core.

---

## 2. Neuron Classification Refresher

`DifferentialAbliterator` classifies every neuron in every layer into four groups at any
point during training (see `APPROACH.md §2.4` for the full definition):

| Group | Definition | Role in this scheme |
|---|---|---|
| **common** | Active in ≥ 1 keep category **and** ≥ 1 drop category | Universal — freeze during training |
| **keep_specific** | Active only in keep categories | Task-positive specific — trainable |
| **drop_specific** | Active only in drop categories | Task-negative specific — trainable (or pruned) |
| **neutral** | Inactive in all observed categories | Dormant — trainable |

For this scheme: **universal = `common`**; **specific = `keep_specific ∪ drop_specific ∪ neutral`**.

---

## 3. The Interleaved Loop

```
╔══════════════════════════════════════════════════════════════════╗
║  INITIALISE: pre-trained base model                              ║
╚══════════════════════════════════════════════════════════════════╝
         │
         ▼ (collect activation statistics on calibration set)
╔══════════════════════════════════════════════════════════════════╗
║  REORDER PHASE (gradient-free)                                   ║
║  1. Run DifferentialAbliterator on current calibration set.      ║
║  2. Compute priority order: common → keep_specific →             ║
║     neutral → drop_specific.                                     ║
║  3. Apply permutation to weight matrices (MatrixDefragmenter).   ║
║  4. Optionally truncate dead tail (hard-prune drop_specific      ║
║     neurons beyond the capacity budget).                         ║
║  5. Snapshot boundary index U (last universal neuron index).     ║
╚══════════════════════════════════════════════════════════════════╝
         │
         ▼
╔══════════════════════════════════════════════════════════════════╗
║  TRAINING PHASE (gradient-based fine-tuning)                     ║
║  1. Freeze weight rows/columns [:U] (universal segment).         ║
║  2. Keep weight rows/columns [U:] trainable (specific segment).  ║
║  3. If expansion criterion is met, append new capacity           ║
║     (extra FFN width or additional attention head).              ║
║  4. Train for N steps on target task data.                       ║
╚══════════════════════════════════════════════════════════════════╝
         │
         ▼ (update activation statistics on post-training samples)
         └──────────────────────────── back to REORDER PHASE ──────
```

The loop runs until convergence (task-performance plateau) or a compute budget is
exhausted.

---

## 4. Reorder Phase – Algorithm Detail

### 4.1 Defining the Universal Boundary

After each activation-collection pass, compute per-layer counts:

```
N_universal(l) = |common(l)| + |keep_specific(l)|
N_specific(l)  = |drop_specific(l)| + |neutral(l)|
```

The **boundary index** `U(l) = N_universal(l)` divides each weight matrix row/column-wise
into a frozen prefix and a trainable suffix.

### 4.2 Applying the Permutation

`MatrixDefragmenter.defragment()` already accepts a custom index tensor.
`DifferentialAbliterator.prioritized_indices()` produces the required ordering.
For SwiGLU layers (Qwen, Llama-3, Mistral, Gemma) the same permutation must be applied
**simultaneously** to rows of `gate_proj`, rows of `up_proj`, and columns of `down_proj`
— already handled by the coordinated permutation logic in the existing code.

### 4.3 Numerical Equivalence

As long as the permutation is applied to all matrices that read from or write to the same
set of neurons within a layer, the model's forward pass is numerically equivalent before
and after reordering.  The first reorder phase can therefore be applied to any pre-trained
model without any accuracy loss.

---

## 5. Training Phase – Algorithm Detail

### 5.1 Selective Freezing

Because PyTorch parameter slices are not first-class objects, the cleanest implementation
uses **gradient hooks** that zero the universal prefix before the optimizer step:

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

This preserves the parameter shape (required for the next reorder phase) while preventing
any update to the universal neurons.

### 5.2 Expansion Criterion – When to Grow

The specific segment `[U:]` has a fixed budget after each reorder phase.  When it becomes
saturated (all neurons are active across both keep and drop categories) the model has no
free capacity to acquire new task-specific features.  A practical trigger:

```
expand = True  if  saturation(l) > τ_sat  AND  val_loss_delta < τ_improve
```

where:

- `saturation(l) = (|keep_specific(l)| + |drop_specific(l)|) / N_specific(l)` — fraction
  of the specific budget that is already assigned to an observed category.
- `τ_sat` — saturation threshold (e.g. 0.85).
- `val_loss_delta` — improvement in validation loss over the last *K* steps; if it is
  below `τ_improve` the model is not learning, suggesting a capacity bottleneck rather
  than a data issue.

### 5.3 What Capacity to Add

Three options in order of architectural invasiveness:

**Option A – Grow FFN width (extra intermediate neurons)**
Append `Δ` new rows to `gate_proj` and `up_proj`, and `Δ` new columns to `down_proj`,
initialised to near-zero.  These neurons start as *neutral* and migrate to
`keep_specific` or `drop_specific` after the next reorder.  This is the least disruptive
change.

**Option B – Add a new attention head**
Append a new head to Q, K, V projections (zero-initialised).  Suitable only if the
saturation signal is specific to attention layers (which can be diagnosed by running the
tracker on attention modules separately).  For GQA models (e.g. Qwen2.5) a new query
head can be added without adding a new KV head, provided the attention implementation
supports a variable query-per-KV ratio.

**Option C – Insert a new transformer block**
A new block is initialised as a near-identity (zero residual contribution) and inserted
after the layer with the highest saturation.  This is the most aggressive option and is
analogous to the Net2Net / progressive-growing approach (see §6.5).

A conservative strategy: start with Option A; escalate to B or C only if saturation
persists across two consecutive loop iterations.

---

## 6. Comparison with Existing Approaches

### 6.1 Iterative Magnitude Pruning + Fine-Tuning (Han et al. 2015; Zhu & Gupta 2017)

The classic **prune → retrain → prune** cycle is the closest structural ancestor:

| Dimension | Han-style iterative pruning | Proposed interleaved loop |
|---|---|---|
| Pruning signal | Weight magnitude (static) | Empirical activation rate (dynamic) |
| Retraining scope | All remaining weights | Only specific segment |
| Neuron reordering | None | Explicit, enabling clean slicing |
| Expansion | No | Yes, via saturation criterion |
| Gradient-free phase | No | Yes (reorder phase) |

### 6.2 Lottery Ticket Hypothesis (Frankle & Carlin 2018; Frankle et al. 2020)

Lottery ticket methods identify a sparse sub-network (the "winning ticket") that can be
trained from scratch to match the full network's accuracy.  Instability-aware rewinding
(Frankle et al. 2020) showed that the relevant epoch to rewind to is early training —
implying that early-training weights already encode a useful prior, similar in spirit to
freezing the universal segment.

The proposed loop differs in that it **does not rewind** (past knowledge is retained in
the frozen universal segment) and identifies the sub-network to freeze via
activation-based semantics rather than weight magnitude at a reference epoch.

### 6.3 Modular Fine-Tuning: Adapter Layers and LoRA

**LoRA (Hu et al. 2022)** restricts gradient updates to low-rank adapter matrices
appended to frozen base weights.  **Pfeiffer adapters (2020)** insert small bottleneck
modules between transformer layers.

The proposed loop achieves a similar separation of concerns — universal neurons frozen,
specific neurons trainable — but **inside the existing weight matrices** rather than as
addenda.  This avoids inference overhead from extra modules.  LoRA and the proposed
approach are complementary: LoRA adapters could be attached to the specific segment only,
further reducing the parameter count that needs to be updated.

### 6.4 Task-Aware Freezing for Continual Learning: PackNet (Mallya & Lazebnik 2018)

PackNet freezes subsets of weights to preserve earlier-task knowledge and allocates fresh
capacity for new tasks.  The proposed loop is conceptually a **single-task variant** of
PackNet where the "old task" is the pre-trained general distribution (represented by
universal neurons) and the "new task" is the fine-tuning target (represented by specific
neurons).  Extending it to multi-task / continual settings is natural: after each new
task, the common neurons grow, the task-specific neurons for the new task are allocated
from the remaining budget, and the loop continues.

### 6.5 Progressive Network Growing: Net2Net and Progressive Growing

**Chen et al. (2015), "Net2Net"** described function-preserving transformations that
widen or deepen a network while preserving the initial function.  **Karras et al. (2018)**
applied progressive growing to GANs to add resolution stages incrementally.

The expansion criterion in §5.2–5.3 implements a **function-preserving width extension**
(Option A) or a **near-identity depth extension** (Option C), triggered adaptively via
the saturation metric rather than on a fixed schedule.

### 6.6 Elastic Weight Consolidation (Kirkpatrick et al. 2017)

EWC penalises changes to weights that are important for previously learned tasks, using
the Fisher information matrix as the importance measure.

The proposed loop replaces the continuous Fisher penalty with a **hard binary mask**:
universal neurons receive zero gradient, specific neurons are updated freely.  This is
more computationally efficient (no second-order statistics to maintain) and more
transparent, at the cost of inflexibility — a neuron cannot be *partially* universal.

### 6.7 Mixture of Experts (Fedus et al. 2022; Jiang et al. 2024)

Sparse MoE models route each token to a subset of expert FFN layers, achieving
conditional computation at the cost of a learned router.

The proposed interleaved loop produces a **statically partitioned** model: universal
neurons are always active; specific neurons activate only on the target distribution.
This can be seen as a soft MoE with two implicit experts (universal and specific) and no
routing overhead.  For multi-task extension, a lightweight task-ID router could
dynamically select which specific segment is active — bridging the gap to true MoE.

### 6.8 Representation Surgery and Weight Disentanglement

Work on representation surgery (Davari et al. 2022) and model merging via DARE / TIES
(Yu et al. 2023; Yadav et al. 2023) observes that fine-tuning entangles general and
task-specific representations in the same weight matrices.  The reorder phase is a form
of **representation disentanglement**: it physically separates neurons by their
generality *before* the gradient update, reducing the risk of fine-tuning overwriting
general knowledge.  The frozen universal segment plays the same role as the "base" model
in weight-merging methods.

### 6.9 Summary Table

| Method | Reorders weights | Freezes by function | Dynamic expansion | Gradient-free phase |
|---|---|---|---|---|
| Magnitude pruning + retrain | No | No | No | No |
| Lottery ticket | No | No | No | No |
| LoRA / Adapters | No | Partially (base frozen) | No | No |
| PackNet | No | Yes (task mask) | No | No |
| EWC | No | No (soft penalty) | No | No |
| Progressive NN | No | Yes (column freeze) | Yes (full column) | No |
| Net2Net | No | No | Yes (width) | No |
| MoE | No | No | No | No |
| **Proposed interleaved loop** | **Yes** | **Yes (activation-based)** | **Yes (saturation criterion)** | **Yes (reorder phase)** |

---

## 7. Convergence Properties and Expected Behaviour

**Why freezing the universal segment should help:**

1. **Stability of representations.** Universal neurons encode features activated across
   all observed task distributions.  Allowing gradients to update them risks destroying
   cross-task generality — the classical catastrophic-forgetting scenario.  Freezing them
   provides stronger, simpler protection than EWC's quadratic penalty.

2. **Gradient isolation.** With universal neurons frozen, the gradient signal flowing
   through the specific segment is not diluted by the much larger universal segment.  The
   effective learning rate for specific neurons is higher relative to their contribution
   to the loss.

3. **Faster convergence of specific neurons.** Fine-tuning experiments consistently show
   that updating a smaller, task-relevant fraction of parameters converges faster —
   consistent with the LoRA literature.

**Potential risks:**

- **Boundary misclassification.** If activation statistics were collected on an
  unrepresentative calibration set, some universal neurons may be incorrectly classified
  as specific and vice versa.  Mitigation: collect calibration data from a broad
  distribution and re-run the reorder phase frequently.

- **Boundary rigidity.** The boundary U is recomputed at discrete intervals.  Between
  reorder phases, neurons that migrate from specific to universal remain in the trainable
  segment.  Mitigation: run shorter training phases (more frequent reordering).

- **Expansion overhead.** Adding new neurons or blocks mid-training changes the parameter
  count and requires extending optimizer state tensors (e.g. Adam moments).  This is
  manageable but adds implementation complexity.

---

## 8. Relationship to Ablirotate's Current Implementation

| Existing capability | Role in the proposed loop |
|---|---|
| `ActivationTracker` | Collects statistics for both reorder and expansion decisions |
| `DifferentialAbliterator.classify_neurons` | Defines the universal / specific boundary |
| `DifferentialAbliterator.prioritized_indices` | Produces the permutation for the reorder phase |
| `MatrixDefragmenter` | Applies the permutation; tail truncation removes drop_specific neurons |
| `ModelPruner` | Optionally hard-prunes the drop_specific tail |

**Not yet implemented** — additions required to realise the full loop:

1. **Gradient masking hooks** — zero gradients in the universal segment (§5.1).
2. **Saturation metric** — compute the expansion criterion (§5.2).
3. **Width / depth expansion utilities** — zero-initialised row/column appension for
   `gate_proj`, `up_proj`, `down_proj`, and attention projections (§5.3, Options A/B/C).
4. **Loop orchestrator** — a training-loop class that alternates between the reorder
   phase (calling the existing gradient-free tools) and the training phase (standard
   PyTorch training with the gradient mask active).

---

## 9. Conclusion

The proposed interleaved reordering–freezing loop occupies a novel position in the design
space:

- **More principled than LoRA** in its identification of what to freeze, using empirical
  activation semantics rather than an arbitrary rank constraint.
- **More efficient than EWC** in protecting old knowledge — a hard boundary rather than a
  quadratic penalty.
- **More adaptive than PackNet** — the frozen/trainable boundary is recomputed at each
  cycle rather than fixed after the first pruning.
- **Less complex than MoE** while achieving similar task-conditional compute routing
  through static structural specialisation.
- **Integrates cleanly with Ablirotate's existing toolkit** — the reorder phase is fully
  supported today; the training phase requires targeted new utilities (gradient masking,
  saturation detection, expansion).

The most promising near-term extension is a **multi-task continual learning** setting
where the universal segment accumulates shared knowledge across successive tasks and the
specific segment grows (via Option A width expansion) to accommodate each new task's
private capacity — producing a model that learns without forgetting by construction
rather than by regularisation.

---

## References

- Han, S., Pool, J., Tran, J., & Dally, W. (2015). Learning both Weights and Connections for Efficient Neural Network. *NeurIPS*.
- Zhu, M., & Gupta, S. (2017). To Prune, or Not to Prune: Exploring the Efficacy of Pruning for Model Compression. *ICLR Workshop*.
- Frankle, J., & Carlin, M. (2018). The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. *ICLR 2019*.
- Frankle, J., Dziugaite, G.K., Roy, D.M., & Carlin, M. (2020). Linear Mode Connectivity and the Lottery Ticket Hypothesis. *ICML*.
- Hu, E., Shen, Y., Wallis, P., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*.
- Pfeiffer, J., Rücklé, A., Poth, C., et al. (2020). AdapterHub: A Framework for Adapting Transformers. *EMNLP*.
- Mallya, A., & Lazebnik, S. (2018). PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning. *CVPR*.
- Kirkpatrick, J., Pascanu, R., Rabinowitz, N., et al. (2017). Overcoming Catastrophic Forgetting in Neural Networks. *PNAS*.
- Chen, T., Goodfellow, I., & Shlens, J. (2015). Net2Net: Accelerating Learning via Knowledge Transfer. *ICLR 2016*.
- Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2018). Progressive Growing of GANs for Improved Quality, Stability, and Variation. *ICLR*.
- Rusu, A.A., Rabinowitz, N.C., Desjardins, G., et al. (2016). Progressive Neural Networks. *arXiv:1606.04671*.
- Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. *JMLR*.
- Jiang, A.Q., Sablayrolles, A., Roux, A., et al. (2024). Mixtral of Experts. *arXiv:2401.04088*.
- Davari, M., Asadi, N., Mudur, S., Ber­dine, R., & Bhatt, U. (2022). Probing the Robustness of Trained Metrics for Conversational Dialogue Systems. *ACL*.
- Yu, L., Yu, B., Yu, H., Huang, F., & Li, Y. (2023). DARE: Language Model Weight Pruning and Merging. *arXiv:2311.03099*.
- Yadav, P., Tam, D., Choshen, L., Raffel, C., & Bansal, M. (2023). TIES-Merging: Resolving Interference When Merging Models. *NeurIPS*.
- Arditi, A., Obeso, O., Conmy, A., et al. (2024). Refusal in Language Models Is Mediated by a Single Direction. *arXiv:2406.11717*.
- Molchanov, P., Mallya, A., Tyree, S., Frosio, I., & Kautz, J. (2019). Importance Estimation for Neural Network Pruning. *CVPR*.
- Frantar, E., & Alistarh, D. (2023). SparseGPT: Massive Language Models Can be Accurately Pruned in One Shot. *ICML*.
- Sun, M., Liu, Z., Bair, A., & Kolter, J.Z. (2023). A Simple and Effective Pruning Approach for Large Language Models. *ICLR 2024*.
- Ma, X., Fang, G., & Wang, X. (2023). LLM-Pruner: On the Structural Pruning of Large Language Models. *NeurIPS*.
- Ashkboos, S., Croci, M.L., do Nascimento, M.G., Hoefler, T., & Hensman, J. (2024). SliceGPT: Compress Large Language Models by Deleting Rows and Columns. *ICLR*.
