"""
Gemma 4 specific toolkit.

Gemma 4's MLP uses a GeGLU (Gated GELU) architecture::

    intermediate = act_fn(gate_proj(x)) * up_proj(x)   # (..., intermediate_size)
    output       = down_proj(intermediate)

This gated structure is identical in shape to Qwen2's SwiGLU MLP, so pruning must
coordinate changes across all three matrices simultaneously:

* ``gate_proj`` – rows corresponding to dead intermediate neurons are zeroed.
* ``up_proj``   – rows corresponding to dead intermediate neurons are zeroed.
* ``down_proj`` – *columns* corresponding to dead intermediate neurons are zeroed.

Tracking
--------
A ``register_forward_pre_hook`` on each ``down_proj`` captures the intermediate
activation tensor (the gate×up product) just before it is projected down.  This
is the cheapest place to observe neuron-level activity without modifying the
model's forward pass.

Attention
---------
Gemma 4 uses alternating local (sliding window, span=1024) and global
self-attention layers.  Every 6th layer (indices 5, 11, 17, …) is a global
attention layer; the remaining layers use local attention.  Both share the same
``o_proj`` naming convention, so the tracker captures the pre-``o_proj`` tensor
for all layers uniformly via a ``register_forward_pre_hook`` on ``o_proj``.

Gemma 4 27B key dimensions
---------------------------
.. list-table::
   :header-rows: 1

   * - Property
     - Value
   * - Hidden layers
     - 62
   * - Hidden size
     - 5120
   * - Intermediate size
     - 40960
   * - Attention heads (query)
     - 32
   * - Key-value heads (GQA)
     - 16
   * - Head dim
     - 256
   * - MLP activation
     - GeGLU (GELU approximate='tanh')
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .tracker import ActivationStats
from .differential import DifferentialAbliterator

# ---------------------------------------------------------------------------
# Architecture configuration
# ---------------------------------------------------------------------------

#: Default configuration for Gemma 4 27B.
GEMMA4_27B_CONFIG: Dict = {
    # Module name patterns (matched against the *last* path component)
    "down_proj": "down_proj",    # MLP down-projection
    "gate_proj": "gate_proj",    # MLP gate-projection
    "up_proj": "up_proj",        # MLP up-projection
    "o_proj": "o_proj",          # Attention output projection
    # Parent-path substring patterns
    "mlp_parent": "mlp",         # Parent module of down/gate/up_proj
    "attn_parent": "self_attn",  # Parent module of o_proj
    # Gemma 4 27B attention geometry
    "num_attention_heads": 32,
}


# ---------------------------------------------------------------------------
# Activation tracker
# ---------------------------------------------------------------------------

class Gemma4ActivationTracker:
    """Track intermediate MLP and attention-head activations in a Gemma 4 model.

    Unlike the generic :class:`~ablirotate.tracker.ActivationTracker`, which
    hooks a sub-module named ``"act"``, this tracker attaches
    ``register_forward_pre_hook`` callbacks to ``down_proj`` (MLP) and
    ``o_proj`` (attention) to capture the intermediate activation tensors that
    are most informative for GeGLU-based models.

    Parameters
    ----------
    model : nn.Module
        A Gemma 4 causal language model (or any model with the same
        ``gate_proj`` / ``up_proj`` / ``down_proj`` / ``o_proj`` structure).
    config : dict, optional
        Architecture configuration.  Defaults to
        :data:`GEMMA4_27B_CONFIG`.
    threshold : float
        Absolute activation value above which a neuron is considered "active".
    track_mlp : bool
        Whether to hook MLP intermediate activations.
    track_attention : bool
        Whether to hook attention head outputs.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[Dict] = None,
        threshold: float = 0.0,
        track_mlp: bool = True,
        track_attention: bool = True,
    ) -> None:
        self.model = model
        self.config = config if config is not None else GEMMA4_27B_CONFIG
        self.threshold = threshold
        self.track_mlp = track_mlp
        self.track_attention = track_attention

        # layer_path → ActivationStats (keyed by parent module path)
        self.mlp_stats: Dict[str, ActivationStats] = {}
        self.attn_stats: Dict[str, ActivationStats] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

        self._register_hooks()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stats(self) -> Dict[str, ActivationStats]:
        """Combined MLP and attention stats (read-only merged view)."""
        return {**self.mlp_stats, **self.attn_stats}

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _register_hooks(self) -> None:
        n_registered = 0
        for name, module in self.model.named_modules():
            parts = name.rsplit(".", 1)
            last = parts[-1] if len(parts) > 1 else name
            parent = parts[0] if len(parts) > 1 else ""

            # MLP intermediate: pre-hook on down_proj
            if (
                self.track_mlp
                and last == self.config["down_proj"]
                and self.config["mlp_parent"] in parent
            ):
                n_units = (
                    module.in_features
                    if hasattr(module, "in_features")
                    else module.weight.shape[1]
                )
                stats = ActivationStats(n_units, self.threshold)
                self.mlp_stats[parent] = stats
                hook = module.register_forward_pre_hook(self._make_mlp_hook(stats))
                self._hooks.append(hook)
                n_registered += 1

            # Attention per-head: pre-hook on o_proj
            elif (
                self.track_attention
                and last == self.config["o_proj"]
                and self.config["attn_parent"] in parent
            ):
                n_heads = self._get_num_heads()
                stats = ActivationStats(n_heads, self.threshold)
                self.attn_stats[parent] = stats
                hook = module.register_forward_pre_hook(
                    self._make_attn_hook(stats, n_heads)
                )
                self._hooks.append(hook)
                n_registered += 1

        if n_registered == 0:
            warnings.warn(
                "No hooks were registered. Check that the config patterns "
                f"(down_proj='{self.config['down_proj']}', "
                f"mlp_parent='{self.config['mlp_parent']}', "
                f"o_proj='{self.config['o_proj']}', "
                f"attn_parent='{self.config['attn_parent']}') "
                "match module names in your model."
            )

    def _get_num_heads(self) -> int:
        """Read head count from model.config when available, else fall back to config dict."""
        if hasattr(self.model, "config") and hasattr(
            self.model.config, "num_attention_heads"
        ):
            return self.model.config.num_attention_heads
        return self.config["num_attention_heads"]

    @staticmethod
    def _make_mlp_hook(stats: ActivationStats):
        """Pre-hook: record intermediate MLP activation (input to down_proj)."""

        def hook(module: nn.Module, args: tuple) -> None:
            with torch.no_grad():
                inp = args[0].detach()
                # Shape: (batch, seq, intermediate_size) or (batch*seq, intermediate_size)
                if inp.dim() == 3:
                    stats.update(inp)
                elif inp.dim() == 2:
                    stats.update(inp)
                else:
                    stats.update(inp.unsqueeze(0))

        return hook

    @staticmethod
    def _make_attn_hook(stats: ActivationStats, n_heads: int):
        """Pre-hook: record per-head activation signal (input to o_proj)."""

        def hook(module: nn.Module, args: tuple) -> None:
            with torch.no_grad():
                inp = args[0].detach()
                # Shape: (batch, seq, n_heads * head_dim)
                if inp.dim() == 3:
                    batch, seq, total = inp.shape
                    if total % n_heads != 0:
                        # Unexpected shape – fall back to flat signal
                        signal = inp.abs().mean(dim=(1, 2)).unsqueeze(-1)
                        signal = signal.expand(-1, stats.n_units)
                    else:
                        head_dim = total // n_heads
                        head_acts = inp.view(batch, seq, n_heads, head_dim)
                        # Per-head L2 norm averaged over sequence positions
                        signal = head_acts.norm(dim=-1).mean(dim=1)  # (batch, n_heads)
                elif inp.dim() == 2:
                    batch, total = inp.shape
                    if total % n_heads != 0:
                        signal = inp.abs().mean(dim=-1).unsqueeze(-1).expand(-1, stats.n_units)
                    else:
                        head_dim = total // n_heads
                        head_acts = inp.view(batch, n_heads, head_dim)
                        signal = head_acts.norm(dim=-1)  # (batch, n_heads)
                else:
                    return

                if signal.shape[-1] == stats.n_units:
                    stats.update(signal)

        return hook

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "Gemma4ActivationTracker":
        return self

    def __exit__(self, *args) -> None:
        self.remove_hooks()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def remove_hooks(self) -> None:
        """Detach all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def reset(self) -> None:
        """Reset all accumulated statistics."""
        for s in self.mlp_stats.values():
            s.reset()
        for s in self.attn_stats.values():
            s.reset()

    def summary(self) -> Dict[str, float]:
        """Return mean activation rate per tracked layer (MLP + attention)."""
        return {
            name: float(s.activation_rate.mean())
            for name, s in self.stats.items()
        }

    def aggregate_activation_mask(
        self, rate_threshold: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """Boolean keep-mask for all tracked layers (MLP + attention).

        Parameters
        ----------
        rate_threshold : float
            Neurons at or above this activation rate are marked ``True``.
        """
        return {
            name: s.activation_rate >= rate_threshold
            for name, s in self.stats.items()
        }

    def aggregate_mlp_mask(
        self, rate_threshold: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """Boolean keep-mask for MLP intermediate neurons only."""
        return {
            name: s.activation_rate >= rate_threshold
            for name, s in self.mlp_stats.items()
        }

    def aggregate_attn_mask(
        self, rate_threshold: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """Boolean keep-mask for attention heads only."""
        return {
            name: s.activation_rate >= rate_threshold
            for name, s in self.attn_stats.items()
        }


# ---------------------------------------------------------------------------
# GeGLU-aware pruner
# ---------------------------------------------------------------------------

class Gemma4MlpPruner:
    """Prune GeGLU MLP neurons across ``gate_proj``, ``up_proj``, and ``down_proj``.

    Unlike :class:`~ablirotate.pruner.ModelPruner`, which modifies a single
    module's weight matrix, this pruner applies **coordinated** changes to all
    three projection matrices of each Gemma 4 MLP block.

    Parameters
    ----------
    model : nn.Module
        The Gemma 4 model to prune (modified **in-place**).
    mlp_stats : dict[str, ActivationStats]
        Per-layer activation statistics keyed by the **parent MLP module path**
        (e.g. ``"model.layers.0.mlp"``), as produced by
        :class:`Gemma4ActivationTracker`.
    config : dict, optional
        Architecture configuration.  Defaults to
        :data:`GEMMA4_27B_CONFIG`.
    rate_threshold : float
        Neurons whose activation rate is **below** this value are pruned.
    high_temp_scale : float
        Scale factor applied to inactive neuron weights in ``mode='soft'``
        (default 0.0 = full zero-out).
    """

    def __init__(
        self,
        model: nn.Module,
        mlp_stats: Dict[str, ActivationStats],
        config: Optional[Dict] = None,
        rate_threshold: float = 0.1,
        high_temp_scale: float = 0.0,
    ) -> None:
        self.model = model
        self.mlp_stats = mlp_stats
        self.config = config if config is not None else GEMMA4_27B_CONFIG
        self.rate_threshold = rate_threshold
        self.high_temp_scale = high_temp_scale

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prune(self, mode: str = "soft") -> Dict[str, int]:
        """Prune all tracked MLP layers.

        Parameters
        ----------
        mode : str
            ``'soft'``  – scale weights; shape unchanged.
            ``'hard'``  – zero-out weights permanently.
            ``'cold'``  – zero-out weights and inject a suppressing bias on
                          ``gate_proj`` so neurons only re-activate at elevated
                          sampling temperatures.

        Returns
        -------
        dict[str, int]
            Number of intermediate neurons pruned per MLP path.
        """
        results: Dict[str, int] = {}
        for mlp_path, stats in self.mlp_stats.items():
            prune_mask = stats.activation_rate < self.rate_threshold
            n_pruned = int(prune_mask.sum().item())
            if n_pruned == 0:
                results[mlp_path] = 0
                continue
            self._prune_geglu(mlp_path, prune_mask, mode)
            results[mlp_path] = n_pruned
        return results

    def prune_to_mask(
        self,
        keep_mask: Dict[str, torch.Tensor],
        mode: str = "soft",
    ) -> Dict[str, int]:
        """Prune using an externally provided boolean keep-mask.

        Parameters
        ----------
        keep_mask : dict[str, torch.BoolTensor]
            ``True`` = keep; ``False`` = prune.  Keys are MLP module paths.
        mode : str
            Same as in :meth:`prune`.
        """
        results: Dict[str, int] = {}
        for mlp_path, mask in keep_mask.items():
            prune_mask = ~mask
            n_pruned = int(prune_mask.sum().item())
            if n_pruned == 0:
                results[mlp_path] = 0
                continue
            self._prune_geglu(mlp_path, prune_mask, mode)
            results[mlp_path] = n_pruned
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune_geglu(
        self, mlp_path: str, prune_mask: torch.Tensor, mode: str
    ) -> None:
        """Apply pruning to gate_proj, up_proj, and down_proj coordinately."""
        gate = self._get_module(f"{mlp_path}.{self.config['gate_proj']}")
        up = self._get_module(f"{mlp_path}.{self.config['up_proj']}")
        down = self._get_module(f"{mlp_path}.{self.config['down_proj']}")

        if gate is None or up is None or down is None:
            warnings.warn(
                f"Could not resolve gate/up/down_proj for '{mlp_path}'. "
                "Skipping this layer."
            )
            return

        indices = prune_mask.nonzero(as_tuple=False).squeeze(1)

        with torch.no_grad():
            if mode == "soft":
                scale = self.high_temp_scale
                gate.weight.data[indices] *= scale
                up.weight.data[indices] *= scale
                down.weight.data[:, indices] *= scale
                if gate.bias is not None:
                    gate.bias.data[indices] *= scale
                if up.bias is not None:
                    up.bias.data[indices] *= scale

            elif mode in ("hard", "cold"):
                gate.weight.data[indices] = 0.0
                up.weight.data[indices] = 0.0
                down.weight.data[:, indices] = 0.0
                if gate.bias is not None:
                    gate.bias.data[indices] = 0.0
                if up.bias is not None:
                    up.bias.data[indices] = 0.0
                if mode == "cold":
                    self._inject_cold_bias(gate, indices)

            else:
                raise ValueError(f"Unknown pruning mode: {mode!r}")

    def _inject_cold_bias(
        self,
        module: nn.Module,
        indices: torch.Tensor,
        bias_value: float = -10.0,
    ) -> None:
        """Add a large negative bias to zeroed neurons (gate_proj only)."""
        bias = getattr(module, "bias", None)
        if bias is None:
            n_units = module.weight.shape[0]
            new_bias = torch.zeros(n_units, device=module.weight.device)
            new_bias[indices] = bias_value
            module.bias = nn.Parameter(new_bias, requires_grad=False)
        else:
            with torch.no_grad():
                bias[indices] = bias_value

    def _get_module(self, path: str) -> Optional[nn.Module]:
        module = self.model
        for part in path.split("."):
            module = getattr(module, part, None)
            if module is None:
                return None
        return module


# ---------------------------------------------------------------------------
# GeGLU-aware defragmenter
# ---------------------------------------------------------------------------

class Gemma4Defragmenter:
    """Reorder and truncate GeGLU MLP weight matrices by neuron activity order.

    The same permutation is applied to:

    * rows of ``gate_proj`` (output neurons → intermediate)
    * rows of ``up_proj`` (output neurons → intermediate)
    * columns of ``down_proj`` (input neurons ← intermediate)

    This keeps the model numerically equivalent up to the truncation boundary
    while moving the most-active neurons to the front, enabling the dead tail
    to be cleanly removed.

    Parameters
    ----------
    model : nn.Module
        The Gemma 4 model to defragment (modified **in-place**).
    mlp_stats : dict[str, ActivationStats]
        Per-MLP activation statistics (keyed by MLP module path) from
        :class:`Gemma4ActivationTracker`.
    config : dict, optional
        Architecture configuration.  Defaults to
        :data:`GEMMA4_27B_CONFIG`.
    keep_fraction : float
        Fraction of intermediate neurons to retain (0 < keep_fraction ≤ 1).
    """

    def __init__(
        self,
        model: nn.Module,
        mlp_stats: Dict[str, ActivationStats],
        config: Optional[Dict] = None,
        keep_fraction: float = 1.0,
    ) -> None:
        if not 0 < keep_fraction <= 1.0:
            raise ValueError("keep_fraction must be in (0, 1]")
        self.model = model
        self.mlp_stats = mlp_stats
        self.config = config if config is not None else GEMMA4_27B_CONFIG
        self.keep_fraction = keep_fraction

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def defragment(self) -> Dict[str, Tuple[int, int]]:
        """Sort neurons by activity and optionally truncate the dead tail.

        Returns
        -------
        dict[str, tuple[int, int]]
            Mapping from MLP path → ``(original_size, new_size)``.
        """
        result: Dict[str, Tuple[int, int]] = {}
        for mlp_path, stats in self.mlp_stats.items():
            order = stats.most_active_indices()
            orig = len(order)
            keep = max(1, int(orig * self.keep_fraction))
            order = order[:keep]
            self._permute_geglu(mlp_path, order)
            result[mlp_path] = (orig, keep)
        return result

    def compute_permutation(self, mlp_path: str) -> Optional[torch.Tensor]:
        """Return the sorted neuron index order for a given MLP path."""
        stats = self.mlp_stats.get(mlp_path)
        if stats is None:
            return None
        order = stats.most_active_indices()
        keep = max(1, int(len(order) * self.keep_fraction))
        return order[:keep]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _permute_geglu(self, mlp_path: str, order: torch.Tensor) -> None:
        """Apply the neuron permutation/truncation to all three projection matrices."""
        gate = self._get_module(f"{mlp_path}.{self.config['gate_proj']}")
        up = self._get_module(f"{mlp_path}.{self.config['up_proj']}")
        down = self._get_module(f"{mlp_path}.{self.config['down_proj']}")

        if gate is None or up is None or down is None:
            warnings.warn(
                f"Could not resolve gate/up/down_proj for '{mlp_path}'. "
                "Skipping this layer."
            )
            return

        with torch.no_grad():
            # gate_proj: permute/truncate output rows
            gate.weight = nn.Parameter(
                gate.weight.data[order], requires_grad=gate.weight.requires_grad
            )
            if gate.bias is not None:
                gate.bias = nn.Parameter(
                    gate.bias.data[order], requires_grad=gate.bias.requires_grad
                )
            if hasattr(gate, "out_features"):
                gate.out_features = len(order)

            # up_proj: permute/truncate output rows
            up.weight = nn.Parameter(
                up.weight.data[order], requires_grad=up.weight.requires_grad
            )
            if up.bias is not None:
                up.bias = nn.Parameter(
                    up.bias.data[order], requires_grad=up.bias.requires_grad
                )
            if hasattr(up, "out_features"):
                up.out_features = len(order)

            # down_proj: permute/truncate input columns
            down.weight = nn.Parameter(
                down.weight.data[:, order], requires_grad=down.weight.requires_grad
            )
            if hasattr(down, "in_features"):
                down.in_features = len(order)

    def _get_module(self, path: str) -> Optional[nn.Module]:
        module = self.model
        for part in path.split("."):
            module = getattr(module, part, None)
            if module is None:
                return None
        return module


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

class Gemma4Pipeline:
    """Convenience end-to-end pipeline for Gemma 4 model specialisation.

    Parameters
    ----------
    model : nn.Module
        A Gemma 4 causal language model.
    tokenizer
        A HuggingFace tokenizer for the model.
    config : dict, optional
        Architecture configuration.  Defaults to
        :data:`GEMMA4_27B_CONFIG`.
    threshold : float
        Activation threshold passed to the tracker.

    Examples
    --------
    ::

        with Gemma4Pipeline(model, tokenizer) as pipeline:
            pipeline.run_prompts(code_samples)
            pipeline.prune(rate_threshold=0.10, mode="hard")
            pipeline.defragment(keep_fraction=0.80)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: Optional[Dict] = None,
        threshold: float = 0.0,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config if config is not None else GEMMA4_27B_CONFIG
        self.tracker = Gemma4ActivationTracker(
            model, config=self.config, threshold=threshold
        )

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------

    def run_prompts(self, prompts: Sequence[str]) -> None:
        """Run inference on *prompts* to accumulate activation statistics.

        Parameters
        ----------
        prompts : sequence of str
            Representative prompts for the target task.
        """
        device = next(self.model.parameters()).device
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                self.model(**inputs)

    # ------------------------------------------------------------------
    # Pruning and defragmentation
    # ------------------------------------------------------------------

    def prune(
        self,
        rate_threshold: float = 0.1,
        mode: str = "soft",
        high_temp_scale: float = 0.0,
    ) -> Dict[str, int]:
        """Prune inactive MLP neurons from all layers.

        Parameters
        ----------
        rate_threshold : float
            Neurons below this activation rate are pruned.
        mode : str
            ``'soft'``, ``'hard'``, or ``'cold'``.
        high_temp_scale : float
            Weight scale for soft pruning (default 0.0 = zero-out).

        Returns
        -------
        dict[str, int]
            Neurons pruned per layer.
        """
        pruner = Gemma4MlpPruner(
            self.model,
            self.tracker.mlp_stats,
            config=self.config,
            rate_threshold=rate_threshold,
            high_temp_scale=high_temp_scale,
        )
        return pruner.prune(mode=mode)

    def defragment(self, keep_fraction: float = 0.8) -> Dict[str, Tuple[int, int]]:
        """Defragment MLP matrices by sorting neurons by descending activity.

        Parameters
        ----------
        keep_fraction : float
            Fraction of intermediate neurons to retain.

        Returns
        -------
        dict[str, tuple[int, int]]
            ``(original_size, new_size)`` per MLP layer.
        """
        defrag = Gemma4Defragmenter(
            self.model,
            self.tracker.mlp_stats,
            config=self.config,
            keep_fraction=keep_fraction,
        )
        return defrag.defragment()

    def build_differential_abliterator(self) -> "DifferentialAbliterator":
        """Return a :class:`~ablirotate.differential.DifferentialAbliterator`
        wrapping the pipeline's tracker.

        Use this for category-based differential specialisation::

            abl = pipeline.build_differential_abliterator()
            for cat, prompts in categories.items():
                pipeline.run_prompts(prompts)
                abl.record(cat)
            keep_mask = abl.compute_keep_mask(
                keep_categories=["code"],
                drop_categories=["prose"],
            )
            pipeline.prune_to_mask(keep_mask)
        """
        return DifferentialAbliterator(self.tracker)

    def prune_to_mask(
        self,
        keep_mask: Dict[str, torch.Tensor],
        mode: str = "soft",
    ) -> Dict[str, int]:
        """Prune using an externally computed keep-mask.

        Parameters
        ----------
        keep_mask : dict[str, BoolTensor]
            ``True`` = keep; ``False`` = prune.  Keys are MLP module paths.
        mode : str
            Same as in :meth:`prune`.
        """
        pruner = Gemma4MlpPruner(
            self.model,
            self.tracker.mlp_stats,
            config=self.config,
        )
        return pruner.prune_to_mask(keep_mask, mode=mode)

    def reset(self) -> None:
        """Reset all accumulated activation statistics."""
        self.tracker.reset()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "Gemma4Pipeline":
        return self

    def __exit__(self, *args) -> None:
        self.tracker.remove_hooks()
