"""
Activation tracker: registers forward hooks on transformer MLP and attention
layers and records per-neuron / per-head activation statistics.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class ActivationStats:
    """Running statistics for a single layer's neurons / heads.

    Attributes
    ----------
    activation_counts : torch.Tensor
        How many times each neuron/head fired above *threshold* so far.
    total_samples : int
        Total number of forward-pass samples observed.
    threshold : float
        Absolute activation value above which a neuron is considered "active".
    """

    def __init__(self, n_units: int, threshold: float = 0.0) -> None:
        self.n_units = n_units
        self.threshold = threshold
        self.activation_counts: torch.Tensor = torch.zeros(n_units, dtype=torch.long)
        self.total_samples: int = 0

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, activations: torch.Tensor) -> None:
        """Record activations from one forward pass.

        Parameters
        ----------
        activations : torch.Tensor
            Shape ``(batch, seq_len, n_units)`` or ``(batch, n_units)``.
            Values are compared against *threshold*; a neuron counts as
            active for this sample if **any** token/position exceeds the
            threshold.
        """
        if activations.dim() == 3:
            # (batch, seq_len, units) → per-sample any-token activation
            fired = (activations.abs() > self.threshold).any(dim=1)  # (batch, units)
        elif activations.dim() == 2:
            fired = (activations.abs() > self.threshold)  # (batch, units)
        else:
            fired = (activations.abs() > self.threshold).unsqueeze(0)

        fired = fired.to(self.activation_counts.device)
        self.activation_counts += fired.sum(dim=0).cpu()
        self.total_samples += fired.shape[0]

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------

    @property
    def activation_rate(self) -> torch.Tensor:
        """Fraction of samples each neuron fired in (0, 1]."""
        if self.total_samples == 0:
            return torch.zeros(self.n_units)
        return self.activation_counts.float() / self.total_samples

    def least_active_indices(self, top_k: Optional[int] = None) -> torch.Tensor:
        """Return neuron indices sorted from *least* to *most* active."""
        order = self.activation_rate.argsort()
        if top_k is not None:
            order = order[:top_k]
        return order

    def most_active_indices(self, top_k: Optional[int] = None) -> torch.Tensor:
        """Return neuron indices sorted from *most* to *least* active."""
        order = self.activation_rate.argsort(descending=True)
        if top_k is not None:
            order = order[:top_k]
        return order

    def reset(self) -> None:
        self.activation_counts.zero_()
        self.total_samples = 0

    def __repr__(self) -> str:  # pragma: no cover
        rate = self.activation_rate
        return (
            f"ActivationStats(n_units={self.n_units}, "
            f"total_samples={self.total_samples}, "
            f"mean_rate={rate.mean():.3f}, "
            f"min_rate={rate.min():.3f}, "
            f"max_rate={rate.max():.3f})"
        )


# ---------------------------------------------------------------------------
# Hook helpers
# ---------------------------------------------------------------------------

def _make_mlp_hook(stats: ActivationStats):
    """Return a forward hook that records MLP intermediate activations."""

    def hook(module: nn.Module, input: tuple, output: torch.Tensor) -> None:  # noqa: A002
        with torch.no_grad():
            stats.update(output.detach())

    return hook


def _make_attn_hook(stats: ActivationStats):
    """Return a forward hook that records per-head attention norms.

    Works with modules that return ``(attn_output, attn_weights, ...)``
    tuples **or** plain tensors.  When attention weights are available
    their per-head L1 norm is used as the activation signal; otherwise
    the raw output tensor is used.
    """

    def hook(module: nn.Module, input: tuple, output) -> None:  # noqa: A002
        with torch.no_grad():
            if isinstance(output, tuple):
                # Most HuggingFace attention modules return
                # (context_layer, attention_probs, ...) or just (context_layer,)
                tensor = output[1] if (len(output) > 1 and output[1] is not None) else output[0]
            else:
                tensor = output

            if tensor is None:
                return

            tensor = tensor.detach()

            if tensor.dim() == 4:
                # attention_probs: (batch, heads, seq, seq) → per-head L1 norm
                head_signal = tensor.abs().mean(dim=(-2, -1))  # (batch, heads)
            elif tensor.dim() == 3:
                head_signal = tensor.abs().mean(dim=1)  # (batch, units)
            else:
                head_signal = tensor.abs()

            stats.update(head_signal)

    return hook


# ---------------------------------------------------------------------------
# Main tracker
# ---------------------------------------------------------------------------

class ActivationTracker:
    """Attach hooks to a transformer model and collect activation statistics.

    Parameters
    ----------
    model : nn.Module
        Any HuggingFace-compatible causal-LM / seq2seq model.
    threshold : float
        Absolute activation value above which a neuron is considered "active".
    track_mlp : bool
        Whether to attach hooks to MLP intermediate layers.
    track_attention : bool
        Whether to attach hooks to attention layers.
    mlp_submodule_name : str
        Name pattern used to detect MLP intermediate sub-modules
        (matched against the *last* component of the module name).
    attn_submodule_name : str
        Name pattern used to detect self-attention sub-modules.
    """

    def __init__(
        self,
        model: nn.Module,
        threshold: float = 0.0,
        track_mlp: bool = True,
        track_attention: bool = True,
        mlp_submodule_name: str = "act",
        attn_submodule_name: str = "attn",
    ) -> None:
        self.model = model
        self.threshold = threshold
        self.track_mlp = track_mlp
        self.track_attention = track_attention
        self.mlp_submodule_name = mlp_submodule_name
        self.attn_submodule_name = attn_submodule_name

        # layer_name → ActivationStats
        self.stats: Dict[str, ActivationStats] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

        self._register_hooks()

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _register_hooks(self) -> None:
        registered = 0
        for name, module in self.model.named_modules():
            last = name.rsplit(".", 1)[-1]

            if self.track_mlp and last == self.mlp_submodule_name:
                # Infer output size from module weight if possible
                n_units = self._infer_output_size(module)
                if n_units is None:
                    warnings.warn(
                        f"Cannot infer output size for MLP module '{name}'. "
                        "Hook will use dynamic sizing on first call."
                    )
                    n_units = 1  # placeholder; updated on first hook call
                stats = ActivationStats(n_units, self.threshold)
                self.stats[name] = stats

                # Wrap hook to handle dynamic size
                hook = module.register_forward_hook(
                    self._make_dynamic_hook(name, stats, "mlp")
                )
                self._hooks.append(hook)
                registered += 1

            elif self.track_attention and self.attn_submodule_name in last:
                n_units = self._infer_output_size(module)
                if n_units is None:
                    n_units = 1
                stats = ActivationStats(n_units, self.threshold)
                self.stats[name] = stats
                hook = module.register_forward_hook(
                    self._make_dynamic_hook(name, stats, "attn")
                )
                self._hooks.append(hook)
                registered += 1

        if registered == 0:
            warnings.warn(
                "No hooks were registered.  Check that 'mlp_submodule_name' "
                f"('{self.mlp_submodule_name}') and 'attn_submodule_name' "
                f"('{self.attn_submodule_name}') match module names in your model."
            )

    @staticmethod
    def _infer_output_size(module: nn.Module) -> Optional[int]:
        """Try to read output feature count from common layer types."""
        if hasattr(module, "out_features"):
            return module.out_features
        if hasattr(module, "weight") and module.weight is not None:
            return module.weight.shape[0]
        return None

    def _make_dynamic_hook(self, name: str, stats: ActivationStats, kind: str):
        """Return a hook that can resize *stats* on the first call."""

        def hook(module: nn.Module, inp: tuple, output) -> None:  # noqa: A002
            with torch.no_grad():
                tensor = output[0].detach() if isinstance(output, tuple) else output.detach()
                if tensor is None:
                    return

                # Determine n_units from actual output shape
                if tensor.dim() == 4:
                    # attention weights: (batch, heads, seq, seq)
                    actual_units = tensor.shape[1]
                    signal = tensor.abs().mean(dim=(-2, -1))
                elif tensor.dim() >= 2:
                    actual_units = tensor.shape[-1]
                    signal = tensor.abs().mean(dim=tuple(range(1, tensor.dim() - 1))) if tensor.dim() > 2 else tensor.abs()
                else:
                    actual_units = tensor.shape[0]
                    signal = tensor.abs().unsqueeze(0)

                # Lazily fix stats size on first call
                if stats.n_units != actual_units:
                    stats.n_units = actual_units
                    stats.activation_counts = torch.zeros(actual_units, dtype=torch.long)

                stats.update(signal if signal.dim() == 2 else signal.unsqueeze(0))

        return hook

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "ActivationTracker":
        return self

    def __exit__(self, *args) -> None:
        self.remove_hooks()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def remove_hooks(self) -> None:
        """Detach all registered hooks from the model."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def reset(self) -> None:
        """Reset all accumulated statistics."""
        for s in self.stats.values():
            s.reset()

    def get_stats(self) -> Dict[str, ActivationStats]:
        return dict(self.stats)

    def summary(self) -> Dict[str, float]:
        """Return mean activation rate per tracked layer."""
        return {
            name: float(s.activation_rate.mean())
            for name, s in self.stats.items()
        }

    def aggregate_activation_mask(
        self, rate_threshold: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """Return a boolean mask per layer: ``True`` = neuron considered active.

        Parameters
        ----------
        rate_threshold : float
            Neurons whose activation rate is at or above this value are kept.
        """
        return {
            name: s.activation_rate >= rate_threshold
            for name, s in self.stats.items()
        }
