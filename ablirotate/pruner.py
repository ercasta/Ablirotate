"""
Model pruner: apply activation-based pruning to transformer weight matrices.

Two modes are supported:

* **soft pruning** – scale down the output projection weights of low-activity
  neurons by a factor proportional to their inactivity, so they effectively
  contribute less without changing the matrix shape.
* **hard pruning** – physically zero-out (or remove) entire neuron rows/
  columns from weight matrices, and optionally reconfigure the zeroed
  neurons to activate only at high temperatures (via scaled bias injection).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from .tracker import ActivationStats


class ModelPruner:
    """Prune MLP neurons and attention heads in a transformer model.

    Parameters
    ----------
    model : nn.Module
        The transformer model to prune (modified **in-place**).
    stats : dict[str, ActivationStats]
        Per-layer activation statistics as produced by
        :class:`~ablirotate.tracker.ActivationTracker`.
    rate_threshold : float
        Neurons whose *activation_rate* is **below** this value are pruned.
    high_temp_scale : float
        When ``mode='soft'``, neurons below *rate_threshold* have their
        output weights scaled by this value (default 0.0 = full zero-out).
        When ``mode='cold'``, zeroed neurons get a negative bias equal to
        ``-high_temp_bias`` so they only fire at high logit temperatures.
    """

    def __init__(
        self,
        model: nn.Module,
        stats: Dict[str, ActivationStats],
        rate_threshold: float = 0.1,
        high_temp_scale: float = 0.0,
    ) -> None:
        self.model = model
        self.stats = stats
        self.rate_threshold = rate_threshold
        self.high_temp_scale = high_temp_scale

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prune(self, mode: str = "soft") -> Dict[str, int]:
        """Apply pruning to all tracked layers.

        Parameters
        ----------
        mode : str
            ``'soft'``  – scale weights; matrix shape is unchanged.
            ``'hard'``  – zero-out weights (neurons become dead).
            ``'cold'``  – zero-out weights AND inject a suppressing bias so
                          neurons re-activate only at elevated temperatures.

        Returns
        -------
        dict[str, int]
            Number of neurons pruned per layer name.
        """
        results: Dict[str, int] = {}
        for layer_name, stats in self.stats.items():
            module = self._get_module(layer_name)
            if module is None:
                continue

            mask = stats.activation_rate < self.rate_threshold  # True = prune
            n_pruned = int(mask.sum().item())
            if n_pruned == 0:
                results[layer_name] = 0
                continue

            if mode == "soft":
                self._soft_prune(module, mask)
            elif mode in ("hard", "cold"):
                self._hard_prune(module, mask, cold=(mode == "cold"))
            else:
                raise ValueError(f"Unknown pruning mode: {mode!r}")

            results[layer_name] = n_pruned

        return results

    def prune_to_mask(
        self,
        keep_mask: Dict[str, torch.Tensor],
        mode: str = "soft",
    ) -> Dict[str, int]:
        """Prune according to an externally provided boolean keep-mask.

        Parameters
        ----------
        keep_mask : dict[str, torch.BoolTensor]
            ``True`` = keep the neuron, ``False`` = prune it.
            Keys match the module paths in the model.
        mode : str
            Same as in :meth:`prune`.
        """
        results: Dict[str, int] = {}
        for layer_name, mask in keep_mask.items():
            prune_mask = ~mask
            module = self._get_module(layer_name)
            if module is None:
                continue

            n_pruned = int(prune_mask.sum().item())
            if n_pruned == 0:
                results[layer_name] = 0
                continue

            if mode == "soft":
                self._soft_prune(module, prune_mask)
            elif mode in ("hard", "cold"):
                self._hard_prune(module, prune_mask, cold=(mode == "cold"))
            else:
                raise ValueError(f"Unknown pruning mode: {mode!r}")

            results[layer_name] = n_pruned

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_module(self, path: str) -> Optional[nn.Module]:
        """Retrieve a sub-module by dotted path."""
        parts = path.split(".")
        module = self.model
        for p in parts:
            module = getattr(module, p, None)
            if module is None:
                return None
        return module

    def _soft_prune(self, module: nn.Module, prune_mask: torch.Tensor) -> None:
        """Scale down output-projection weights for pruned neurons."""
        with torch.no_grad():
            weight = self._get_weight(module)
            if weight is None:
                return

            indices = prune_mask.nonzero(as_tuple=False).squeeze(1)
            if weight.shape[0] == prune_mask.shape[0]:
                # Row-wise: each row is one neuron's output weights
                weight[indices] *= self.high_temp_scale
            elif weight.shape[1] == prune_mask.shape[0]:
                # Column-wise
                weight[:, indices] *= self.high_temp_scale

    def _hard_prune(
        self, module: nn.Module, prune_mask: torch.Tensor, cold: bool = False
    ) -> None:
        """Zero-out weights for pruned neurons (and optionally add cold bias)."""
        with torch.no_grad():
            weight = self._get_weight(module)
            if weight is None:
                return

            indices = prune_mask.nonzero(as_tuple=False).squeeze(1)
            if weight.shape[0] == prune_mask.shape[0]:
                weight[indices] = 0.0
            elif weight.shape[1] == prune_mask.shape[0]:
                weight[:, indices] = 0.0

            if cold:
                self._inject_cold_bias(module, indices)

    def _inject_cold_bias(
        self, module: nn.Module, indices: torch.Tensor, bias_value: float = -10.0
    ) -> None:
        """Add a large negative bias to zeroed neurons so they only fire at
        high temperatures (where logit scaling reduces the effective bias).
        """
        bias = getattr(module, "bias", None)
        if bias is None:
            # Add a new bias parameter
            n_units = self._get_weight(module).shape[0]
            new_bias = torch.zeros(n_units, device=self._get_weight(module).device)
            new_bias[indices] = bias_value
            module.bias = nn.Parameter(new_bias, requires_grad=False)
        else:
            with torch.no_grad():
                bias[indices] = bias_value

    @staticmethod
    def _get_weight(module: nn.Module) -> Optional[torch.Tensor]:
        """Return the primary weight tensor of a module."""
        if hasattr(module, "weight") and module.weight is not None:
            return module.weight.data
        return None
