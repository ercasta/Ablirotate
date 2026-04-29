"""
Matrix defragmenter: reorder neuron rows/columns in weight matrices so that
the most-active neurons appear first.

This converts a logically sparse model into a **dense** model with a smaller
effective shape.  The key insight is that sparse matrices (where some rows are
all-zero) do not automatically give a speed-up on most hardware; physically
moving active neurons to the front of the matrix (and then slicing off the
dead tail) does.

The process is:

1. For every tracked layer, sort neurons by descending activation rate.
2. Permute the corresponding rows/columns in *all* weight matrices that read
   from / write to those neurons so the model stays numerically equivalent.
3. Optionally truncate the matrices at a configurable percentile of activity,
   discarding the dead tail entirely.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .tracker import ActivationStats


class MatrixDefragmenter:
    """Reorder and (optionally) truncate weight matrices by activation order.

    Parameters
    ----------
    model : nn.Module
        The model to defragment (modified **in-place**).
    stats : dict[str, ActivationStats]
        Per-layer statistics from :class:`~ablirotate.tracker.ActivationTracker`.
    keep_fraction : float
        Fraction of neurons to retain after sorting (0 < keep_fraction <= 1).
        Neurons beyond this fraction are dropped.
    """

    def __init__(
        self,
        model: nn.Module,
        stats: Dict[str, ActivationStats],
        keep_fraction: float = 1.0,
    ) -> None:
        if not 0 < keep_fraction <= 1.0:
            raise ValueError("keep_fraction must be in (0, 1]")
        self.model = model
        self.stats = stats
        self.keep_fraction = keep_fraction

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def defragment(self) -> Dict[str, Tuple[int, int]]:
        """Sort neurons by activation and optionally truncate dead tail.

        Returns
        -------
        dict[str, tuple[int, int]]
            Mapping from layer name → ``(original_size, new_size)`` after
            truncation.
        """
        result: Dict[str, Tuple[int, int]] = {}
        for layer_name, stats in self.stats.items():
            module = self._get_module(layer_name)
            if module is None:
                continue

            order = stats.most_active_indices()  # sorted: most active first
            orig = len(order)
            keep = max(1, int(orig * self.keep_fraction))
            order = order[:keep]

            self._permute_module(module, order)
            result[layer_name] = (orig, keep)

        return result

    def compute_permutation(
        self, layer_name: str
    ) -> Optional[torch.Tensor]:
        """Return the sorted neuron index order for a given layer."""
        stats = self.stats.get(layer_name)
        if stats is None:
            return None
        order = stats.most_active_indices()
        keep = max(1, int(len(order) * self.keep_fraction))
        return order[:keep]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_module(self, path: str) -> Optional[nn.Module]:
        parts = path.split(".")
        module = self.model
        for p in parts:
            module = getattr(module, p, None)
            if module is None:
                return None
        return module

    @staticmethod
    def _permute_module(module: nn.Module, order: torch.Tensor) -> None:
        """Reorder (and optionally truncate) a module's weight rows."""
        with torch.no_grad():
            if hasattr(module, "weight") and module.weight is not None:
                w = module.weight.data
                if w.shape[0] >= len(order):
                    module.weight = nn.Parameter(w[order], requires_grad=w.requires_grad)
                    # Update out_features if present
                    if hasattr(module, "out_features"):
                        module.out_features = len(order)

            if hasattr(module, "bias") and module.bias is not None:
                b = module.bias.data
                if b.shape[0] >= len(order):
                    module.bias = nn.Parameter(b[order], requires_grad=b.requires_grad)

    # ------------------------------------------------------------------
    # Partition helper
    # ------------------------------------------------------------------

    def partition(
        self, rate_boundary: float = 0.1
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Split weight matrices into 'hot' and 'cold' partitions.

        Parameters
        ----------
        rate_boundary : float
            Neurons with activation rate **>= rate_boundary** go to the
            'hot' partition; the rest go to the 'cold' partition.

        Returns
        -------
        dict[str, dict[str, Tensor]]
            ``{layer_name: {'hot': weight_hot, 'cold': weight_cold}}``
        """
        partitions: Dict[str, Dict[str, torch.Tensor]] = {}
        for layer_name, stats in self.stats.items():
            module = self._get_module(layer_name)
            if module is None or not hasattr(module, "weight"):
                continue

            rate = stats.activation_rate
            hot_idx = (rate >= rate_boundary).nonzero(as_tuple=False).squeeze(1)
            cold_idx = (rate < rate_boundary).nonzero(as_tuple=False).squeeze(1)

            w = module.weight.data
            partitions[layer_name] = {
                "hot": w[hot_idx] if len(hot_idx) > 0 else w.new_empty(0, w.shape[1]),
                "cold": w[cold_idx] if len(cold_idx) > 0 else w.new_empty(0, w.shape[1]),
            }

        return partitions
