"""
Differential abliterator: collect activation masks across multiple prompt
categories, then compute which neurons are *universal* (active in all desired
categories) versus *category-specific* (active only in some).

Typical workflow
----------------
1. Create a :class:`DifferentialAbliterator` wrapping your model.
2. Call :meth:`record` for each category of prompts (e.g. "python",
   "javascript", "italian", "english").
3. Call :meth:`compute_keep_mask` with the categories you want to preserve
   (e.g. ``keep_categories=["python", "italian"]`` and
   ``drop_categories=["cobol", "japanese"]``).
4. Pass the resulting ``keep_mask`` to
   :class:`~ablirotate.pruner.ModelPruner` to actually remove the unwanted
   neurons.

The algorithm
-------------
* A neuron is **kept** if its activation rate in *every* keep category is
  above ``keep_rate_threshold``.
* A neuron is **dropped** if its activation rate in *any* drop category is
  above ``drop_rate_threshold`` AND it was not already kept by the above rule.
* All other neurons are **kept** (neutral → no change).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from .tracker import ActivationStats, ActivationTracker


class DifferentialAbliterator:
    """Identify and optionally remove category-specific neurons.

    Parameters
    ----------
    tracker : ActivationTracker
        An already-initialised tracker attached to the model.
    keep_rate_threshold : float
        Minimum activation rate for a neuron to be considered "universally
        active" in a category.
    drop_rate_threshold : float
        Minimum activation rate in *drop* categories for a neuron to be
        flagged for removal.
    """

    def __init__(
        self,
        tracker: ActivationTracker,
        keep_rate_threshold: float = 0.1,
        drop_rate_threshold: float = 0.1,
    ) -> None:
        self.tracker = tracker
        self.keep_rate_threshold = keep_rate_threshold
        self.drop_rate_threshold = drop_rate_threshold

        # category → {layer_name → ActivationStats snapshot}
        self._category_stats: Dict[str, Dict[str, ActivationStats]] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, category: str) -> None:
        """Snapshot current tracker statistics under *category*.

        Call this **after** running inference for one category of prompts.
        The tracker is then reset for the next category.
        """
        snapshot: Dict[str, ActivationStats] = {}
        for name, stats in self.tracker.stats.items():
            snap = ActivationStats(stats.n_units, stats.threshold)
            snap.activation_counts = stats.activation_counts.clone()
            snap.total_samples = stats.total_samples
            snapshot[name] = snap
        self._category_stats[category] = snapshot
        self.tracker.reset()

    @property
    def categories(self) -> List[str]:
        return list(self._category_stats.keys())

    # ------------------------------------------------------------------
    # Mask computation
    # ------------------------------------------------------------------

    def compute_keep_mask(
        self,
        keep_categories: Sequence[str],
        drop_categories: Optional[Sequence[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute a boolean keep-mask for every tracked layer.

        Parameters
        ----------
        keep_categories : sequence of str
            Neurons active in **all** of these categories are kept.
        drop_categories : sequence of str, optional
            Neurons active **only** in these categories (and not in any keep
            category) are dropped.  If ``None``, only the intersection rule
            applies.

        Returns
        -------
        dict[str, torch.BoolTensor]
            ``True`` = keep this neuron.  Suitable for passing directly to
            :meth:`~ablirotate.pruner.ModelPruner.prune_to_mask`.
        """
        missing = [c for c in keep_categories if c not in self._category_stats]
        if missing:
            raise ValueError(f"Categories not recorded: {missing}")

        if drop_categories:
            missing_drop = [c for c in drop_categories if c not in self._category_stats]
            if missing_drop:
                raise ValueError(f"Drop categories not recorded: {missing_drop}")

        # Collect layer names from the first keep category
        first = self._category_stats[keep_categories[0]]
        result: Dict[str, torch.Tensor] = {}

        for layer_name, base_stats in first.items():
            n = base_stats.n_units

            # A neuron is universally active if it passes threshold in EVERY
            # keep category.
            universal_active = torch.ones(n, dtype=torch.bool)
            for cat in keep_categories:
                cat_stats = self._category_stats[cat].get(layer_name)
                if cat_stats is None:
                    universal_active[:] = False
                    break
                rate = cat_stats.activation_rate
                universal_active &= rate >= self.keep_rate_threshold

            # Build keep mask: start by keeping universally active neurons
            keep_mask = universal_active.clone()

            if drop_categories:
                # Additionally, mark neurons that are active ONLY in drop
                # categories (not in any keep category) as dropped.
                active_in_any_keep = torch.zeros(n, dtype=torch.bool)
                for cat in keep_categories:
                    cat_stats = self._category_stats[cat].get(layer_name)
                    if cat_stats is not None:
                        active_in_any_keep |= (
                            cat_stats.activation_rate >= self.keep_rate_threshold
                        )

                active_in_any_drop = torch.zeros(n, dtype=torch.bool)
                for cat in drop_categories:
                    cat_stats = self._category_stats[cat].get(layer_name)
                    if cat_stats is not None:
                        active_in_any_drop |= (
                            cat_stats.activation_rate >= self.drop_rate_threshold
                        )

                # Neurons active only in drop categories (not in any keep) → remove
                drop_only = active_in_any_drop & ~active_in_any_keep
                keep_mask = keep_mask | (~drop_only)

            result[layer_name] = keep_mask

        return result

    def activation_overlap(
        self, cat_a: str, cat_b: str
    ) -> Dict[str, float]:
        """Jaccard overlap of active neuron sets between two categories.

        Returns a value in [0, 1] per layer; 1 = identical active sets.
        """
        if cat_a not in self._category_stats:
            raise ValueError(f"Category not recorded: {cat_a!r}")
        if cat_b not in self._category_stats:
            raise ValueError(f"Category not recorded: {cat_b!r}")

        overlaps: Dict[str, float] = {}
        for layer_name in self._category_stats[cat_a]:
            stats_a = self._category_stats[cat_a].get(layer_name)
            stats_b = self._category_stats[cat_b].get(layer_name)
            if stats_a is None or stats_b is None:
                continue

            active_a = stats_a.activation_rate >= self.keep_rate_threshold
            active_b = stats_b.activation_rate >= self.keep_rate_threshold

            intersection = (active_a & active_b).sum().item()
            union = (active_a | active_b).sum().item()
            overlaps[layer_name] = intersection / union if union > 0 else 1.0

        return overlaps

    # ------------------------------------------------------------------
    # Convenience: full pipeline
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Mean activation rate per category per layer."""
        out: Dict[str, Dict[str, float]] = {}
        for cat, layer_map in self._category_stats.items():
            out[cat] = {
                name: float(s.activation_rate.mean())
                for name, s in layer_map.items()
            }
        return out
