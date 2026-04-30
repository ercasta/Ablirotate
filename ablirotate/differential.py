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
Neurons are classified into four groups by :meth:`classify_neurons`:

* **common** – active in at least one keep category *and* at least one drop
  category.  These are central neurons shared across tasks; they must be
  preserved unconditionally.
* **keep_specific** – active in at least one keep category but *not* in any
  drop category.  They reinforce desired capabilities and are kept close to the
  common neurons.
* **drop_specific** – active in at least one drop category but *not* in any
  keep category.  These are the primary candidates for removal.
* **neutral** – active in neither group.

:meth:`compute_keep_mask` keeps **common**, **keep_specific**, and **neutral**
neurons and drops **drop_specific** ones.

:meth:`prioritized_indices` returns neuron indices ordered
common → keep_specific → neutral → drop_specific, which is the preferred
ordering for :class:`~ablirotate.defrag.MatrixDefragmenter`: the most
important neurons appear at the front so that tail truncation is safe.
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

    def classify_neurons(
        self,
        keep_categories: Sequence[str],
        drop_categories: Optional[Sequence[str]] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Classify neurons per layer into four mutually exclusive groups.

        Parameters
        ----------
        keep_categories : sequence of str
            Categories the model should retain capabilities for.
        drop_categories : sequence of str, optional
            Categories the model should suppress.

        Returns
        -------
        dict[str, dict[str, torch.BoolTensor]]
            Per layer, a dict with keys ``"common"``, ``"keep_specific"``,
            ``"drop_specific"``, and ``"neutral"``.  Each value is a boolean
            mask of length *n_units* where ``True`` means the neuron belongs
            to that group.

            - **common**: active in at least one keep category *and* at least
              one drop category.  These central neurons are shared across tasks
              and must be preserved unconditionally.
            - **keep_specific**: active in at least one keep category but *not*
              in any drop category.  They reinforce desired capabilities and
              should be placed next to the common neurons.
            - **drop_specific**: active in at least one drop category but *not*
              in any keep category.  Primary candidates for removal.
            - **neutral**: active in neither group.
        """
        missing = [c for c in keep_categories if c not in self._category_stats]
        if missing:
            raise ValueError(f"Categories not recorded: {missing}")

        if drop_categories:
            missing_drop = [c for c in drop_categories if c not in self._category_stats]
            if missing_drop:
                raise ValueError(f"Drop categories not recorded: {missing_drop}")

        first = self._category_stats[keep_categories[0]]
        result: Dict[str, Dict[str, torch.Tensor]] = {}

        for layer_name, base_stats in first.items():
            n = base_stats.n_units

            active_in_any_keep = torch.zeros(n, dtype=torch.bool)
            for cat in keep_categories:
                cat_stats = self._category_stats[cat].get(layer_name)
                if cat_stats is not None:
                    active_in_any_keep |= (
                        cat_stats.activation_rate >= self.keep_rate_threshold
                    )

            active_in_any_drop = torch.zeros(n, dtype=torch.bool)
            if drop_categories:
                for cat in drop_categories:
                    cat_stats = self._category_stats[cat].get(layer_name)
                    if cat_stats is not None:
                        active_in_any_drop |= (
                            cat_stats.activation_rate >= self.drop_rate_threshold
                        )

            common = active_in_any_keep & active_in_any_drop
            keep_specific = active_in_any_keep & ~active_in_any_drop
            drop_specific = active_in_any_drop & ~active_in_any_keep
            neutral = ~active_in_any_keep & ~active_in_any_drop

            result[layer_name] = {
                "common": common,
                "keep_specific": keep_specific,
                "drop_specific": drop_specific,
                "neutral": neutral,
            }

        return result

    def compute_keep_mask(
        self,
        keep_categories: Sequence[str],
        drop_categories: Optional[Sequence[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute a boolean keep-mask for every tracked layer.

        Parameters
        ----------
        keep_categories : sequence of str
            Neurons active in **all** of these categories are kept when no
            *drop_categories* are provided.  When *drop_categories* are given,
            any neuron active in at least one keep category is retained and
            only **drop_specific** neurons (active in drop but not in any keep
            category) are removed.
        drop_categories : sequence of str, optional
            Neurons active **only** in these categories (and not in any keep
            category) are dropped.  If ``None``, only the intersection rule
            applies (neurons must appear in *every* keep category to be kept).

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
            # Delegate to classify_neurons: keep everything except drop_specific.
            # common, keep_specific, and neutral neurons are all preserved.
            classifications = self.classify_neurons(keep_categories, drop_categories)
            return {
                layer_name: ~groups["drop_specific"]
                for layer_name, groups in classifications.items()
            }

        # No drop categories: keep only neurons present in ALL keep categories
        # (intersection / "universal" rule).
        first = self._category_stats[keep_categories[0]]
        result: Dict[str, torch.Tensor] = {}

        for layer_name, base_stats in first.items():
            n = base_stats.n_units

            universal_active = torch.ones(n, dtype=torch.bool)
            for cat in keep_categories:
                cat_stats = self._category_stats[cat].get(layer_name)
                if cat_stats is None:
                    universal_active[:] = False
                    break
                universal_active &= cat_stats.activation_rate >= self.keep_rate_threshold

            result[layer_name] = universal_active

        return result

    def prioritized_indices(
        self,
        keep_categories: Sequence[str],
        drop_categories: Optional[Sequence[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return per-layer neuron indices sorted by preservation priority.

        The ordering places the most important neurons first so that
        :class:`~ablirotate.defrag.MatrixDefragmenter` can safely truncate
        from the tail:

        1. **common** – neurons active in both keep and drop categories (central; must preserve).
        2. **keep_specific** – active only in keep categories (desired).
        3. **neutral** – active in neither group (harmless).
        4. **drop_specific** – active only in drop categories (remove queue).

        Parameters
        ----------
        keep_categories : sequence of str
        drop_categories : sequence of str, optional

        Returns
        -------
        dict[str, torch.LongTensor]
            Per layer, a 1-D tensor of neuron indices in priority order.
        """
        classifications = self.classify_neurons(keep_categories, drop_categories)
        result: Dict[str, torch.Tensor] = {}

        for layer_name, groups in classifications.items():
            common_idx = groups["common"].nonzero(as_tuple=False).squeeze(1)
            keep_idx = groups["keep_specific"].nonzero(as_tuple=False).squeeze(1)
            neutral_idx = groups["neutral"].nonzero(as_tuple=False).squeeze(1)
            drop_idx = groups["drop_specific"].nonzero(as_tuple=False).squeeze(1)
            result[layer_name] = torch.cat([common_idx, keep_idx, neutral_idx, drop_idx])

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
