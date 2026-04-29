"""Tests for DifferentialAbliterator."""

import torch
import pytest

from ablirotate.tracker import ActivationStats, ActivationTracker
from ablirotate.differential import DifferentialAbliterator
from conftest import make_model, random_input


def _fake_tracker_with_stats(model, layer_stats: dict) -> ActivationTracker:
    """Create a tracker and replace its stats dict with preset values."""
    tracker = ActivationTracker(
        model,
        mlp_submodule_name="act",
        track_attention=False,
    )
    tracker.stats = layer_stats
    return tracker


def _make_stats(n_units: int, rates: list) -> ActivationStats:
    stats = ActivationStats(n_units=n_units, threshold=0.0)
    stats.total_samples = 100
    stats.activation_counts = torch.tensor(
        [int(r * 100) for r in rates], dtype=torch.long
    )
    return stats


class TestDifferentialAbliterator:

    def _setup_abliterator(self):
        """Build a DifferentialAbliterator with two pre-recorded categories."""
        model = make_model(d_ff=4)

        # Python category: neurons 0,1,2 active; 3 inactive
        python_stats = {"layer0.ffn.act": _make_stats(4, [0.9, 0.8, 0.7, 0.0])}
        # Cobol category: only neuron 3 very active; others weak
        cobol_stats = {"layer0.ffn.act": _make_stats(4, [0.05, 0.05, 0.05, 0.95])}

        tracker = _fake_tracker_with_stats(model, python_stats)
        abliterator = DifferentialAbliterator(tracker, keep_rate_threshold=0.1, drop_rate_threshold=0.1)

        # Simulate recording python category
        abliterator._category_stats["python"] = python_stats
        abliterator._category_stats["cobol"] = cobol_stats

        tracker.remove_hooks()
        return abliterator

    def test_categories_registered(self):
        abl = self._setup_abliterator()
        assert "python" in abl.categories
        assert "cobol" in abl.categories

    def test_keep_mask_keeps_active_neurons(self):
        abl = self._setup_abliterator()
        mask = abl.compute_keep_mask(keep_categories=["python"])
        # Neurons 0,1,2 active in python → should be kept
        m = mask["layer0.ffn.act"]
        assert m[0].item() is True
        assert m[1].item() is True
        assert m[2].item() is True

    def test_keep_mask_drops_cobol_only_neurons(self):
        abl = self._setup_abliterator()
        mask = abl.compute_keep_mask(
            keep_categories=["python"],
            drop_categories=["cobol"],
        )
        m = mask["layer0.ffn.act"]
        # Neuron 3 is active only in cobol → should NOT be kept
        assert m[3].item() is False

    def test_missing_category_raises(self):
        abl = self._setup_abliterator()
        with pytest.raises(ValueError, match="not recorded"):
            abl.compute_keep_mask(keep_categories=["javascript"])

    def test_activation_overlap_same_category(self):
        abl = self._setup_abliterator()
        # Overlap of python with itself should be 1.0
        overlap = abl.activation_overlap("python", "python")
        for v in overlap.values():
            assert v == pytest.approx(1.0)

    def test_activation_overlap_different_categories(self):
        abl = self._setup_abliterator()
        overlap = abl.activation_overlap("python", "cobol")
        # Neurons 0-2 active in python, neuron 3 active in cobol → no overlap
        for v in overlap.values():
            assert 0.0 <= v <= 1.0

    def test_summary(self):
        abl = self._setup_abliterator()
        summary = abl.summary()
        assert "python" in summary
        assert "cobol" in summary
        for cat_data in summary.values():
            for v in cat_data.values():
                assert 0.0 <= v <= 1.0

    def test_record_snapshots_and_resets(self):
        """Test that record() captures stats and resets the tracker."""
        model = make_model(d_ff=4)
        init_stats = {"layer0.ffn.act": _make_stats(4, [0.5, 0.5, 0.5, 0.5])}
        tracker = _fake_tracker_with_stats(model, init_stats)
        abliterator = DifferentialAbliterator(tracker)

        abliterator.record("my_category")
        assert "my_category" in abliterator._category_stats
        # Tracker stats should be reset
        for stats in tracker.stats.values():
            assert stats.total_samples == 0
        tracker.remove_hooks()
