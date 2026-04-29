"""Tests for ActivationTracker and ActivationStats."""

import torch
import pytest

from ablirotate.tracker import ActivationStats, ActivationTracker
from conftest import make_model, random_input


# ---------------------------------------------------------------------------
# ActivationStats unit tests
# ---------------------------------------------------------------------------

class TestActivationStats:

    def test_update_3d(self):
        stats = ActivationStats(n_units=4, threshold=0.5)
        # batch=2, seq=3, units=4; all > 0.5 for first 2 neurons
        acts = torch.tensor([[[1.0, 0.0, 1.0, 0.0]] * 3] * 2)  # shape (2,3,4)
        stats.update(acts)
        assert stats.total_samples == 2
        # neurons 0 and 2 fire; 1 and 3 do not
        assert stats.activation_counts[0].item() == 2
        assert stats.activation_counts[1].item() == 0

    def test_update_2d(self):
        stats = ActivationStats(n_units=3, threshold=0.0)
        acts = torch.tensor([[1.0, -1.5, 0.0]])  # batch=1
        stats.update(acts)
        assert stats.activation_counts[0] == 1   # abs(1.0) > 0.0
        assert stats.activation_counts[1] == 1   # abs(-1.5) > 0.0
        assert stats.activation_counts[2] == 0   # abs(0.0) not > 0.0

    def test_activation_rate(self):
        stats = ActivationStats(n_units=2, threshold=0.0)
        acts = torch.tensor([[1.0, 0.0], [1.0, 0.0]])  # 2 samples, neuron 0 fires both
        stats.update(acts)
        rate = stats.activation_rate
        assert rate[0].item() == pytest.approx(1.0)
        assert rate[1].item() == pytest.approx(0.0)

    def test_least_most_active(self):
        stats = ActivationStats(n_units=3, threshold=0.0)
        # neuron 1 fires most, neuron 2 fires least
        stats.activation_counts = torch.tensor([2, 5, 1])
        stats.total_samples = 5
        least = stats.least_active_indices()
        most = stats.most_active_indices()
        assert least[0].item() == 2
        assert most[0].item() == 1

    def test_reset(self):
        stats = ActivationStats(n_units=2, threshold=0.0)
        stats.update(torch.tensor([[1.0, 1.0]]))
        stats.reset()
        assert stats.total_samples == 0
        assert stats.activation_counts.sum().item() == 0

    def test_zero_samples_rate(self):
        stats = ActivationStats(n_units=3, threshold=0.0)
        rate = stats.activation_rate
        assert (rate == 0).all()


# ---------------------------------------------------------------------------
# ActivationTracker integration tests
# ---------------------------------------------------------------------------

class TestActivationTracker:

    def test_hooks_registered(self):
        model = make_model()
        tracker = ActivationTracker(model, mlp_submodule_name="act")
        # 2 layers, each has 1 "act" sub-module
        act_keys = [k for k in tracker.stats if k.endswith(".act")]
        assert len(act_keys) == 2
        tracker.remove_hooks()

    def test_stats_accumulate_after_forward(self):
        model = make_model()
        x = random_input()
        tracker = ActivationTracker(model, mlp_submodule_name="act", track_attention=False)

        with torch.no_grad():
            model(x)

        for name, stats in tracker.stats.items():
            assert stats.total_samples > 0, f"No samples recorded for {name}"

        tracker.remove_hooks()

    def test_context_manager_removes_hooks(self):
        model = make_model()
        with ActivationTracker(model, mlp_submodule_name="act", track_attention=False) as tracker:
            with torch.no_grad():
                model(random_input())
        # After exiting, hooks should be removed
        assert len(tracker._hooks) == 0

    def test_reset_clears_stats(self):
        model = make_model()
        x = random_input()
        with ActivationTracker(model, mlp_submodule_name="act", track_attention=False) as tracker:
            with torch.no_grad():
                model(x)
            tracker.reset()
            for stats in tracker.stats.values():
                assert stats.total_samples == 0

    def test_aggregate_activation_mask(self):
        model = make_model()
        x = random_input()
        with ActivationTracker(model, mlp_submodule_name="act", track_attention=False) as tracker:
            with torch.no_grad():
                model(x)
            masks = tracker.aggregate_activation_mask(rate_threshold=0.0)
        for name, mask in masks.items():
            assert mask.dtype == torch.bool
            # All neurons that fired at least once should be True (threshold=0.0)
            assert mask.any(), f"No active neurons in {name}"

    def test_summary_returns_floats(self):
        model = make_model()
        with ActivationTracker(model, mlp_submodule_name="act", track_attention=False) as tracker:
            with torch.no_grad():
                model(random_input())
            summary = tracker.summary()
        for v in summary.values():
            assert isinstance(v, float)
            assert 0.0 <= v <= 1.0
