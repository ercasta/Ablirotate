"""Tests for ModelPruner."""

import torch
import torch.nn as nn
import pytest

from ablirotate.tracker import ActivationStats, ActivationTracker
from ablirotate.pruner import ModelPruner
from conftest import make_model, random_input


def _make_stats_with_rates(n_units: int, rates: list) -> ActivationStats:
    """Create an ActivationStats object with preset activation rates."""
    stats = ActivationStats(n_units=n_units, threshold=0.0)
    total = 100
    stats.total_samples = total
    stats.activation_counts = torch.tensor(
        [int(r * total) for r in rates], dtype=torch.long
    )
    return stats


class TestModelPrunerSoft:

    def test_soft_prune_scales_down_weights(self):
        model = make_model(d_ff=4)
        # Manually set the act layer weight to all-ones for easy checking
        act_module = model.layer0.ffn.act
        act_module.weight.data.fill_(1.0)

        stats = {
            "layer0.ffn.act": _make_stats_with_rates(
                n_units=4, rates=[0.9, 0.0, 0.9, 0.0]  # neurons 1,3 below 0.1
            )
        }
        pruner = ModelPruner(model, stats, rate_threshold=0.1, high_temp_scale=0.0)
        pruner.prune(mode="soft")

        w = act_module.weight.data
        # Rows 1 and 3 should be zeroed (scaled by 0.0)
        assert w[1].abs().sum().item() == pytest.approx(0.0)
        assert w[3].abs().sum().item() == pytest.approx(0.0)
        # Rows 0 and 2 untouched
        assert w[0].abs().sum().item() > 0
        assert w[2].abs().sum().item() > 0

    def test_soft_prune_returns_count(self):
        model = make_model(d_ff=4)
        stats = {
            "layer0.ffn.act": _make_stats_with_rates(
                n_units=4, rates=[0.9, 0.0, 0.0, 0.9]
            )
        }
        pruner = ModelPruner(model, stats, rate_threshold=0.1)
        result = pruner.prune(mode="soft")
        assert result.get("layer0.ffn.act", 0) == 2


class TestModelPrunerHard:

    def test_hard_prune_zeros_weights(self):
        model = make_model(d_ff=4)
        act_module = model.layer0.ffn.act
        act_module.weight.data.fill_(1.0)

        stats = {
            "layer0.ffn.act": _make_stats_with_rates(
                n_units=4, rates=[0.0, 0.9, 0.0, 0.9]
            )
        }
        pruner = ModelPruner(model, stats, rate_threshold=0.1)
        pruner.prune(mode="hard")

        w = act_module.weight.data
        assert w[0].abs().sum().item() == pytest.approx(0.0)
        assert w[2].abs().sum().item() == pytest.approx(0.0)
        assert w[1].abs().sum().item() > 0
        assert w[3].abs().sum().item() > 0


class TestModelPrunerCold:

    def test_cold_prune_adds_negative_bias(self):
        model = make_model(d_ff=4)
        act_module = model.layer0.ffn.act

        stats = {
            "layer0.ffn.act": _make_stats_with_rates(
                n_units=4, rates=[0.0, 0.9, 0.0, 0.9]
            )
        }
        pruner = ModelPruner(model, stats, rate_threshold=0.1)
        pruner.prune(mode="cold")

        # Cold neurons should have negative bias
        assert hasattr(act_module, "bias") and act_module.bias is not None
        bias = act_module.bias.data
        assert bias[0].item() < 0
        assert bias[2].item() < 0


class TestPruneToMask:

    def test_prune_to_mask(self):
        model = make_model(d_ff=4)
        act_module = model.layer0.ffn.act
        act_module.weight.data.fill_(1.0)

        keep_mask = {
            "layer0.ffn.act": torch.tensor([True, False, True, False])
        }
        stats = {
            "layer0.ffn.act": _make_stats_with_rates(4, [0.9] * 4)
        }
        pruner = ModelPruner(model, stats, rate_threshold=0.0, high_temp_scale=0.0)
        result = pruner.prune_to_mask(keep_mask, mode="soft")

        w = act_module.weight.data
        assert w[1].abs().sum().item() == pytest.approx(0.0)
        assert w[3].abs().sum().item() == pytest.approx(0.0)
        assert w[0].abs().sum().item() > 0
        assert w[2].abs().sum().item() > 0
        assert result["layer0.ffn.act"] == 2
