"""Tests for MatrixDefragmenter."""

import torch
import pytest

from ablirotate.tracker import ActivationStats
from ablirotate.defrag import MatrixDefragmenter
from conftest import make_model


def _make_stats(n_units: int, rates: list) -> ActivationStats:
    stats = ActivationStats(n_units=n_units, threshold=0.0)
    stats.total_samples = 100
    stats.activation_counts = torch.tensor(
        [int(r * 100) for r in rates], dtype=torch.long
    )
    return stats


class TestMatrixDefragmenter:

    def test_defragment_reorders_weights(self):
        model = make_model(d_ff=4)
        act_module = model.layer0.ffn.act
        # Set distinct row values so we can track reordering
        act_module.weight.data = torch.arange(
            act_module.weight.data.numel(), dtype=torch.float
        ).reshape(act_module.weight.data.shape)

        original_row0 = act_module.weight.data[0].clone()
        original_row3 = act_module.weight.data[3].clone()

        # neuron 3 is most active, neuron 0 is least
        stats = {
            "layer0.ffn.act": _make_stats(4, rates=[0.1, 0.4, 0.6, 0.9])
        }
        defrag = MatrixDefragmenter(model, stats, keep_fraction=1.0)
        result = defrag.defragment()

        assert "layer0.ffn.act" in result
        new_w = act_module.weight.data
        # Row 0 of the new weight should be the old row 3 (most active)
        assert torch.allclose(new_w[0], original_row3)

    def test_keep_fraction_truncates(self):
        model = make_model(d_ff=4)
        stats = {
            "layer0.ffn.act": _make_stats(4, rates=[0.1, 0.4, 0.6, 0.9])
        }
        defrag = MatrixDefragmenter(model, stats, keep_fraction=0.5)
        result = defrag.defragment()

        orig, new = result["layer0.ffn.act"]
        assert orig == 4
        assert new == 2

        # Weight should now have 2 rows
        assert model.layer0.ffn.act.weight.data.shape[0] == 2

    def test_invalid_keep_fraction_raises(self):
        model = make_model()
        with pytest.raises(ValueError):
            MatrixDefragmenter(model, {}, keep_fraction=0.0)
        with pytest.raises(ValueError):
            MatrixDefragmenter(model, {}, keep_fraction=1.5)

    def test_partition_returns_hot_cold(self):
        model = make_model(d_ff=4)
        stats = {
            "layer0.ffn.act": _make_stats(4, rates=[0.05, 0.5, 0.05, 0.8])
        }
        defrag = MatrixDefragmenter(model, stats)
        parts = defrag.partition(rate_boundary=0.1)

        assert "layer0.ffn.act" in parts
        hot = parts["layer0.ffn.act"]["hot"]
        cold = parts["layer0.ffn.act"]["cold"]
        # neurons 1 and 3 are hot (rate >= 0.1)
        assert hot.shape[0] == 2
        # neurons 0 and 2 are cold
        assert cold.shape[0] == 2

    def test_compute_permutation(self):
        model = make_model(d_ff=4)
        stats = {
            "layer0.ffn.act": _make_stats(4, rates=[0.1, 0.4, 0.6, 0.9])
        }
        defrag = MatrixDefragmenter(model, stats, keep_fraction=1.0)
        perm = defrag.compute_permutation("layer0.ffn.act")
        assert perm is not None
        assert perm[0].item() == 3   # most active first
        assert len(perm) == 4
