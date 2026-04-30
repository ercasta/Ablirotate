"""Tests for the Gemma 4 specific toolkit."""

from typing import Dict

import torch
import torch.nn as nn
import pytest

from ablirotate.gemma4 import (
    GEMMA4_27B_CONFIG,
    Gemma4ActivationTracker,
    Gemma4MlpPruner,
    Gemma4Defragmenter,
    Gemma4Pipeline,
)
from ablirotate.tracker import ActivationStats


# ---------------------------------------------------------------------------
# Minimal Gemma 4-like model for testing
# ---------------------------------------------------------------------------

class _GemmaMlp(nn.Module):
    """Minimal GeGLU MLP matching Gemma 4 internal structure."""

    def __init__(self, hidden_size: int = 16, intermediate_size: int = 32) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.GELU(approximate="tanh")  # Gemma 4 uses GeGLU

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _GemmaSelfAttn(nn.Module):
    """Minimal attention module with separate o_proj (Gemma 4 naming)."""

    def __init__(self, hidden_size: int = 16, n_heads: int = 2) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.n_heads = n_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simplified: apply v_proj then o_proj (no actual multi-head attention)
        return self.o_proj(self.v_proj(x))


class _GemmaLayer(nn.Module):
    """One Gemma 4-like transformer layer (self_attn + mlp)."""

    def __init__(
        self,
        hidden_size: int = 16,
        intermediate_size: int = 32,
        n_heads: int = 2,
    ) -> None:
        super().__init__()
        self.self_attn = _GemmaSelfAttn(hidden_size, n_heads)
        self.mlp = _GemmaMlp(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(x)
        return x + self.mlp(x)


class _TinyGemmaModel(nn.Module):
    """Tiny Gemma 4-like model used across all tests."""

    def __init__(
        self,
        n_layers: int = 2,
        hidden_size: int = 16,
        intermediate_size: int = 32,
        n_heads: int = 2,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _GemmaLayer(hidden_size, intermediate_size, n_heads)
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        # Accept either positional `x` or the HuggingFace `inputs_embeds` kwarg.
        if x is None:
            x = inputs_embeds
        for layer in self.layers:
            x = layer(x)
        return x


# Config override for the tiny test model
_TEST_CONFIG = {
    **GEMMA4_27B_CONFIG,
    "num_attention_heads": 2,  # tiny model has 2 heads, not 32
}

_HIDDEN = 16
_INTERMEDIATE = 32
_N_HEADS = 2
_N_LAYERS = 2


def _make_model(n_layers: int = _N_LAYERS) -> _TinyGemmaModel:
    torch.manual_seed(0)
    return _TinyGemmaModel(
        n_layers=n_layers,
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        n_heads=_N_HEADS,
    ).eval()


def _random_input(batch: int = 2, seq: int = 4) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(batch, seq, _HIDDEN)


def _make_mlp_stats(n_units: int, rates: list) -> ActivationStats:
    stats = ActivationStats(n_units=n_units, threshold=0.0)
    stats.total_samples = 100
    stats.activation_counts = torch.tensor(
        [int(r * 100) for r in rates], dtype=torch.long
    )
    return stats


# ---------------------------------------------------------------------------
# Gemma4ActivationTracker tests
# ---------------------------------------------------------------------------

class TestGemma4ActivationTracker:

    def test_mlp_hooks_registered(self):
        model = _make_model()
        tracker = Gemma4ActivationTracker(
            model, config=_TEST_CONFIG, track_attention=False
        )
        # 2 layers → 2 MLP stats entries keyed by parent path
        assert len(tracker.mlp_stats) == _N_LAYERS
        assert len(tracker.attn_stats) == 0
        tracker.remove_hooks()

    def test_attn_hooks_registered(self):
        model = _make_model()
        tracker = Gemma4ActivationTracker(
            model, config=_TEST_CONFIG, track_mlp=False
        )
        assert len(tracker.attn_stats) == _N_LAYERS
        assert len(tracker.mlp_stats) == 0
        tracker.remove_hooks()

    def test_mlp_stats_accumulate_after_forward(self):
        model = _make_model()
        x = _random_input()
        tracker = Gemma4ActivationTracker(
            model, config=_TEST_CONFIG, track_attention=False
        )
        with torch.no_grad():
            model(x)

        for name, stats in tracker.mlp_stats.items():
            assert stats.total_samples > 0, f"No MLP samples for {name}"
            assert stats.n_units == _INTERMEDIATE
        tracker.remove_hooks()

    def test_attn_stats_accumulate_after_forward(self):
        model = _make_model()
        x = _random_input()
        tracker = Gemma4ActivationTracker(
            model, config=_TEST_CONFIG, track_mlp=False
        )
        with torch.no_grad():
            model(x)

        for name, stats in tracker.attn_stats.items():
            assert stats.total_samples > 0, f"No attn samples for {name}"
            assert stats.n_units == _N_HEADS
        tracker.remove_hooks()

    def test_stats_property_combines_both(self):
        model = _make_model()
        tracker = Gemma4ActivationTracker(model, config=_TEST_CONFIG)
        # 2 layers × (1 mlp + 1 attn) = 4 entries
        assert len(tracker.stats) == _N_LAYERS * 2
        tracker.remove_hooks()

    def test_context_manager_removes_hooks(self):
        model = _make_model()
        with Gemma4ActivationTracker(
            model, config=_TEST_CONFIG
        ) as tracker:
            with torch.no_grad():
                model(_random_input())
        assert len(tracker._hooks) == 0

    def test_reset_clears_stats(self):
        model = _make_model()
        with Gemma4ActivationTracker(
            model, config=_TEST_CONFIG, track_attention=False
        ) as tracker:
            with torch.no_grad():
                model(_random_input())
            tracker.reset()
            for stats in tracker.mlp_stats.values():
                assert stats.total_samples == 0
                assert stats.activation_counts.sum().item() == 0

    def test_aggregate_activation_mask_returns_bool(self):
        model = _make_model()
        with Gemma4ActivationTracker(
            model, config=_TEST_CONFIG, track_attention=False
        ) as tracker:
            with torch.no_grad():
                model(_random_input())
            masks = tracker.aggregate_mlp_mask(rate_threshold=0.0)
        for name, mask in masks.items():
            assert mask.dtype == torch.bool
            assert mask.any(), f"No active neurons in {name}"

    def test_summary_returns_floats_in_unit_interval(self):
        model = _make_model()
        with Gemma4ActivationTracker(model, config=_TEST_CONFIG) as tracker:
            with torch.no_grad():
                model(_random_input())
            summary = tracker.summary()
        for v in summary.values():
            assert isinstance(v, float)
            assert 0.0 <= v <= 1.0

    def test_mlp_keys_contain_mlp_parent(self):
        """MLP stats keys should be parent mlp paths, not down_proj paths."""
        model = _make_model()
        tracker = Gemma4ActivationTracker(
            model, config=_TEST_CONFIG, track_attention=False
        )
        for key in tracker.mlp_stats:
            assert key.endswith("mlp"), f"Unexpected key: {key}"
        tracker.remove_hooks()

    def test_aggregate_attn_mask_returns_bool(self):
        model = _make_model()
        with Gemma4ActivationTracker(
            model, config=_TEST_CONFIG, track_mlp=False
        ) as tracker:
            with torch.no_grad():
                model(_random_input())
            masks = tracker.aggregate_attn_mask(rate_threshold=0.0)
        for name, mask in masks.items():
            assert mask.dtype == torch.bool

    def test_no_hooks_registered_warns(self):
        """A bad config that matches nothing should emit a warning."""
        model = _make_model()
        bad_config = {**_TEST_CONFIG, "down_proj": "nonexistent_proj"}
        with pytest.warns(UserWarning, match="No hooks were registered"):
            Gemma4ActivationTracker(
                model, config=bad_config, track_attention=False
            ).remove_hooks()


# ---------------------------------------------------------------------------
# Gemma4MlpPruner tests
# ---------------------------------------------------------------------------

class TestGemma4MlpPruner:

    def _stats_for_layer0(self, rates: list) -> Dict:
        return {"layers.0.mlp": _make_mlp_stats(_INTERMEDIATE, rates)}

    def test_soft_prune_zeros_gate_up_down(self):
        model = _make_model(n_layers=1)
        mlp = model.layers[0].mlp
        mlp.gate_proj.weight.data.fill_(1.0)
        mlp.up_proj.weight.data.fill_(1.0)
        mlp.down_proj.weight.data.fill_(1.0)

        # Neurons 0 and 1 inactive
        rates = [0.0, 0.0] + [0.9] * (_INTERMEDIATE - 2)
        pruner = Gemma4MlpPruner(
            model,
            self._stats_for_layer0(rates),
            config=_TEST_CONFIG,
            rate_threshold=0.1,
            high_temp_scale=0.0,
        )
        result = pruner.prune(mode="soft")

        assert result.get("layers.0.mlp", 0) == 2
        # gate_proj rows 0, 1 should be zeroed
        assert mlp.gate_proj.weight.data[0].abs().sum().item() == pytest.approx(0.0)
        assert mlp.gate_proj.weight.data[1].abs().sum().item() == pytest.approx(0.0)
        # up_proj rows 0, 1 should be zeroed
        assert mlp.up_proj.weight.data[0].abs().sum().item() == pytest.approx(0.0)
        assert mlp.up_proj.weight.data[1].abs().sum().item() == pytest.approx(0.0)
        # down_proj columns 0, 1 should be zeroed
        assert mlp.down_proj.weight.data[:, 0].abs().sum().item() == pytest.approx(0.0)
        assert mlp.down_proj.weight.data[:, 1].abs().sum().item() == pytest.approx(0.0)
        # Active neurons untouched
        assert mlp.gate_proj.weight.data[2].abs().sum().item() > 0
        assert mlp.up_proj.weight.data[2].abs().sum().item() > 0
        assert mlp.down_proj.weight.data[:, 2].abs().sum().item() > 0

    def test_hard_prune_zeros_weights(self):
        model = _make_model(n_layers=1)
        mlp = model.layers[0].mlp
        mlp.gate_proj.weight.data.fill_(1.0)

        rates = [0.0] * 4 + [0.9] * (_INTERMEDIATE - 4)
        pruner = Gemma4MlpPruner(
            model,
            self._stats_for_layer0(rates),
            config=_TEST_CONFIG,
            rate_threshold=0.1,
        )
        pruner.prune(mode="hard")

        for i in range(4):
            assert mlp.gate_proj.weight.data[i].abs().sum().item() == pytest.approx(0.0)
        assert mlp.gate_proj.weight.data[4].abs().sum().item() > 0

    def test_cold_prune_adds_negative_bias(self):
        model = _make_model(n_layers=1)
        mlp = model.layers[0].mlp

        rates = [0.0, 0.0] + [0.9] * (_INTERMEDIATE - 2)
        pruner = Gemma4MlpPruner(
            model,
            self._stats_for_layer0(rates),
            config=_TEST_CONFIG,
            rate_threshold=0.1,
        )
        pruner.prune(mode="cold")

        assert mlp.gate_proj.bias is not None
        bias = mlp.gate_proj.bias.data
        assert bias[0].item() < 0
        assert bias[1].item() < 0

    def test_prune_returns_count(self):
        model = _make_model(n_layers=1)
        rates = [0.9] * _INTERMEDIATE
        rates[5] = 0.0
        rates[10] = 0.0
        pruner = Gemma4MlpPruner(
            model,
            self._stats_for_layer0(rates),
            config=_TEST_CONFIG,
            rate_threshold=0.1,
        )
        result = pruner.prune(mode="soft")
        assert result["layers.0.mlp"] == 2

    def test_prune_to_mask(self):
        model = _make_model(n_layers=1)
        mlp = model.layers[0].mlp
        mlp.gate_proj.weight.data.fill_(1.0)

        keep_mask = {
            "layers.0.mlp": torch.tensor(
                [False, False] + [True] * (_INTERMEDIATE - 2)
            )
        }
        rates = [0.9] * _INTERMEDIATE
        pruner = Gemma4MlpPruner(
            model,
            self._stats_for_layer0(rates),
            config=_TEST_CONFIG,
            rate_threshold=0.0,
            high_temp_scale=0.0,
        )
        result = pruner.prune_to_mask(keep_mask, mode="soft")

        assert result["layers.0.mlp"] == 2
        assert mlp.gate_proj.weight.data[0].abs().sum().item() == pytest.approx(0.0)
        assert mlp.gate_proj.weight.data[2].abs().sum().item() > 0

    def test_no_prune_when_all_active(self):
        model = _make_model(n_layers=1)
        rates = [0.9] * _INTERMEDIATE
        pruner = Gemma4MlpPruner(
            model,
            self._stats_for_layer0(rates),
            config=_TEST_CONFIG,
            rate_threshold=0.1,
        )
        result = pruner.prune(mode="hard")
        assert result.get("layers.0.mlp", 0) == 0

    def test_invalid_mode_raises(self):
        model = _make_model(n_layers=1)
        rates = [0.0] * _INTERMEDIATE
        pruner = Gemma4MlpPruner(
            model,
            self._stats_for_layer0(rates),
            config=_TEST_CONFIG,
            rate_threshold=0.5,  # all neurons below threshold → pruning is triggered
        )
        with pytest.raises(ValueError, match="Unknown pruning mode"):
            pruner.prune(mode="unknown")

    def test_multi_layer_prune(self):
        model = _make_model(n_layers=2)
        rates = [0.0] * 2 + [0.9] * (_INTERMEDIATE - 2)
        mlp_stats = {
            "layers.0.mlp": _make_mlp_stats(_INTERMEDIATE, rates),
            "layers.1.mlp": _make_mlp_stats(_INTERMEDIATE, rates),
        }
        pruner = Gemma4MlpPruner(
            model, mlp_stats, config=_TEST_CONFIG, rate_threshold=0.1
        )
        result = pruner.prune(mode="hard")
        assert result["layers.0.mlp"] == 2
        assert result["layers.1.mlp"] == 2

    def test_soft_prune_with_nonzero_scale(self):
        """Soft prune with high_temp_scale=0.5 should halve the weights."""
        model = _make_model(n_layers=1)
        mlp = model.layers[0].mlp
        mlp.gate_proj.weight.data.fill_(1.0)

        rates = [0.0] + [0.9] * (_INTERMEDIATE - 1)
        pruner = Gemma4MlpPruner(
            model,
            self._stats_for_layer0(rates),
            config=_TEST_CONFIG,
            rate_threshold=0.1,
            high_temp_scale=0.5,
        )
        pruner.prune(mode="soft")

        assert mlp.gate_proj.weight.data[0].abs().sum().item() == pytest.approx(
            0.5 * _HIDDEN
        )
        assert mlp.gate_proj.weight.data[1].abs().sum().item() == pytest.approx(_HIDDEN)


# ---------------------------------------------------------------------------
# Gemma4Defragmenter tests
# ---------------------------------------------------------------------------

class TestGemma4Defragmenter:

    def test_defragment_returns_size_info(self):
        model = _make_model(n_layers=1)
        rates = [float(i) / (_INTERMEDIATE - 1) for i in range(_INTERMEDIATE)]
        mlp_stats = {"layers.0.mlp": _make_mlp_stats(_INTERMEDIATE, rates)}

        defrag = Gemma4Defragmenter(
            model, mlp_stats, config=_TEST_CONFIG, keep_fraction=0.5
        )
        result = defrag.defragment()

        assert "layers.0.mlp" in result
        orig, new = result["layers.0.mlp"]
        assert orig == _INTERMEDIATE
        assert new == _INTERMEDIATE // 2

    def test_defragment_truncates_all_three_matrices(self):
        model = _make_model(n_layers=1)
        mlp = model.layers[0].mlp
        rates = [float(i) / (_INTERMEDIATE - 1) for i in range(_INTERMEDIATE)]
        mlp_stats = {"layers.0.mlp": _make_mlp_stats(_INTERMEDIATE, rates)}

        defrag = Gemma4Defragmenter(
            model, mlp_stats, config=_TEST_CONFIG, keep_fraction=0.5
        )
        defrag.defragment()

        half = _INTERMEDIATE // 2
        assert mlp.gate_proj.weight.shape[0] == half
        assert mlp.up_proj.weight.shape[0] == half
        assert mlp.down_proj.weight.shape[1] == half

    def test_most_active_neuron_placed_first(self):
        model = _make_model(n_layers=1)
        mlp = model.layers[0].mlp
        # Neuron _INTERMEDIATE-1 has the highest rate
        rates = [float(i) / (_INTERMEDIATE - 1) for i in range(_INTERMEDIATE)]
        orig_gate_last = mlp.gate_proj.weight.data[-1].clone()
        mlp_stats = {"layers.0.mlp": _make_mlp_stats(_INTERMEDIATE, rates)}

        defrag = Gemma4Defragmenter(
            model, mlp_stats, config=_TEST_CONFIG, keep_fraction=1.0
        )
        defrag.defragment()

        # After sorting, the originally most-active neuron should be at index 0
        assert torch.allclose(mlp.gate_proj.weight.data[0], orig_gate_last)

    def test_keep_fraction_one_does_not_change_shape(self):
        model = _make_model(n_layers=1)
        mlp = model.layers[0].mlp
        rates = [0.5] * _INTERMEDIATE
        mlp_stats = {"layers.0.mlp": _make_mlp_stats(_INTERMEDIATE, rates)}

        defrag = Gemma4Defragmenter(
            model, mlp_stats, config=_TEST_CONFIG, keep_fraction=1.0
        )
        defrag.defragment()

        assert mlp.gate_proj.weight.shape[0] == _INTERMEDIATE
        assert mlp.down_proj.weight.shape[1] == _INTERMEDIATE

    def test_invalid_keep_fraction_raises(self):
        model = _make_model(n_layers=1)
        with pytest.raises(ValueError, match="keep_fraction"):
            Gemma4Defragmenter(model, {}, config=_TEST_CONFIG, keep_fraction=0.0)

    def test_compute_permutation(self):
        model = _make_model(n_layers=1)
        rates = [float(i) / (_INTERMEDIATE - 1) for i in range(_INTERMEDIATE)]
        mlp_stats = {"layers.0.mlp": _make_mlp_stats(_INTERMEDIATE, rates)}

        defrag = Gemma4Defragmenter(
            model, mlp_stats, config=_TEST_CONFIG, keep_fraction=0.5
        )
        perm = defrag.compute_permutation("layers.0.mlp")

        assert perm is not None
        assert len(perm) == _INTERMEDIATE // 2
        # First element should be the index of the most active neuron
        assert perm[0].item() == _INTERMEDIATE - 1

    def test_compute_permutation_unknown_layer(self):
        model = _make_model(n_layers=1)
        defrag = Gemma4Defragmenter(model, {}, config=_TEST_CONFIG)
        assert defrag.compute_permutation("nonexistent.mlp") is None

    def test_defragment_preserves_forward_numerics(self):
        """After keep_fraction=1.0 defragment, output should be numerically identical."""
        model = _make_model(n_layers=1)
        x = _random_input(batch=1, seq=2)

        with torch.no_grad():
            out_before = model(x).clone()

        rates = [float(i) / (_INTERMEDIATE - 1) for i in range(_INTERMEDIATE)]
        mlp_stats = {"layers.0.mlp": _make_mlp_stats(_INTERMEDIATE, rates)}

        defrag = Gemma4Defragmenter(
            model, mlp_stats, config=_TEST_CONFIG, keep_fraction=1.0
        )
        defrag.defragment()

        with torch.no_grad():
            out_after = model(x)

        assert torch.allclose(out_before, out_after, atol=1e-5)


# ---------------------------------------------------------------------------
# Integration: tracker → pruner roundtrip
# ---------------------------------------------------------------------------

class TestTrackerPrunerIntegration:

    def test_tracker_output_feeds_pruner(self):
        model = _make_model()
        x = _random_input()

        tracker = Gemma4ActivationTracker(
            model, config=_TEST_CONFIG, track_attention=False
        )
        with torch.no_grad():
            model(x)

        pruner = Gemma4MlpPruner(
            model,
            tracker.mlp_stats,
            config=_TEST_CONFIG,
            rate_threshold=0.0,  # prune nothing (all fire at threshold=0)
        )
        result = pruner.prune(mode="soft")

        # At threshold=0 all active neurons should be kept → 0 pruned
        for n_pruned in result.values():
            assert n_pruned == 0

        tracker.remove_hooks()

    def test_tracker_output_feeds_defragmenter(self):
        model = _make_model()
        x = _random_input()

        tracker = Gemma4ActivationTracker(
            model, config=_TEST_CONFIG, track_attention=False
        )
        with torch.no_grad():
            model(x)

        defrag = Gemma4Defragmenter(
            model,
            tracker.mlp_stats,
            config=_TEST_CONFIG,
            keep_fraction=0.75,
        )
        result = defrag.defragment()

        expected_keep = max(1, int(_INTERMEDIATE * 0.75))
        for orig, new in result.values():
            assert orig == _INTERMEDIATE
            assert new == expected_keep

        tracker.remove_hooks()


# ---------------------------------------------------------------------------
# Gemma4Pipeline smoke tests (no tokenizer needed for the hook-only parts)
# ---------------------------------------------------------------------------

class TestGemma4Pipeline:

    def test_pipeline_context_manager_removes_hooks(self):
        model = _make_model()

        class _FakeTokenizer:
            def __call__(self, text, return_tensors=None):
                return {"inputs_embeds": _random_input(batch=1, seq=3)}

        with Gemma4Pipeline(model, _FakeTokenizer(), config=_TEST_CONFIG) as pipeline:
            assert len(pipeline.tracker._hooks) > 0
        assert len(pipeline.tracker._hooks) == 0

    def test_pipeline_prune_returns_dict(self):
        model = _make_model()

        class _FakeTokenizer:
            def __call__(self, text, return_tensors=None):
                return {"inputs_embeds": _random_input(batch=1, seq=3)}

        with Gemma4Pipeline(model, _FakeTokenizer(), config=_TEST_CONFIG) as pipeline:
            pipeline.run_prompts(["hello", "world"])
            result = pipeline.prune(rate_threshold=0.0, mode="soft")

        assert isinstance(result, dict)
        assert len(result) == _N_LAYERS

    def test_pipeline_defragment_returns_dict(self):
        model = _make_model()

        class _FakeTokenizer:
            def __call__(self, text, return_tensors=None):
                return {"inputs_embeds": _random_input(batch=1, seq=3)}

        with Gemma4Pipeline(model, _FakeTokenizer(), config=_TEST_CONFIG) as pipeline:
            pipeline.run_prompts(["hello"])
            result = pipeline.defragment(keep_fraction=0.5)

        assert isinstance(result, dict)
        assert len(result) == _N_LAYERS
        for orig, new in result.values():
            assert new == _INTERMEDIATE // 2

    def test_pipeline_reset_clears_stats(self):
        model = _make_model()

        class _FakeTokenizer:
            def __call__(self, text, return_tensors=None):
                return {"inputs_embeds": _random_input(batch=1, seq=3)}

        with Gemma4Pipeline(model, _FakeTokenizer(), config=_TEST_CONFIG) as pipeline:
            pipeline.run_prompts(["hello"])
            pipeline.reset()
            for stats in pipeline.tracker.mlp_stats.values():
                assert stats.total_samples == 0

    def test_pipeline_prune_to_mask(self):
        model = _make_model()

        class _FakeTokenizer:
            def __call__(self, text, return_tensors=None):
                return {"inputs_embeds": _random_input(batch=1, seq=3)}

        with Gemma4Pipeline(model, _FakeTokenizer(), config=_TEST_CONFIG) as pipeline:
            pipeline.run_prompts(["hello"])
            keep_mask = {
                k: torch.ones(_INTERMEDIATE, dtype=torch.bool)
                for k in pipeline.tracker.mlp_stats
            }
            result = pipeline.prune_to_mask(keep_mask, mode="hard")

        # All kept → 0 pruned per layer
        for n_pruned in result.values():
            assert n_pruned == 0
