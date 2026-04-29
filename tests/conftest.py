"""
Shared test fixtures and tiny transformer-like toy model used across all tests.
"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Minimal MLP with an explicit activation layer (named 'act')."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.act = nn.Linear(d_hidden, d_hidden)   # the "activation" sub-module we hook
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.act(x)          # tracked by ActivationTracker
        x = torch.relu(x)
        return self.fc2(x)


class TinyTransformerLayer(nn.Module):
    """One transformer-like block: self-attention + FFN."""

    def __init__(self, d_model: int = 16, n_heads: int = 2, d_ff: int = 32) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = FeedForward(d_model, d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        return self.ffn(attn_out)


class TinyModel(nn.Module):
    """Two-layer tiny transformer for testing."""

    def __init__(self, d_model: int = 16, n_heads: int = 2, d_ff: int = 32) -> None:
        super().__init__()
        self.layer0 = TinyTransformerLayer(d_model, n_heads, d_ff)
        self.layer1 = TinyTransformerLayer(d_model, n_heads, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)
        return self.layer1(x)


def make_model(d_model: int = 16, n_heads: int = 2, d_ff: int = 32) -> TinyModel:
    torch.manual_seed(0)
    return TinyModel(d_model, n_heads, d_ff).eval()


def random_input(batch: int = 2, seq: int = 4, d_model: int = 16) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(batch, seq, d_model)
