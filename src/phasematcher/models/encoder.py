import math

import torch
import torch.nn.functional as F
from torch import nn


def _group_count(channels, max_groups=8):
    for groups in range(min(int(max_groups), int(channels)), 0, -1):
        if int(channels) % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(
                int(in_channels),
                int(out_channels),
                kernel_size=int(kernel_size),
                stride=int(stride),
                padding=(int(kernel_size) - 1) // 2,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), int(out_channels)),
            nn.GELU(),
        )

    def forward(self, x):
        return self.layers(x)


class SpectrumTokenEncoder(nn.Module):
    """Convert a fixed-grid PXRD pattern into contextual spectrum tokens."""

    def __init__(self, cfg):
        super().__init__()
        input_points = int(cfg.input_points)
        channel = int(cfg.channel)
        d_model = int(cfg.d_model)
        layers = int(cfg.layers)
        nhead = int(cfg.nhead)
        ff_dim = int(cfg.ff_dim)
        dropout = float(cfg.dropout)

        self.input_points = input_points
        self.stem = nn.Sequential(
            ConvNormAct(1, channel // 2, kernel_size=15, stride=2),
            ConvNormAct(channel // 2, channel, kernel_size=11, stride=2),
            ConvNormAct(channel, channel, kernel_size=7, stride=2),
            ConvNormAct(channel, channel, kernel_size=5, stride=2),
        )
        self.token_projection = nn.Sequential(
            nn.Conv1d(channel, d_model, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(d_model), d_model),
            nn.GELU(),
        )
        max_tokens = math.ceil(input_points / 16)
        self.position = nn.Parameter(torch.randn(1, max_tokens, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, pattern):
        if pattern.ndim != 3 or pattern.shape[1] != 1:
            raise ValueError(f"pattern must have shape [B, 1, L], got {tuple(pattern.shape)}")
        if pattern.shape[-1] != self.input_points:
            raise ValueError(f"Expected L={self.input_points}, got L={pattern.shape[-1]}")

        tokens = self.stem(pattern.float())
        tokens = self.token_projection(tokens).transpose(1, 2).contiguous()
        if tokens.shape[1] > self.position.shape[1]:
            raise RuntimeError("Spectrum token sequence exceeds positional embedding length")
        tokens = self.dropout(tokens + self.position[:, : tokens.shape[1]])
        return self.transformer(tokens)


class AttentionPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(int(d_model), int(d_model)),
            nn.Tanh(),
            nn.Linear(int(d_model), 1),
        )

    def forward(self, tokens):
        weights = torch.softmax(self.score(tokens).squeeze(-1), dim=1)
        return torch.sum(tokens * weights.unsqueeze(-1), dim=1)


class SpectrumEncoder(nn.Module):
    """Shared single-phase and mixture encoder for fixed-grid full spectra."""

    def __init__(self, cfg):
        super().__init__()
        self.token_encoder = SpectrumTokenEncoder(cfg)
        self.pool = AttentionPool(int(cfg.d_model))

    def forward_tokens(self, pattern):
        return self.token_encoder(pattern)

    def forward(self, pattern):
        return F.normalize(self.pool(self.forward_tokens(pattern)), dim=-1)
