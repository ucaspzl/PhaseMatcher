from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from phasematcher.config import load_config
from phasematcher.data import SpectrumLibrary


@pytest.fixture
def cfg(tmp_path):
    config = load_config(Path(__file__).parents[1] / "configs/rruff.yaml")
    config.model = OmegaConf.create(
        dict(
            num_phases=6,
            max_components=4,
            d_model=16,
            decoder_layers=1,
            stop_decoder_layers=1,
            nhead=2,
            ff_dim=32,
            dropout=0.0,
            temperature=0.07,
            spectrum_encoder=dict(
                input_points=64, channel=16, d_model=16, layers=1, nhead=2, ff_dim=32, dropout=0.0
            ),
        )
    )
    config.decomposition.base_channels = 8
    config.decomposition.levels = 2
    config.decomposition.attention_heads = 2
    config.decomposition.attention_layers = 1
    config.decomposition.alignment_regions = 4
    config.decomposition.max_shift_bins = 2
    config.data_root = str(tmp_path)
    config.training.workers = 0
    config.run_output = str(tmp_path / "output")
    torch.set_num_threads(2)
    torch.manual_seed(7)
    return config


@pytest.fixture
def library(cfg):
    root = Path(cfg.data_root) / "reference"
    root.mkdir()
    rng = np.random.default_rng(19)
    patterns = rng.random((6, 64), dtype=np.float32) * 100
    np.save(root / "patterns.npy", patterns)
    np.save(root / "entry.npy", np.array([f"entry-{i}" for i in range(6)]))
    np.save(root / "axis_two_theta.npy", np.linspace(10, 80, 64, dtype=np.float32))
    return SpectrumLibrary(root)


@pytest.fixture
def batch(cfg):
    counts = torch.tensor([1, 2, 3, 4])
    components = torch.rand(4, 4, 1, 64)
    ids = torch.full((4, 4), -1, dtype=torch.long)
    targets = torch.full((4, 5), -1, dtype=torch.long)
    for row, count in enumerate(counts):
        ids[row, :count] = torch.arange(count)
        targets[row, :count] = ids[row, :count]
        targets[row, count] = cfg.model.num_phases
        components[row, count:] = 0
    observation = components.sum(dim=1) + 0.01
    scale = observation.amax(dim=-1, keepdim=True)
    return dict(
        mixture=observation / scale,
        counts=counts,
        targets=targets,
        phase_ids=ids,
        component_contributions=components / scale[:, None],
        reference_patterns=components / components.amax(dim=-1, keepdim=True).clamp_min(1e-8),
    )
