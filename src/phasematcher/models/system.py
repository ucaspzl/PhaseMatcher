"""The two trainable systems: single-phase initialization and PhaseMatcher."""

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F

from .decomposition import PhysicsGuidedSpectralDecomposition
from .encoder import SpectrumEncoder
from .heads import PhaseRetriever, PhaseStopRetriever


class ClassificationHead(nn.Module):
    def __init__(self, num_phases, width):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_phases, width))
        self.logit_scale = nn.Parameter(torch.tensor(10.0).log())
        nn.init.normal_(self.weight, std=width**-0.5)

    def forward(self, features):
        return (
            F.normalize(features, dim=-1)
            @ F.normalize(self.weight, dim=-1).T
            * self.logit_scale.exp().clamp(max=100)
        )


class SinglePhaseModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        cfg = OmegaConf.create(model_config)
        self.encoder = SpectrumEncoder(cfg.spectrum_encoder)
        self.classifier = ClassificationHead(int(cfg.num_phases), int(cfg.d_model))

    def forward(self, patterns):
        return self.classifier(self.encoder(patterns))


class PhaseMatcher(nn.Module):
    """Phase/STOP prediction and reference-conditioned spectral decomposition.

    State-dict names are stable; checkpoints contain no training-framework objects.
    Phase IDs are 0..N-1, STOP is output column N, BOS uses the private sentinel N+1.
    """

    def __init__(self, model_config, decomposition_config, *, with_stop=True):
        super().__init__()
        cfg = OmegaConf.create(model_config)
        self.num_phases = int(cfg.num_phases)
        self.max_components = int(cfg.max_components)
        self.bos_id = self.num_phases + 1
        self.with_stop = bool(with_stop)
        self.model = (PhaseStopRetriever if with_stop else PhaseRetriever)(cfg)
        self.spectral_decomposition = PhysicsGuidedSpectralDecomposition(**decomposition_config)

    def initialize_single(self, state):
        prefix = "encoder.token_encoder."
        encoder = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        self.model.spectrum_encoder.load_state_dict(encoder, strict=True)
        with torch.no_grad():
            self.model.phase_table.weight.copy_(state["classifier.weight"])

    def initialize_phase(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected or any(
            not k.startswith(("model.stop_decoder.", "model.stop_head.")) for k in missing
        ):
            raise ValueError(f"Invalid phase initialization: missing={missing}, extra={unexpected}")
