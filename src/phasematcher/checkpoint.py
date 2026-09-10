"""Strict, versioned tensor checkpoints. No historical runtime aliases."""

import hashlib
from pathlib import Path

import torch

from .models import PhaseMatcher, SinglePhaseModel


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("stage") not in {
        "single",
        "phase",
        "stop",
    }:
        raise ValueError("Expected a PhaseMatcher v1 .pt checkpoint")
    return checkpoint


def load_model(path, device="cpu", library=None):
    checkpoint = read_checkpoint(path)
    cfg = checkpoint["model_config"]
    if library is not None:
        if len(library) != cfg["num_phases"]:
            raise ValueError("Checkpoint and reference library sizes differ")
        if library.points != cfg["spectrum_encoder"]["input_points"]:
            raise ValueError("Checkpoint and reference spectrum lengths differ")
        expected = checkpoint.get("library_entry_sha256")
        if expected and sha256(library.root / "entry.npy") != expected:
            raise ValueError("Reference ID order differs from the checkpoint")
    if checkpoint["stage"] == "single":
        model = SinglePhaseModel(cfg)
    else:
        model = PhaseMatcher(
            cfg, checkpoint["decomposition_config"], with_stop=checkpoint["stage"] == "stop"
        )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def save_weights(path, network, cfg, stage, step, library_entry_sha256):
    from omegaconf import OmegaConf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        format_version=1,
        stage=stage,
        dataset=str(cfg.dataset),
        model_config=OmegaConf.to_container(cfg.model, resolve=True),
        decomposition_config=OmegaConf.to_container(cfg.decomposition, resolve=True),
        global_step=int(step),
        library_entry_sha256=library_entry_sha256,
        state_dict={k: v.detach().cpu() for k, v in network.state_dict().items()},
    )
    temporary = path.with_suffix(".partial")
    torch.save(payload, temporary)
    temporary.replace(path)
