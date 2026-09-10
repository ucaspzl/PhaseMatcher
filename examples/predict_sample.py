"""Predict one held-out mixture using real dataset assets and final weights.

Run from the repository root after installation. Labels are not passed to inference.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from phasematcher.cli import predict
from phasematcher.config import load_config
from phasematcher.data import MixtureDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phasemix.yaml")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--strategy", choices=("greedy", "beam"), default="greedy")
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("outputs/example"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = load_config(args.config)
    checkpoint = Path(cfg.checkpoint_root) / "last.pt"
    if not checkpoint.is_file():
        parser.error(f"Missing {checkpoint}; see docs/resources.md")
    data = MixtureDataset(cfg.data_root, "test", seed=cfg.seed, measurement=cfg.measurement)
    if not 0 <= args.index < len(data):
        parser.error(f"--index must be in 0..{len(data) - 1}")
    observation = data[args.index]["mixture"].numpy().reshape(-1)
    args.output.mkdir(parents=True, exist_ok=True)
    input_path = args.output / "input.npy"
    np.save(input_path, observation)
    report = predict(
        SimpleNamespace(
            checkpoint=str(checkpoint),
            library=str(Path(cfg.data_root) / "reference"),
            input=str(input_path),
            device=args.device,
            strategy=args.strategy,
            beam_size=args.beam_size,
            history=None,
            output=str(args.output / "predictions.json"),
            spectra=str(args.output / "decomposition.npz"),
        )
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved example to {args.output.resolve()}")


if __name__ == "__main__":
    main()
