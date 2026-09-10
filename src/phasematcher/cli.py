"""Small explicit command line: check, train, predict, evaluate."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .checkpoint import load_model, sha256
from .config import load_config
from .data import MixtureDataset, SpectrumLibrary
from .inference import Search
from .metrics import SetMetrics


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def check(cfg, verify_hash=False):
    root = Path(cfg.data_root).parents[1]
    manifest = json.loads((root / "assets.json").read_text(encoding="utf-8"))
    selected = [f for f in manifest["files"] if f"/{cfg.dataset}/" in f["path"]]
    if not selected:
        raise ValueError("Dataset is missing from assets.json")
    for item in tqdm(selected, desc="Checking assets"):
        path = root / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Missing or incomplete asset: {path}")
        if verify_hash and sha256(path) != item["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}")
        if "shape" in item:
            array = np.load(path, mmap_mode="r")
            if list(array.shape) != item["shape"] or str(array.dtype) != item["dtype"]:
                raise ValueError(f"Array shape/dtype mismatch: {path}")
    library = SpectrumLibrary(Path(cfg.data_root) / "reference")
    model, _ = load_model(Path(cfg.checkpoint_root) / "last.pt", library=library)
    del model
    splits = {}
    for split in ("train", "val", "test"):
        data = MixtureDataset(cfg.data_root, split, seed=cfg.seed, measurement=cfg.measurement)
        sample = data[0]
        if not torch.isfinite(sample["mixture"]).all():
            raise ValueError(f"Invalid {split} sample")
        splits[split] = len(data)
    return dict(
        dataset=str(cfg.dataset),
        assets=len(selected),
        sha256_verified=verify_hash,
        phases=len(library),
        points=library.points,
        splits=splits,
    )


def evaluate(cfg, args):
    data = MixtureDataset(cfg.data_root, args.split, seed=cfg.seed, measurement=cfg.measurement)
    checkpoint_path = args.checkpoint or Path(cfg.checkpoint_root) / "last.pt"
    model, checkpoint = load_model(checkpoint_path, args.device, data.library)
    if checkpoint["stage"] != "stop":
        raise ValueError("Full-set evaluation requires a STOP-stage checkpoint")
    total = min(args.limit, len(data)) if args.limit is not None else len(data)
    if total < 1 or not 0 <= args.rank < args.world_size:
        raise ValueError("Invalid sample limit or shard rank")
    indices = range(args.rank, total, args.world_size)
    loader = DataLoader(
        Subset(data, indices), batch_size=args.batch_size, num_workers=args.workers, shuffle=False
    )
    search, metrics = Search(model, data.library), SetMetrics(model.max_components)
    output = Path(args.output or Path(cfg.output_root) / f"{args.split}_{args.strategy}")
    output.mkdir(parents=True, exist_ok=True)
    details = output / f"predictions.rank{args.rank}.jsonl"
    with details.open("w", encoding="utf-8") as stream:
        for batch in tqdm(loader, desc=f"{args.strategy} rank {args.rank}"):
            paths = search.predict(
                batch["mixture"], strategy=args.strategy, beam_size=args.beam_size
            )
            metrics.update(paths, batch["phase_ids"].tolist())
            for index, target, candidates in zip(
                batch["index"].tolist(), batch["phase_ids"].tolist(), paths
            ):
                stream.write(
                    json.dumps(
                        dict(
                            index=index,
                            target=[p for p in target if p >= 0],
                            predictions=[asdict(p) for p in candidates],
                        )
                    )
                    + "\n"
                )
    report = dict(
        dataset=str(cfg.dataset),
        split=args.split,
        checkpoint_sha256=sha256(checkpoint_path),
        strategy=args.strategy,
        beam_size=args.beam_size if args.strategy == "beam" else 1,
        rank=args.rank,
        world_size=args.world_size,
        sample_limit=total,
        metric_sums=metrics.sums.tolist(),
        **metrics.result(),
    )
    if args.strategy == "greedy":
        for row in [report["overall"], *report["by_phase_count"].values()]:
            row["exact_set_at_10"] = None
    _write_json(output / f"metrics.rank{args.rank}.json", report)
    return report


def predict(args):
    library = SpectrumLibrary(args.library)
    model, checkpoint = load_model(args.checkpoint, args.device, library)
    if checkpoint["stage"] != "stop":
        raise ValueError("Prediction requires a STOP-stage checkpoint")
    x = np.load(args.input, allow_pickle=False).astype(np.float32)
    if x.ndim == 1:
        x = x[None]
    if x.ndim != 2 or x.shape[-1] != library.points:
        raise ValueError("Input .npy must have shape [L] or [B,L] on the reference grid")
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Input spectra must be finite and nonnegative")
    scale = x.max(axis=1, keepdims=True)
    x = np.divide(x, np.maximum(scale, 1e-8), out=np.zeros_like(x), where=scale > 0)
    search = Search(model, library)
    histories = [args.history for _ in x] if args.history is not None else None
    paths = search.predict(x, strategy=args.strategy, beam_size=args.beam_size, histories=histories)
    report = [
        [
            dict(**asdict(p), entries=[str(library.entries[i]) for i in p.phase_ids])
            for p in candidates
        ]
        for candidates in paths
    ]
    _write_json(args.output, report)
    if args.spectra:
        output = search.decompose(x, [p[0].phase_ids for p in paths])
        arrays = {k: v.cpu().numpy() for k, v in output.items()}
        arrays["contributions"] *= scale[:, None]
        arrays["remainder"] *= scale
        target = Path(args.spectra)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **arrays, axis_two_theta=library.axis_two_theta)
    return report


def main():
    parser = argparse.ArgumentParser(prog="phasematcher")
    parser.add_argument("--threads", type=int, default=4, help="CPU intra-op threads")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("check", help="Validate dataset/checkpoint assets")
    p.add_argument("--config", required=True)
    p.add_argument("--hash", action="store_true")
    p = commands.add_parser("train", help="Train one of the three stages")
    p.add_argument("--config", required=True)
    p.add_argument("--stage", choices=("single", "phase", "stop"), required=True)
    initialization = p.add_mutually_exclusive_group()
    initialization.add_argument("--init")
    initialization.add_argument("--resume")
    p.add_argument("--devices", type=int)
    p.add_argument("--max-steps", type=int)
    for name in ("predict", "evaluate"):
        p = commands.add_parser(name)
        p.add_argument("--checkpoint", required=name == "predict")
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--strategy", choices=("greedy", "beam"), default="greedy")
        p.add_argument("--beam-size", type=int, default=10)
        if name == "evaluate":
            p.add_argument("--config", required=True)
            p.add_argument("--split", choices=("val", "test"), default="test")
            p.add_argument("--batch-size", type=int, default=4)
            p.add_argument("--workers", type=int, default=0)
            p.add_argument("--limit", type=int)
            p.add_argument("--rank", type=int, default=0)
            p.add_argument("--world-size", type=int, default=1)
            p.add_argument("--output")
        else:
            p.add_argument("--library", required=True)
            p.add_argument("--input", required=True)
            p.add_argument("--history", type=int, nargs="*")
            p.add_argument("--output", default="predictions.json")
            p.add_argument(
                "--spectra",
                help="Optional .npz of contributions and residual in input intensity scale",
            )
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.command == "predict":
        result = predict(args)
    else:
        cfg = load_config(args.config)
        if args.command == "check":
            result = check(cfg, args.hash)
        elif args.command == "evaluate":
            result = evaluate(cfg, args)
        else:
            from .training.runner import train

            train(
                cfg,
                args.stage,
                init=args.init,
                resume=args.resume,
                devices=args.devices,
                max_steps=args.max_steps,
            )
            return
    print(json.dumps(result, ensure_ascii=False, indent=2))
