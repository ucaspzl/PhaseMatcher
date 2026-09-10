"""Two dataset presets sharing one architecture and loss definition."""

from pathlib import Path

from omegaconf import OmegaConf


def load_config(path):
    path = Path(path).resolve()
    cfg = OmegaConf.load(path)
    base = cfg.pop("base", None)
    if base:
        cfg = OmegaConf.merge(OmegaConf.load(path.parent / base), cfg)
    root = path.parent.parent
    for key in ("data_root", "checkpoint_root", "output_root"):
        cfg[key] = str((root / str(cfg[key])).resolve())
    OmegaConf.resolve(cfg)
    if cfg.decomposition.max_shift_bins <= 0 or cfg.decomposition.max_strain <= 0:
        raise ValueError("Physical bounds must be positive")
    if any(value < 0 for key, value in cfg.loss.items() if key.startswith("lambda_")):
        raise ValueError("Loss weights must be nonnegative")
    return cfg
