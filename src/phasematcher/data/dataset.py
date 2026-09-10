"""Memory-mapped reference banks and the current noisy-mixture protocol."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .observation import ExperimentalObservationModel


class SpectrumLibrary:
    def __init__(self, root, intensity_scale=100.0):
        self.root = Path(root)
        self.intensity_scale = float(intensity_scale)
        self.patterns = np.load(self.root / "patterns.npy", mmap_mode="r")
        self.entries = np.load(self.root / "entry.npy", mmap_mode="r")
        self.axis_two_theta = np.load(self.root / "axis_two_theta.npy")
        if self.patterns.ndim not in (2, 3):
            raise ValueError("patterns must be [N,L] or [N,R,L]")
        if self.entries.shape != (len(self),) or self.axis_two_theta.shape != (self.points,):
            raise ValueError("Reference entries/grid do not align with patterns")
        if self.intensity_scale <= 0:
            raise ValueError("intensity_scale must be positive")

    def __len__(self):
        return self.patterns.shape[0]

    @property
    def points(self):
        return self.patterns.shape[-1]

    def __getstate__(self):
        return {"root": self.root, "intensity_scale": self.intensity_scale}

    def __setstate__(self, state):
        self.__init__(**state)

    def phase_patterns(self, phase_ids, realization_ids=None):
        ids = np.asarray(phase_ids, dtype=np.int64)
        if np.any(ids < 0) or np.any(ids >= len(self)):
            raise ValueError("Phase IDs outside the reference library")
        if self.patterns.ndim == 3:
            if realization_ids is None:
                raise ValueError("Observation banks require realization IDs")
            values = self.patterns[ids, np.asarray(realization_ids, dtype=np.int64)]
        else:
            values = self.patterns[ids]
        return np.clip(np.asarray(values, dtype=np.float32) / self.intensity_scale, 0, None)

    def phase_pattern(self, phase_id):
        return self.phase_patterns([phase_id])[0]


class SinglePhaseDataset(Dataset):
    def __init__(self, library):
        self.library = library

    def __len__(self):
        return len(self.library)

    def __getitem__(self, index):
        return {
            "pattern": torch.from_numpy(self.library.phase_pattern(index)[None]),
            "phase_id": torch.tensor(index, dtype=torch.long),
        }


class MixtureDataset(Dataset):
    """Train uses random realizations 0..2; validation/test use fixed 3/4.

    RandomState and seed offsets preserve the published observation realizations.
    No scalar subtraction, distractors, random order, or alternative noise modes.
    """

    def __init__(self, root, split, *, seed, measurement=None, max_components=4):
        self.root, self.split = Path(root), split
        self.seed, self.max_components = int(seed), int(max_components)
        self.measurement_config = dict(measurement or {})
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self.library = SpectrumLibrary(self.root / "reference")
        self.observations = SpectrumLibrary(self.root / "observations")
        if not np.array_equal(self.library.entries, self.observations.entries):
            raise ValueError("Observation/reference identity order differs")
        if not np.array_equal(self.library.axis_two_theta, self.observations.axis_two_theta):
            raise ValueError("Observation/reference grids differ")
        self.measurement = ExperimentalObservationModel(
            self.observations.axis_two_theta, self.measurement_config
        )
        folder = self.root / "manifest" / split
        self.ids = np.load(folder / f"{split}_ids.npy", mmap_mode="r")
        self.weights = np.load(folder / f"{split}_weights.npy", mmap_mode="r")
        self.counts = np.load(folder / f"{split}_counts.npy", mmap_mode="r")
        if self.ids.shape != self.weights.shape or self.counts.shape != (len(self.ids),):
            raise ValueError("Manifest array shapes differ")

    def __getstate__(self):
        return dict(
            root=self.root,
            split=self.split,
            seed=self.seed,
            measurement=self.measurement_config,
            max_components=self.max_components,
        )

    def __setstate__(self, state):
        self.__init__(**state)

    def __len__(self):
        return len(self.counts)

    def __getitem__(self, index):
        index = int(index)
        count = int(self.counts[index])
        if not 1 <= count <= self.max_components:
            raise ValueError(f"Invalid phase count at row {index}")
        ids = np.asarray(self.ids[index, :count], dtype=np.int64)
        weights = np.asarray(self.weights[index, :count], dtype=np.float32)
        if len(set(ids.tolist())) != count:
            raise ValueError(f"Duplicate phase IDs at row {index}")
        offset = {"train": 0, "val": 1, "test": 2}[self.split]
        choices = [0, 1, 2] if self.split == "train" else [3 if self.split == "val" else 4]
        rng = (
            np.random
            if self.split == "train"
            else np.random.RandomState(self.seed + offset + index)
        )
        realizations = rng.choice(choices, size=count, replace=True).astype(np.int64)
        # Reset the deterministic measurement RNG exactly as the original loader does.
        rng = (
            np.random
            if self.split == "train"
            else np.random.RandomState(self.seed + offset + index)
        )
        mixture, _, _, _, common = self.measurement.build(
            self.observations.phase_patterns(ids, realizations),
            weights,
            self.max_components,
            rng,
            return_common_scale=True,
        )
        padded_ids = np.full(self.max_components, -1, dtype=np.int64)
        padded_ids[:count] = ids
        padded_weights = np.zeros(self.max_components, dtype=np.float32)
        padded_weights[:count] = weights
        targets = np.full(self.max_components + 1, -1, dtype=np.int64)
        targets[:count], targets[count] = ids, len(self.library)
        references = np.zeros((self.max_components, 1, self.library.points), dtype=np.float32)
        references[:count, 0] = self.library.phase_patterns(ids)
        return dict(
            index=torch.tensor(index),
            mixture=torch.from_numpy(mixture[None].copy()),
            counts=torch.tensor(count),
            phase_ids=torch.from_numpy(padded_ids),
            phase_weights=torch.from_numpy(padded_weights),
            targets=torch.from_numpy(targets),
            reference_patterns=torch.from_numpy(references),
            component_contributions=torch.from_numpy(
                common["component_contributions"][:, None].copy()
            ),
            residual_patterns_common=torch.from_numpy(
                common["residual_patterns_common"][:, None].copy()
            ),
            observation_scale=torch.from_numpy(common["observation_scale"].copy()),
        )
