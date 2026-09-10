from pathlib import Path

import numpy as np
import pytest
from pytorch_lightning import Trainer

from phasematcher.checkpoint import read_checkpoint
from phasematcher.data import MixtureDataset
from phasematcher.training.module import TrainingModule
from phasematcher.training.runner import TrainingData


def make_mixtures(cfg, library):
    root = Path(cfg.data_root)
    observations = root / "observations"
    observations.mkdir()
    np.save(observations / "patterns.npy", np.repeat(library.patterns[:, None], 5, axis=1))
    np.save(observations / "entry.npy", library.entries)
    np.save(observations / "axis_two_theta.npy", library.axis_two_theta)
    for split in ("train", "val", "test"):
        folder = root / "manifest" / split
        folder.mkdir(parents=True)
        ids = np.full((4, 4), -1, dtype=np.int64)
        weights = np.zeros((4, 4), dtype=np.float32)
        for k in range(1, 5):
            ids[k - 1, :k] = np.arange(k)
            weights[k - 1, :k] = np.arange(k, 0, -1) / sum(range(1, k + 1))
        for key, value in dict(ids=ids, weights=weights, counts=np.arange(1, 5)).items():
            np.save(folder / f"{split}_{key}.npy", value)


def test_deterministic_data_and_pickle(cfg, library):
    import pickle

    make_mixtures(cfg, library)
    dataset = MixtureDataset(cfg.data_root, "test", seed=cfg.seed, measurement=cfg.measurement)
    restored = pickle.loads(pickle.dumps(dataset))
    for index in range(len(dataset)):
        a, b = dataset[index], restored[index]
        for key in a:
            assert np.array_equal(a[key].numpy(), b[key].numpy())
        assert a["mixture"].max() == 1


@pytest.mark.parametrize("stage", ["single", "phase", "stop"])
def test_real_trainer_and_resume(cfg, library, stage, tmp_path):
    make_mixtures(cfg, library)
    cfg.training[stage].batch_size = 2
    cfg.training.eval_batch_size = 2
    data = TrainingData(cfg, stage)
    module = TrainingModule(cfg, stage)
    options = dict(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        val_check_interval=1,
        check_val_every_n_epoch=None,
        limit_val_batches=1,
        default_root_dir=str(tmp_path),
    )
    trainer = Trainer(max_steps=1, **options)
    trainer.fit(module, datamodule=data)
    checkpoint = read_checkpoint(Path(cfg.run_output) / "last.pt")
    assert checkpoint["stage"] == stage and checkpoint["global_step"] == 1
    recovery = tmp_path / "resume.ckpt"
    trainer.save_checkpoint(recovery)
    restored = TrainingModule(cfg, stage)
    resumed = Trainer(max_steps=2, **options)
    resumed.fit(restored, datamodule=data, ckpt_path=str(recovery))
    assert resumed.global_step == 2
    if stage != "single":
        assert restored.scheduler.last_epoch == 2
        assert restored._last_schedule_step == 2
