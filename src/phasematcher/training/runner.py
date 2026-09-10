"""Portable training entry; no cluster paths, implicit checkpoint lookup or jobs."""

from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, DistributedSampler, Subset

from ..checkpoint import read_checkpoint, sha256
from ..data import MixtureDataset, SinglePhaseDataset, SpectrumLibrary
from .module import TrainingModule


class TrainingData(LightningDataModule):
    def __init__(self, cfg, stage):
        super().__init__()
        self.cfg, self.stage = cfg, stage
        self.library = SpectrumLibrary(Path(cfg.data_root) / "reference")
        if len(self.library) != cfg.model.num_phases:
            raise ValueError("Reference library and model phase counts differ")

    def setup(self, stage=None):
        if self.stage == "single":
            # Classification initialization evaluates the same reference bank.
            self.train_data = self.val_data = SinglePhaseDataset(self.library)
        else:
            options = dict(
                seed=self.cfg.seed,
                measurement=self.cfg.measurement,
                max_components=self.cfg.model.max_components,
            )
            self.train_data = MixtureDataset(self.cfg.data_root, "train", **options)
            self.val_data = MixtureDataset(self.cfg.data_root, "val", **options)

    def _loader(self, dataset, training):
        workers = int(self.cfg.training.workers)
        world, rank = self.trainer.world_size, self.trainer.global_rank
        sampler = None
        if world > 1:
            if training:
                sampler = DistributedSampler(
                    dataset, num_replicas=world, rank=rank, seed=int(self.cfg.seed), shuffle=True
                )
            else:
                # No duplicate padding samples in distributed validation.
                dataset = Subset(dataset, range(rank, len(dataset), world))
        extra = (
            dict(persistent_workers=True, prefetch_factor=2, multiprocessing_context="spawn")
            if workers
            else {}
        )
        return DataLoader(
            dataset,
            batch_size=int(
                self.cfg.training[self.stage].batch_size
                if training
                else self.cfg.training.eval_batch_size
            ),
            shuffle=training and sampler is None,
            sampler=sampler,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            **extra,
        )

    def train_dataloader(self):
        return self._loader(self.train_data, True)

    def val_dataloader(self):
        return self._loader(self.val_data, False)


def train(cfg, stage, *, init=None, resume=None, devices=None, max_steps=None):
    if init and resume:
        raise ValueError(
            "Use --init for stage initialization OR --resume for interruption recovery"
        )
    seed_everything(int(cfg.seed), workers=True)
    torch.set_float32_matmul_precision("high")
    settings = cfg.training[stage]
    if max_steps is not None:
        settings.max_steps = max_steps
    # Ranks launched by Lightning receive the resolved output path through env.
    import os

    output = os.environ.get("PHASEMATCHER_RUN_OUTPUT")
    owns_output = output is None
    if output is None:
        output = str(
            Path(cfg.output_root) / (stage + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        )
        os.environ["PHASEMATCHER_RUN_OUTPUT"] = output
    cfg.run_output = output
    data = TrainingData(cfg, stage)
    module = TrainingModule(cfg, stage)
    if not resume and stage != "single":
        init = init or Path(cfg.checkpoint_root) / ("single.pt" if stage == "phase" else "phase.pt")
        checkpoint = read_checkpoint(init)
        if checkpoint["stage"] != ("single" if stage == "phase" else "phase"):
            raise ValueError("Initialization must come from the preceding stage")
        if checkpoint["library_entry_sha256"] != sha256(data.library.root / "entry.npy"):
            raise ValueError("Initialization checkpoint uses another reference ordering")
        if stage == "phase":
            module.network.initialize_single(checkpoint["state_dict"])
        else:
            module.network.initialize_phase(checkpoint["state_dict"])
    Path(output).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, Path(output) / "config.yaml")
    count = int(devices if devices is not None else cfg.training.devices)
    gpu = torch.cuda.is_available()
    if count > 1 and not gpu:
        raise ValueError("Multi-device training requires CUDA GPUs")
    validation_steps = settings.validation_steps
    callbacks = [
        ModelCheckpoint(
            dirpath=str(Path(output) / "checkpoints"),
            filename="step{step}",
            monitor="val/accuracy" if stage == "single" else "val/exact_set_at_1",
            mode="max",
            save_top_k=1,
            save_last=True,
            save_on_train_epoch_end=False,
        )
    ]
    trainer = Trainer(
        accelerator="gpu" if gpu else "cpu",
        devices=count,
        strategy="ddp_find_unused_parameters_true" if count > 1 else "auto",
        precision=str(cfg.training.precision) if gpu else "32-true",
        max_steps=int(settings.max_steps),
        max_epochs=int(settings.max_epochs),
        accumulate_grad_batches=int(settings.accumulate),
        gradient_clip_val=float(cfg.training.gradient_clip),
        num_sanity_val_steps=0,
        val_check_interval=int(validation_steps * settings.accumulate) if validation_steps else 1.0,
        check_val_every_n_epoch=None if validation_steps else int(settings.validation_epochs),
        use_distributed_sampler=False,
        callbacks=callbacks,
        logger=CSVLogger(output, name="logs"),
        log_every_n_steps=25,
    )
    try:
        trainer.fit(module, datamodule=data, ckpt_path=str(resume) if resume else None)
        # Retain a final resumable state even when stopping between validations.
        trainer.save_checkpoint(str(Path(output) / "checkpoints" / "last.ckpt"))
    finally:
        if owns_output:
            os.environ.pop("PHASEMATCHER_RUN_OUTPUT", None)
