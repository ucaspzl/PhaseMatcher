"""Three stages, one explicit objective and one checkpointed scheduling policy."""

import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule
from torch.nn import functional as F

from ..checkpoint import save_weights, sha256
from ..inference import Search
from ..metrics import SetMetrics
from ..models import PhaseMatcher, SinglePhaseModel
from .losses import decomposition_loss, normalize_query, prefix_targets
from .sampling import sample_candidate_ids


class TrainingModule(LightningModule):
    def __init__(self, cfg, stage):
        super().__init__()
        if stage not in {"single", "phase", "stop"}:
            raise ValueError("stage must be single, phase or stop")
        self.cfg, self.stage = cfg, stage
        self.save_hyperparameters(
            {"config": OmegaConf.to_container(cfg, resolve=True), "stage": stage}
        )
        self.network = (
            SinglePhaseModel(cfg.model)
            if stage == "single"
            else PhaseMatcher(cfg.model, cfg.decomposition, with_stop=stage == "stop")
        )
        self.rng = np.random.default_rng(int(cfg.seed))
        self.scheduler = None
        self._loaded_scheduler = None
        self._last_schedule_step = -1

    def compute_losses(self, batch):
        if self.stage == "single":
            loss = F.cross_entropy(
                self.network(batch["pattern"]),
                batch["phase_id"],
                label_smoothing=float(self.cfg.loss.label_smoothing),
            )
            return {"loss": loss}
        network, model = self.network, self.network.model
        candidates = sample_candidate_ids(
            batch["targets"],
            num_phases=network.num_phases,
            stop_id=network.num_phases,
            pad_id=-1,
            rng=self.rng,
            device=batch["mixture"].device,
            phase_embeddings=model.phase_table.weight,
            **OmegaConf.to_container(self.cfg.sampling),
        )
        embeddings = model.candidate_embeddings(candidates)
        observation_tokens = model.encode_patterns(batch["mixture"]) if network.with_stop else None
        counts, targets = batch["counts"], batch["targets"]
        phase_logits, phase_targets, stop_logits = [], [], []
        terms, prefix_samples = {}, 0
        for selected in range(network.max_components + 1):
            active = counts >= selected
            size = int(active.sum())
            if not size:
                continue
            if selected:
                observation, contributions, remainder = prefix_targets(batch, active, selected)
                valid = batch["phase_ids"][active, :selected] >= 0
                output = network.spectral_decomposition(
                    batch["mixture"][active, 0],
                    batch["reference_patterns"][active, :selected, 0],
                    valid,
                )
                values = decomposition_loss(
                    output,
                    observation,
                    contributions,
                    remainder,
                    valid,
                    self.cfg.loss.decomposition,
                    self.cfg.decomposition,
                )
                for name, value in values.items():
                    terms[name] = terms.get(name, 0.0) + size * value
                prefix_samples += size
                query = output["remainder"][:, None]
            else:
                query = batch["mixture"][active]
            next_mask = counts[active] > selected
            step_targets = targets[active]
            if network.with_stop:
                score_mask = torch.ones_like(next_mask)
            else:
                score_mask = next_mask
                if not next_mask.any():
                    continue
            history = torch.full(
                (int(score_mask.sum()), selected + 1),
                network.bos_id,
                dtype=torch.long,
                device=query.device,
            )
            if selected:
                history[:, 1:] = step_targets[score_mask, :selected]
            tokens = model.encode_patterns(normalize_query(query[score_mask]))
            scores = model.phase_logits(tokens, history, embeddings, network.bos_id)
            if network.with_stop:
                stop = model.stop_logit(observation_tokens[active], history, network.bos_id)
                scores = torch.cat([scores, stop[:, None]], dim=1)
                if (~next_mask).any():
                    stop_logits.append(scores[~next_mask])
                scores = scores[next_mask]
            if next_mask.any():
                phase_logits.append(scores)
                phase_targets.append(
                    torch.searchsorted(candidates, step_targets[next_mask, selected])
                )
        phase = F.cross_entropy(torch.cat(phase_logits), torch.cat(phase_targets))
        values = {
            "phase": phase,
            **{f"decomposition/{k}": v / prefix_samples for k, v in terms.items()},
        }
        stop = phase.new_zeros(())
        if network.with_stop:
            logits = torch.cat(stop_logits)
            stop = F.cross_entropy(
                logits,
                torch.full((len(logits),), len(candidates), dtype=torch.long, device=logits.device),
            )
            values["stop"] = stop
        weights = self.cfg.loss
        values["loss"] = (
            weights.lambda_phase * phase
            + weights.lambda_stop * stop
            + weights.lambda_decomposition * values["decomposition/loss"]
        )
        return values

    def training_step(self, batch, batch_idx):
        values = self.compute_losses(batch)
        size = len(batch["phase_id"] if self.stage == "single" else batch["counts"])
        for name, value in values.items():
            self.log(
                f"train/{name}",
                value,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=size,
                prog_bar=name == "loss",
            )
        return values["loss"]

    def on_validation_epoch_start(self):
        self._metrics = SetMetrics(int(self.cfg.model.max_components))
        self._single_counts = torch.zeros(2, device=self.device, dtype=torch.float64)

    def validation_step(self, batch, batch_idx):
        if self.stage == "single":
            predicted = self.network(batch["pattern"]).argmax(dim=-1)
            self._single_counts += torch.stack(
                [predicted.new_tensor(len(predicted)), (predicted == batch["phase_id"]).sum()]
            )
            return
        search = Search(self.network, self.trainer.datamodule.library)
        predictions = search.predict(
            batch["mixture"], counts=batch["counts"] if self.stage == "phase" else None
        )
        self._metrics.update(predictions, batch["phase_ids"].cpu().tolist())

    def on_validation_epoch_end(self):
        sums = (
            self._single_counts
            if self.stage == "single"
            else torch.as_tensor(self._metrics.sums, device=self.device)
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(sums)
        if self.stage == "single":
            self.log("val/accuracy", sums[1] / sums[0].clamp_min(1), sync_dist=False)
            return
        for k, row in enumerate(sums):
            if row[0] > 0:
                prefix = "val/" if k == 0 else f"val/{k}phase/"
                for name, value in zip(SetMetrics.names, row[1:]):
                    self.log(prefix + name, value / row[0], sync_dist=False)
        if not self.trainer.sanity_checking:
            self.step_schedule(float(sums[0, 1] / sums[0, 0].clamp_min(1)), int(self.global_step))

    def on_train_epoch_end(self):
        if self.stage == "single":
            self.step_schedule(
                float(self.trainer.callback_metrics["train/loss_epoch"]), int(self.global_step)
            )

    def step_schedule(self, metric, step):
        if step > self._last_schedule_step:
            self.scheduler.step(metric)
            self._last_schedule_step = step

    def configure_optimizers(self):
        cfg = self.cfg.training[self.stage]
        if self.stage == "single":
            parameters = self.network.parameters()
        else:
            parameters = [
                {"params": self.network.model.parameters(), "name": "prediction"},
                {
                    "params": self.network.spectral_decomposition.parameters(),
                    "name": "decomposition",
                },
            ]
        optimizer = torch.optim.AdamW(
            parameters, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay)
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min" if self.stage == "single" else "max",
            factor=float(cfg.factor),
            patience=int(cfg.patience),
            threshold=1e-4,
            threshold_mode="rel",
            min_lr=float(cfg.min_lr),
        )
        if self._loaded_scheduler is not None:
            self.scheduler.load_state_dict(self._loaded_scheduler)
        # Not registered with Lightning: only the explicit hooks above step it.
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        checkpoint["phasematcher_training"] = dict(
            scheduler=self.scheduler.state_dict(),
            last_step=self._last_schedule_step,
            candidate_rng=self.rng.bit_generator.state,
        )

    def on_load_checkpoint(self, checkpoint):
        state = checkpoint["phasematcher_training"]
        self._loaded_scheduler = state["scheduler"]
        if self.scheduler is not None:
            self.scheduler.load_state_dict(self._loaded_scheduler)
        self._last_schedule_step = state["last_step"]
        self.rng.bit_generator.state = state["candidate_rng"]

    def on_train_end(self):
        if self.trainer.is_global_zero:
            save_weights(
                self.cfg.run_output + "/last.pt",
                self.network,
                self.cfg,
                self.stage,
                self.global_step,
                sha256(self.trainer.datamodule.library.root / "entry.npy"),
            )
