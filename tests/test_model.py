import torch

from phasematcher.checkpoint import load_model, save_weights, sha256
from phasematcher.models import PhaseMatcher, SinglePhaseModel
from phasematcher.training.losses import prefix_targets
from phasematcher.training.module import TrainingModule


def test_conservation_and_empty_history(cfg, batch):
    network = PhaseMatcher(cfg.model, cfg.decomposition)
    x = batch["mixture"][:, 0]
    references = batch["reference_patterns"][:, :, 0]
    valid = batch["phase_ids"] >= 0
    output = network.spectral_decomposition(x, references, valid)
    assert (output["contributions"] >= 0).all()
    assert (output["remainder"] >= 0).all()
    assert (output["contributions"][~valid] == 0).all()
    torch.testing.assert_close(output["contributions"].sum(1) + output["remainder"], x)
    empty = network.spectral_decomposition(x, references[:, :0], valid[:, :0])
    torch.testing.assert_close(empty["remainder"], x)


def test_prefix_targets_conserve_observation(batch):
    for selected in range(1, 5):
        x, components, residual = prefix_targets(batch, batch["counts"] >= selected, selected)
        torch.testing.assert_close(components.sum(1) + residual, x)
        assert (residual >= 0).all()


def test_stage_objectives_and_gradients(cfg, batch):
    for stage in ("phase", "stop"):
        module = TrainingModule(cfg, stage)
        values = module.compute_losses(batch)
        assert len([k for k in values if k.startswith("decomposition/")]) == 8
        torch.testing.assert_close(
            values["loss"], values["phase"] + values.get("stop", 0) + values["decomposition/loss"]
        )
        values["loss"].backward()
        for group in (module.network.model, module.network.spectral_decomposition):
            gradients = [p.grad for p in group.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            assert sum(float(g.abs().sum()) for g in gradients) > 0
        if stage == "stop":
            assert module.network.model.stop_head.weight.grad.abs().sum() > 0


def test_next_phase_gradient_reaches_decomposition(cfg, batch):
    module = TrainingModule(cfg, "phase")
    module.compute_losses(batch)["phase"].backward()
    assert (
        sum(
            float(p.grad.abs().sum())
            for p in module.network.spectral_decomposition.parameters()
            if p.grad is not None
        )
        > 0
    )


def test_stage_initialization_and_tensor_checkpoint(cfg, library, tmp_path):
    single = SinglePhaseModel(cfg.model)
    phase = PhaseMatcher(cfg.model, cfg.decomposition, with_stop=False)
    phase.initialize_single(single.state_dict())
    torch.testing.assert_close(phase.model.phase_table.weight, single.classifier.weight)
    stop = PhaseMatcher(cfg.model, cfg.decomposition)
    stop.initialize_phase(phase.state_dict())
    file = tmp_path / "last.pt"
    save_weights(file, stop, cfg, "stop", 17, sha256(library.root / "entry.npy"))
    loaded, metadata = load_model(file, library=library)
    assert metadata["global_step"] == 17
    for name, tensor in stop.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], tensor)


def test_plateau_every_four_bad_validations_and_resume(cfg):
    module = TrainingModule(cfg, "phase")
    optimizer = module.configure_optimizers()
    module.step_schedule(0.8, 2000)
    for step in (4000, 6000, 8000):
        module.step_schedule(0.79, step)
    assert optimizer.param_groups[0]["lr"] == 5e-5
    module.step_schedule(0.79, 10000)
    assert all(g["lr"] == 2.5e-5 for g in optimizer.param_groups)
    state = {}
    module.on_save_checkpoint(state)
    resumed = TrainingModule(cfg, "phase")
    resumed.on_load_checkpoint(state)
    resumed_optimizer = resumed.configure_optimizers()
    resumed_optimizer.load_state_dict(optimizer.state_dict())
    resumed.step_schedule(0.1, 10000)
    assert resumed.scheduler.num_bad_epochs == 0
    resumed.step_schedule(0.1, 12000)
    assert resumed.scheduler.num_bad_epochs == 1
