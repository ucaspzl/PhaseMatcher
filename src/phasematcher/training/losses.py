"""Seven-term decomposition loss and conservative true-prefix supervision."""

import torch
from torch.nn import functional as F


def normalize_query(pattern):
    maximum = pattern.amax(dim=-1, keepdim=True)
    return torch.where(maximum > 0, pattern / maximum.clamp_min(1e-8), 0.0)


def cosine_loss(prediction, target, valid=None):
    active = target.norm(dim=-1) > 1e-8
    if valid is not None:
        active = active & valid.bool()
    if not active.any():
        return prediction.new_zeros(())
    return (1 - F.cosine_similarity(prediction[active], target[active], dim=-1, eps=1e-8)).mean()


def weighted_mean(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def prefix_targets(batch, active, selected):
    observation = batch["mixture"][active, 0]
    components = batch["component_contributions"][active, :selected, 0]
    total = components.sum(dim=1)
    removed = torch.minimum(observation, total)
    shares = components / total[:, None].clamp_min(torch.finfo(components.dtype).tiny)
    shares = torch.where((total > 0)[:, None], shares, 0.0)
    return observation, shares * removed[:, None], observation - removed


def decomposition_loss(output, observation, targets, remainder_target, valid, cfg, physical):
    contributions = output["contributions"]
    dtype = contributions.dtype
    valid_weights = valid.to(dtype)
    normalized = targets / targets.amax(dim=-1, keepdim=True).clamp_min(1e-8)
    point_weights = (cfg.background_point_weight + normalized.clamp_min(0).sqrt()) * valid_weights[
        :, :, None
    ]
    terms = {
        "component": weighted_mean(
            F.smooth_l1_loss(contributions, targets, beta=cfg.smooth_l1_beta, reduction="none"),
            point_weights,
        )
    }
    terms["component_shape"] = cosine_loss(contributions, targets, valid)
    removed, removed_target = contributions.sum(dim=1), targets.sum(dim=1)
    scale = removed_target / removed_target.amax(dim=-1, keepdim=True).clamp_min(1e-8)
    weights = cfg.background_point_weight + scale.clamp_min(0).sqrt()
    terms["removal"] = weighted_mean(
        F.smooth_l1_loss(removed, removed_target, beta=cfg.smooth_l1_beta, reduction="none"),
        weights,
    )
    sources = torch.cat([targets, remainder_target[:, None]], dim=1)
    active = observation > 1e-8
    fractions = torch.where(active[:, None], sources / observation[:, None].clamp_min(1e-8), 0.0)
    log_prob = F.log_softmax(
        torch.cat([output["phase_logits"], output["remainder_logits"][:, None]], dim=1), dim=1
    )
    positive = fractions > 0
    kl = torch.where(
        positive,
        fractions * (fractions.clamp_min(1e-8).log() - torch.where(positive, log_prob, 0.0)),
        0.0,
    ).sum(dim=1)
    observed_scale = observation / observation.amax(dim=-1, keepdim=True).clamp_min(1e-8)
    allocation_weights = (
        cfg.background_point_weight
        + observed_scale.clamp_min(0).sqrt()
        + cfg.removal_focus_weight * scale.clamp_min(0).sqrt()
    ) * active
    terms["allocation"] = weighted_mean(kl, allocation_weights)
    terms["removal_area"] = (
        (removed.sum(dim=-1) - removed_target.sum(dim=-1)).abs()
        / removed_target.sum(dim=-1).clamp_min(cfg.minimum_target_area)
    ).mean()
    terms["query_shape"] = cosine_loss(
        normalize_query(output["remainder"]), normalize_query(remainder_target)
    )
    parameters = output["alignment_parameters"]
    parameter_scale = parameters.new_tensor([physical.max_shift_bins, physical.max_strain])
    shift_strain = (parameters[..., :2] / parameter_scale).square().mean(dim=-1)
    blur = parameters[..., 3] + 2 * parameters[..., 4]
    terms["alignment"] = (
        (shift_strain + blur.square()) * valid_weights
    ).sum() / valid_weights.sum().clamp_min(1.0)
    terms["loss"] = sum(float(cfg[f"{key}_weight"]) * value for key, value in terms.items())
    return terms
