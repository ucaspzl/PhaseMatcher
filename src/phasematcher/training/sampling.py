from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def sample_candidate_ids(
    targets: torch.Tensor,
    *,
    num_phases: int,
    stop_id: int,
    pad_id: int,
    negative_count: int,
    rng: np.random.Generator,
    device: torch.device,
    mode: str = "hard_mixed",
    phase_embeddings: torch.Tensor | None = None,
    hard_fraction: float = 0.5,
    hard_pool_size: int = 32768,
) -> torch.Tensor:
    """Return a sorted shared candidate set for sampled or full-library training."""
    if mode == "all":
        return torch.arange(num_phases, dtype=torch.long, device=device)

    positives = (
        targets[(targets != stop_id) & (targets != pad_id)].detach().cpu().numpy().astype(np.int64)
    )
    positives = np.unique(positives)
    negative_count = min(int(negative_count), int(num_phases))
    if mode == "random" or phase_embeddings is None:
        negatives = rng.choice(num_phases, size=negative_count, replace=False)
    elif mode == "hard_mixed":
        hard_count = min(int(round(negative_count * float(hard_fraction))), negative_count)
        random_count = negative_count - hard_count
        pool_size = min(max(int(hard_pool_size), hard_count), int(num_phases))
        pool = rng.choice(num_phases, size=pool_size, replace=False)
        if positives.size and hard_count:
            with torch.no_grad():
                pool_ids = torch.as_tensor(pool, dtype=torch.long, device=device)
                positive_ids = torch.as_tensor(positives, dtype=torch.long, device=device)
                pool_emb = F.normalize(phase_embeddings.index_select(0, pool_ids).detach(), dim=1)
                positive_emb = F.normalize(
                    phase_embeddings.index_select(0, positive_ids).detach(), dim=1
                )
                hardness = positive_emb @ pool_emb.transpose(0, 1)
                hard_ids = (
                    pool_ids.index_select(0, hardness.amax(dim=0).topk(hard_count).indices)
                    .cpu()
                    .numpy()
                )
        else:
            hard_ids = np.empty((0,), dtype=np.int64)
        random_ids = rng.choice(num_phases, size=random_count, replace=False)
        negatives = np.concatenate([hard_ids, random_ids])
    else:
        raise ValueError(f"Unsupported candidate sampling mode: {mode}")
    return torch.as_tensor(
        np.unique(np.concatenate([positives, negatives])), dtype=torch.long, device=device
    )
