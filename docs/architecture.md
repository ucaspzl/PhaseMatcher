# Architecture

[Home](../README.md) · [中文](architecture.zh-CN.md) · [Training](training.md)

## Prediction loop

1. Start from the original observation and an empty phase history.
2. The Phase Head predicts a library entry from the residual query and history. The STOP Head uses the original observation and history to judge completion.
3. After selecting a phase, decompose the original observation using **all** selected reference spectra.
4. Max-normalize the reconstructed residual for the next query. Continue until STOP or the four-phase limit.

The method recomputes the decomposition from the original observation; it does not repeatedly subtract from the previous residual. Each Beam path owns its history and query.

## Components

| File | Responsibility |
|---|---|
| [encoder.py](../src/phasematcher/models/encoder.py) | Strided convolutions and Transformer spectrum encoding |
| [heads.py](../src/phasematcher/models/heads.py) | Phase identity/history representation, six-layer Phase and two-layer STOP decoders |
| [decomposition.py](../src/phasematcher/models/decomposition.py) | Multiscale features, local correlation, deformation and intensity allocation |
| [system.py](../src/phasematcher/models/system.py) | Model composition and stage-to-stage initialization |
| [losses.py](../src/phasematcher/training/losses.py) | Contribution targets and seven decomposition losses |
| [module.py](../src/phasematcher/training/module.py) | Prefix supervision and validation-driven scheduling |
| [search.py](../src/phasematcher/inference/search.py) | Full-library Greedy/Beam and per-path residual updates |

## Spectral decomposition

The module estimates a shared shift, phase-specific strain and three broadening weights. It resamples and smooths **reference features along the angular coordinate**, not atomic coordinates. The adapted features condition pointwise intensity allocation among selected phases and one residual source.

A source-axis softmax produces nonnegative allocation fractions. Multiplication by the original observation gives contributions and residual whose pointwise sum equals the observation.

| Interface | Shape | Meaning |
|---|---|---|
| Observation | `[B,L]` | Common-scale observation |
| References | `[B,K,L]` | Reference spectra for the selected set |
| Valid mask | `[B,K]` | Active reference slots |
| Contributions | `[B,K,L]` | Allocated phase intensity |
| Remainder | `[B,L]` | Unexplained intensity |
| Alignment parameters | `[B,K,5]` | Shift in bins, dimensionless strain, three broadening weights |

The shift is shared across valid phases. Only the query residual is max-normalized; physical contribution/residual outputs retain the common observation scale.

## Supervision

Training uses true phase prefixes ordered by decreasing mixture weight. Every nonempty prefix, including a complete set, receives decomposition supervision. Next-phase cross-entropy applies only to incomplete prefixes; complete prefixes receive STOP supervision in stage three.

Phase and STOP losses use the joint action logits in stage three, averaged separately over their respective step types. Phase classification gradients can propagate through residual queries into decomposition.

| Decomposition term | Constraint | Weight |
|---|---|---:|
| `component` | Weighted pointwise SmoothL1 on individual contributions | 1.0 |
| `component_shape` | Contribution cosine shape error | 0.1 |
| `removal` | Weighted pointwise SmoothL1 on selected total contribution | 2.0 |
| `allocation` | Pointwise source-allocation KL | 1.0 |
| `removal_area` | Relative integrated-intensity error | 0.1 |
| `query_shape` | Normalized residual cosine shape error | 0.2 |
| `alignment` | Shift/strain and expected broadening magnitude regularization | 0.0001 |

See `loss.decomposition` in [base.yaml](../configs/base.yaml) for the remaining constants and [training](training.md) for stage budgets and scheduling.

## IDs and search semantics

Library IDs are **0 through N−1**. STOP is the final joint-logit column N, not a reference entry. BOS uses an internal sentinel N+1; padding uses −1.

Inference masks selected phases and disallows STOP before selecting a first phase. Training does not apply those action masks. Beam separately retains up to W unfinished and W completed paths, scored by cumulative action log-probability. Different orders of the same set are not deduplicated. Reaching four phases ends a path without adding a STOP score.

References are sorted by ID for decomposition, while decoder history preserves prediction order. Both search strategies accept known-phase histories. Inference and training retain their respective residual-normalization thresholds; these numerical conventions have not been changed during repository cleanup.
