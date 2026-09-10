# Data specification

[Home](../README.md) · [中文](data.zh-CN.md) · [Downloads](resources.md)

Both datasets contain a reference library, five perturbed single-phase realizations per entry, and mixture manifests. Mixtures are constructed on demand rather than stored as a dense mixture matrix.

| Dataset | Library entries | Training mixtures | Validation | Test |
|---|---:|---:|---:|---:|
| PhaseMix-135K | 135258 | 12135258 | 100000 | 100000 |
| RRUFF-based mixtures | 740 | 100000 | 7400 | 7400 |

PhaseMix entries are structure records; distinct record IDs are not deduplicated into a mineral/species label. RRUFF references are preprocessed measured single-phase profiles. `entry.npy` defines the unique row-ID to source-record mapping and must not be reordered.

## Files

Paths are relative to `dataset/<phasemix|rruff>/`.

| Path | Shape | Meaning |
|---|---|---|
| `reference/patterns.npy` | `[N,3501]` | Reference patterns, float32 |
| `reference/entry.npy` | `[N]` | Source record IDs |
| `reference/axis_two_theta.npy` | `[3501]` | 2θ in degrees |
| `observations/patterns.npy` | `[N,5,3501]` | Five perturbed realizations |
| `observations/entry.npy` | `[N]` | Same ID order as reference |
| `observations/axis_two_theta.npy` | `[3501]` | Same angle grid as reference |
| `manifest/SPLIT/SPLIT_ids.npy` | `[M,4]` | IDs in descending weight order; padding −1 |
| `manifest/SPLIT/SPLIT_weights.npy` | `[M,4]` | Mixture weights; padding 0 |
| `manifest/SPLIT/SPLIT_counts.npy` | `[M]` | Phase counts, 1–4 |

`SPLIT` is `train`, `val` or `test`. Library patterns are stored at peak scale 100 and divided by 100 when read. PhaseMix perturbations use float16; RRUFF perturbations use float32. Both are converted to float32 for computation. Manifest IDs are int32 for PhaseMix and int64 for RRUFF, converted to int64 when loaded.

Exact dtypes, shapes, sizes and SHA-256 digests are in [assets.json](../assets.json).

## Mixture construction

Training samples realizations 0, 1 and 2; validation uses realization 3 and testing uses realization 4. Additional mixture-level perturbations are:

| Perturbation | Range |
|---|---|
| Shared zero shift | −0.03° to 0.03° |
| Additional FWHM | 0 to 0.05° |
| Background peak / structural-mixture peak | 0 to 0.01 |
| Gaussian noise standard deviation / structural-mixture peak | 0.0008 to 0.002 |

These are additional mixture-level perturbations, not the complete parameter ranges used to generate the stored single-phase bank. Observations are clipped to nonnegative values and divided by their overall maximum. All contribution targets use **the same observation scale**, not separate per-component normalization.

Validation/test use `seed + 1 + sample_index` / `seed + 2 + sample_index` with NumPy RandomState. Seeds are 20260826 (PhaseMix) and 20260909 (RRUFF). Training uses worker RNG states.

## Batch fields

| Field | Shape | Meaning |
|---|---|---|
| `index` | `[B]` | Manifest row |
| `mixture` | `[B,1,3501]` | Normalized observation |
| `phase_ids`, `phase_weights` | `[B,4]` | Ordered targets and weights |
| `counts` | `[B]` | True phase count; supervision only |
| `targets` | `[B,5]` | Phase sequence, STOP column N, padding −1 |
| `reference_patterns` | `[B,4,1,3501]` | Target references |
| `component_contributions` | `[B,4,1,3501]` | Common-scale contribution targets |
| `residual_patterns_common` | `[B,5,1,3501]` | Residual targets for all prefixes |
| `observation_scale` | `[B,1]` | Pre-normalization observation maximum |

Conservative targets cap selected total contribution pointwise at the observation and distribute it among selected phases in proportion to their target intensity. Residual targets are the observation minus that contribution. This handles local noise-induced discrepancies while preserving nonnegativity and intensity conservation.

At inference the model receives only the observation and reference library, not labels, weights or phase counts. The test example uses labels only inside data construction, never as model input.
