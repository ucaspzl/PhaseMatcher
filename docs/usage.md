# Inference and outputs

[Home](../README.md) · [Resources](resources.md) · [Architecture](architecture.md)

## Input contract

| Property | Requirement |
|---|---|
| File | NumPy `.npy`, numeric, finite and nonnegative |
| Shape | `[3501]` for one pattern or `[B,3501]` for a batch |
| Axis | 2θ = 10–80°, spacing 0.02° |
| Normalization | Prediction divides each spectrum by its maximum |
| Preprocessing | No automatic interpolation or background removal |

Resample to the reference grid before prediction if necessary. Matching the grid alone does not guarantee generalization to a new instrument or database.

## Command line

The following multiline commands use Bash continuation. In PowerShell, put each command on one line or replace the trailing backslash with a backtick.

```bash
phasematcher predict \
  --checkpoint ckpt/phasemix/last.pt \
  --library dataset/phasemix/reference \
  --input outputs/example/input.npy \
  --strategy beam --beam-size 10 \
  --output outputs/example/beam.json \
  --spectra outputs/example/beam-spectra.npz
```

Create the input with [the runnable example](../examples/predict_sample.py), or substitute your own `.npy` file. Add `--device cpu` or `--device cuda` to select the device.

For known-phase continuation, add `--history 12 34`, replacing these example IDs with valid IDs for your library. Both Greedy and Beam accept a history. Already selected IDs are masked; STOP is masked before the first phase.

## Prediction JSON

The outer list corresponds to input spectra. Each inner list contains candidate paths, ordered by cumulative action log-probability:

| Field | Meaning |
|---|---|
| `phase_ids` | Predicted library row IDs, in selection order |
| `entries` | Corresponding source record IDs from `entry.npy` |
| `log_probability` | Sum of action log-probabilities, not a calibrated confidence |

Greedy returns one path; Beam can return several. Different orders of the same unordered set are not deduplicated. IDs are `0..N-1`; STOP is not a library entry.

## Decomposition NPZ

`--spectra` decomposes the top-ranked candidate for each input. With batch size B, maximum phase count K=4 and spectrum length L=3501:

| Field | Shape | Meaning |
|---|---|---|
| `phase_ids` | `[B,K]` | Selected IDs in sorted ID order; empty slots are −1 |
| `valid_mask` | `[B,K]` | Which slots contain a selected phase |
| `contributions` | `[B,K,L]` | Contribution spectra, in the input intensity scale |
| `remainder` | `[B,L]` | Residual spectrum, in the input intensity scale |
| `alignment_parameters` | `[B,K,5]` | Shift in bins, dimensionless strain and three broadening weights |
| `axis_two_theta` | `[L]` | Angle grid in degrees |

Match spectra by `phase_ids` and `valid_mask`, not JSON selection order. Contributions and remainder sum pointwise to the input (up to floating-point error). Physical parameter units and feature-space operations are described in the model implementation; they are not deformed atomic coordinates.

## Python interfaces

`load_model(checkpoint, device, library)` returns an evaluation-mode model and checkpoint metadata, after validating library compatibility. `Search(model, library).predict(observations, ...)` expects already max-normalized observations and returns lists of `Prediction` objects. The CLI performs normalization for you.

`Search.decompose(observations, histories)` returns tensors in the supplied observation scale. Empty histories return zero contributions and the unchanged observation. See [search.py](../src/phasematcher/inference/search.py) for signatures.
