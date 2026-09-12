# Data and pretrained weights

[Home](../README.md) · [简体中文](resources.zh-CN.md)

## Data and weights

| Resource | Link | Contents |
|---|---|---|
| PhaseMix-135K dataset | [Hugging Face](https://huggingface.co/datasets/pengzhonglong/PhaseMix-135k) | References, perturbed patterns, and mixture manifests |
| RRUFF-based mixture data | [Data layout](#rruff-data) | Required folders and reference ordering |
| Final inference weights | [checkpoints-v1](https://github.com/ucaspzl/PhaseMatcher/releases/tag/checkpoints-v1) | PhaseMix and RRUFF models |

| Checkpoint | Size (bytes) | Local path |
|---|---:|---|
| [phasemix-last.pt](https://github.com/ucaspzl/PhaseMatcher/releases/download/checkpoints-v1/phasemix-last.pt) | 493,651,977 | `ckpt/phasemix/last.pt` |
| [rruff-last.pt](https://github.com/ucaspzl/PhaseMatcher/releases/download/checkpoints-v1/rruff-last.pt) | 218,159,113 | `ckpt/rruff/last.pt` |

## What to download

| Task | Required assets |
|---|---|
| Predict your own spectrum | Dataset's `reference/` folder + final `last.pt` |
| Run the example or evaluate | All three dataset folders + final `last.pt` |
| Train from scratch | All three dataset folders; each stage uses the preceding stage's output |
| Initialize a later training stage | The preceding training run's exported `last.pt` |

The final weights include the encoder, Phase/STOP heads and spectral decomposition module. They do not include optimizer state. Both final weights use the [MIT License](../LICENSE); datasets and reference libraries retain separate terms ([details](licensing.md)).

## File placement

Download the contents of the Hugging Face dataset repository into `dataset/phasemix/` (not a second nested `phasemix/` directory):

```bash
hf download pengzhonglong/PhaseMix-135k --repo-type dataset --local-dir dataset/phasemix
```

Download the checkpoint above and rename it to `last.pt` in the corresponding directory:

```text
PhaseMatcher/
├── ckpt/
│   ├── phasemix/last.pt
│   └── rruff/last.pt
└── dataset/
    ├── phasemix/
    │   ├── reference/
    │   ├── observations/
    │   └── manifest/
    └── rruff/
        ├── reference/
        ├── observations/
        └── manifest/
```

Do not reorder `entry.npy` or reference-library rows. Checkpoints validate the reference ID ordering.

## RRUFF data

The RRUFF model uses 740 reference patterns. Place its `reference/`, `observations/`, and `manifest/` folders under `dataset/rruff/` as shown above, and select `configs/rruff.yaml`. Prediction requires the matching reference library; test-set examples and evaluation also require the observations and manifests. See the [data specification](data.md) for array fields and the asset manifest for file hashes.

## Verify

```bash
phasematcher check --config configs/phasemix.yaml --hash
phasematcher check --config configs/phasemix.yaml --scope dataset --hash
```

The default checks only the final weights and reference library. `--scope dataset` additionally checks the perturbation bank and all mixture manifests; `--scope all` also requires the two earlier-stage weights. Omit `--hash` for size, shape and model/library checks without hashing every asset.

[`assets.json`](../assets.json) records exact sizes, array shapes and SHA-256 digests. See [checkpoint metadata](../ckpt/README.md) and the [data specification](data.md) for provenance and fields.
