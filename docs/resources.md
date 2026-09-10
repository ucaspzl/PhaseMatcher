# Data and pretrained weights

[Home](../README.md) · [简体中文](resources.zh-CN.md)

## Release status

| Resource | Location | Status |
|---|---|---|
| PhaseMix-135K dataset | [Hugging Face](https://huggingface.co/datasets/pengzhonglong/PhaseMix-135k) | Upload in progress; not yet verified complete |
| RRUFF-based mixture dataset | Hugging Face repository pending | Not uploaded |
| Final inference weights | [GitHub Releases](https://github.com/ucaspzl/PhaseMatcher/releases) | RRUFF verified in draft; PhaseMix upload retry in progress |

The checkpoint release tag will be `checkpoints-v1`. GitHub is currently private: downloading requires repository access. The dataset repository and code repository have independent visibility settings.

## What to download

| Task | Required assets |
|---|---|
| Predict your own spectrum | Dataset's `reference/` folder + final `last.pt` |
| Run the example or evaluate | All three dataset folders + final `last.pt` |
| Train from scratch | All three dataset folders; each stage uses the preceding stage's output |
| Initialize from an archived stage | Corresponding `single.pt` or `phase.pt` (not currently published) |

The final weights include the encoder, Phase/STOP heads and spectral decomposition module. They do not include optimizer state.

## File placement

Download the contents of the Hugging Face dataset repository into `dataset/phasemix/` (not a second nested `phasemix/` directory). For example, after the dataset upload is complete:

```bash
hf download pengzhonglong/PhaseMix-135k --repo-type dataset --local-dir dataset/phasemix
```

After the checkpoint release is published, download `phasemix-last.pt` or `rruff-last.pt` and rename it to `last.pt` in the corresponding directory:

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

## Verify

```bash
phasematcher check --config configs/phasemix.yaml --hash
phasematcher check --config configs/phasemix.yaml --scope dataset --hash
```

The default checks only the final weights and reference library. `--scope dataset` additionally checks the perturbation bank and all mixture manifests; `--scope all` also requires the two earlier-stage weights. Omit `--hash` for size, shape and model/library checks without hashing every asset.

[`assets.json`](../assets.json) records exact sizes, array shapes and SHA-256 digests. See [checkpoint metadata](../ckpt/README.md) and the [data specification](data.md) for provenance and fields.
