# Training and evaluation

[Home](../README.md) · [简体中文](training.zh-CN.md) · [Data](data.md) · [Architecture](architecture.md)

## Three training stages

| Stage | Initialization | Objective |
|---|---|---|
| `single` | Random | Single-phase classification cross-entropy |
| `phase` | Single-phase encoder and identity weights | Weighted next-phase cross-entropy + spectral decomposition loss |
| `stop` | Phase-stage model; new STOP branch | Weighted phase cross-entropy + STOP cross-entropy + spectral decomposition loss |

All outer loss weights default to 1. The seven decomposition terms and their weights are listed in [the architecture reference](architecture.md). The phase stage jointly optimizes prediction and decomposition; there is no separate decomposition-pretraining command.

Run from the repository root. The first command produces a timestamped run directory under `outputs/`; replace the placeholder paths in later commands with the actual preceding run's exported `last.pt`:

```bash
phasematcher train --config configs/phasemix.yaml --stage single --devices 1
phasematcher train --config configs/phasemix.yaml --stage phase --init outputs/phasemix/single_RUN/last.pt --devices 1
phasematcher train --config configs/phasemix.yaml --stage stop --init outputs/phasemix/phase_RUN/last.pt --devices 1
```

The paths above illustrate the pattern; they are not pre-existing files. Check the configured `output_root` and the actual run directory. Use `configs/rruff.yaml` for RRUFF. Seven GPUs were used for the joint-training runs; changing device count changes the effective batch size and training trajectory.

## Optimization

| Setting | PhaseMix phase | PhaseMix stop | RRUFF phase | RRUFF stop |
|---|---:|---:|---:|---:|
| Optimizer steps | 120000 | 40000 | 3715 | 1429 |
| Validation interval (optimizer steps) | 2000 | 2000 | 143 | 143 |
| Batch size per device | 18 | 16 | 16 | 16 |
| Gradient accumulation | 1 | 2 | 1 | 1 |
| Initial learning rate | 5e-5 | 5e-5 | 5e-5 | 5e-5 |

Prediction and decomposition use the same initial learning rate. `ReduceLROnPlateau` is stepped after validation, halves the rate after four consecutive non-improving checks (`patience=3`), and bottoms out at 5e-6. The phase stage monitors fixed-count identification; the STOP stage monitors automatic-stop identification.

The single-phase default recipe is an executable starting point, not a claim to exactly reproduce the archived single-phase initialization. For full settings, see [base.yaml](../configs/base.yaml), [phasemix.yaml](../configs/phasemix.yaml), and [rruff.yaml](../configs/rruff.yaml).

## Outputs and recovery

Each run saves configuration, CSV logs, exported `last.pt`, and resumable `checkpoints/last.ckpt`.

```bash
phasematcher train --config configs/phasemix.yaml --stage phase --resume outputs/phasemix/phase_RUN/checkpoints/last.ckpt
```

Use `--init` for the next stage and `--resume` for an interrupted run; they are mutually exclusive. Released `.pt` exports do not contain optimizer or scheduler state. DataLoader prefetch/shuffle state is not guaranteed to resume sample-for-sample.

## Evaluation

```bash
phasematcher evaluate --config configs/phasemix.yaml --strategy greedy --device cuda
phasematcher evaluate --config configs/phasemix.yaml --strategy beam --beam-size 10 --device cuda
```

The default split is `test`; `--split val` selects validation. Use `--limit 8 --device cpu` for a small smoke run. Evaluation uses final `last.pt` unless `--checkpoint` is provided.

Reports contain overall and per-phase-count Exact-set@1, phase recall, count accuracy and Exact-set@10. Values in JSON are fractions, not percentages. Greedy's Exact-set@10 is null. Detailed predictions are saved as JSONL.

For G GPUs, launch G separate processes with `--rank i --world-size G --device cuda:i` and a shared `--output` directory. Combine `metric_sums` and divide by sample counts; do not average per-GPU percentages. Each process writes rank-specific files.
