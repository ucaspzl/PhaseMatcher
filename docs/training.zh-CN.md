# 训练与评测

[中文首页](../README.zh-CN.md) · [English](training.md) · [数据规范](data.zh-CN.md) · [模型结构](architecture.zh-CN.md)

## 三阶段训练

| 阶段 | 初始化 | 目标 |
|---|---|---|
| `single` | 随机参数 | 单相分类交叉熵 |
| `phase` | 单相编码器与物相身份权重 | 加权下一物相交叉熵＋谱分解损失 |
| `stop` | Phase 阶段模型；新初始化 STOP 分支 | 加权物相交叉熵＋STOP 交叉熵＋谱分解损失 |

各项损失的外层权重默认均为 1。谱分解的七项内部损失及权重见[模型结构](architecture.zh-CN.md)。第二阶段联合优化预测与谱分解。

在仓库根目录运行。第一条命令会在 `outputs/` 下创建带时间戳的运行目录；后续命令中的占位路径需替换为实际前一阶段导出的 `last.pt`：

```bash
phasematcher train --config configs/phasemix.yaml --stage single --devices 1
phasematcher train --config configs/phasemix.yaml --stage phase --init outputs/phasemix/single_RUN/last.pt --devices 1
phasematcher train --config configs/phasemix.yaml --stage stop --init outputs/phasemix/phase_RUN/last.pt --devices 1
```

`single_RUN` 和 `phase_RUN` 是路径示意，实际位置由配置中的 `output_root` 和运行时间决定。RRUFF 使用 `configs/rruff.yaml`。联合训练使用七张 GPU；改变设备数会改变有效批量和训练轨迹。

## 优化设置

| 设置 | PhaseMix phase | PhaseMix stop | RRUFF phase | RRUFF stop |
|---|---:|---:|---:|---:|
| 优化器步数 | 120000 | 40000 | 3715 | 1429 |
| 验证间隔，按优化器步计 | 2000 | 2000 | 143 | 143 |
| 每卡批量大小 | 18 | 16 | 16 | 16 |
| 梯度累积步数 | 1 | 2 | 1 | 1 |
| 初始学习率 | 5e-5 | 5e-5 | 5e-5 | 5e-5 |

预测部分和谱分解部分使用相同的初始学习率。每次验证后触发 `ReduceLROnPlateau`；连续四次验证未改善后将学习率减半（`patience=3`），最低为 5e-6。Phase 阶段监控固定相数识别，STOP 阶段监控自动停止识别。

单相默认配置提供参考训练配方；评测模型的单相预训练初始学习率、有效批量大小和训练精度缺少记录。配置选项见 [base.yaml](../configs/base.yaml)、[phasemix.yaml](../configs/phasemix.yaml) 和 [rruff.yaml](../configs/rruff.yaml)。

## 训练输出与恢复

每次运行保存配置、CSV 日志、导出的 `last.pt` 和可恢复训练的 `checkpoints/last.ckpt`。

```bash
phasematcher train --config configs/phasemix.yaml --stage phase --resume outputs/phasemix/phase_RUN/checkpoints/last.ckpt
```

`--init` 用于初始化下一阶段，`--resume` 用于恢复中断训练，两者不能同时使用。发布的 `.pt` 不含优化器或调度器状态。DataLoader 的预取和打乱状态不保证逐样本无缝恢复。

## 评测

```bash
phasematcher evaluate --config configs/phasemix.yaml --strategy greedy --device cuda
phasematcher evaluate --config configs/phasemix.yaml --strategy beam --beam-size 10 --device cuda
```

默认使用 `test` 划分；添加 `--split val` 使用验证集。`--limit 8 --device cpu` 可用于小规模试运行。默认加载最终 `last.pt`，也可通过 `--checkpoint` 指定其他权重。

评测报告提供总体及分相数的 Exact-set@1、phase recall、count accuracy 和 Exact-set@10。JSON 中指标为 0–1 比例，不是百分数；Greedy 的 Exact-set@10 为 null。逐样本预测另存为 JSONL。

使用 G 张 GPU 时，启动 G 个独立进程，分别设置 `--rank i --world-size G --device cuda:i`，并使用相同的 `--output` 目录。汇总时先累加 `metric_sums`，再除以样本数，不要直接平均各卡百分比。各进程写入带 rank 编号的独立文件。
