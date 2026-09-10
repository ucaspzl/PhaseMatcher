# PhaseMatcher

独立的多相鉴定代码：单相预训练 → 多相联合训练 → 加入 STOP 的联合训练。
支持 PhaseMix-135K、RRUFF；不依赖旧项目目录。

此 Git 仓库只保存代码、配置、说明和资产校验清单，**不包含数据集或模型权重**。
数据与权重的下载位置尚未发布；获得资产后按下述目录放置，再运行预测、评测或训练。
单元测试使用临时生成的小样本，无需下载完整资产。

## 目录

```text
phasematcher/
├── src/phasematcher/
│   ├── models/          谱编码器、Phase/STOP 双头、物理引导谱分解
│   ├── data/            参考库、混合谱构造、共同强度尺度监督
│   ├── training/        三阶段目标、候选采样、学习率调度
│   ├── inference/       Greedy/Beam，共用完整历史残差重建
│   ├── checkpoint.py    严格加载和导出权重
│   ├── metrics.py       集合级指标与分相数统计
│   └── cli.py           统一命令入口
├── configs/             共享定义 + 两个数据集配置
├── ckpt/                两套数据各 single.pt / phase.pt / last.pt
├── dataset/             参考谱、扰动单相谱、train/val/test 混合清单
├── docs/                架构、数据字段、训练说明
├── tests/               数值、梯度、搜索、训练与恢复测试
├── verification/        本地校验结果（不纳入 Git）
└── assets.json          所有数据与权重的大小、形状及 SHA-256
```

从 `models/system.py` 看模型组成，从 `training/module.py` 看训练目标，
从 `inference/search.py` 看每一步如何更新状态。

## 安装与检查

在本目录执行。Python ≥ 3.10；使用 GPU 时先安装与本机 CUDA 匹配的 PyTorch。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux:   source .venv/bin/activate
python -m pip install -e ".[dev]"
phasematcher check --config configs/phasemix.yaml --hash
phasematcher check --config configs/rruff.yaml --hash
python -m pytest -q
```

本地整理目录中的 `.venv` 不纳入 Git；克隆或复制到其他机器后重新创建环境。
`check --hash` 会读取全部资产。只做快速大小、形状和模型匹配检查时省略 `--hash`。

## 预测与评测

输入为固定 2θ 网格上的非负谱，`.npy` 形状 `[3501]` 或 `[B,3501]`。
网格为 10–80°，步长 0.02°。预测入口执行最大值归一化，不自动做背景扣除或插值。

```bash
phasematcher predict --checkpoint ckpt/phasemix/last.pt --library dataset/phasemix/reference --input sample.npy --strategy beam --beam-size 10 --output prediction.json --spectra decomposition.npz
phasematcher evaluate --config configs/phasemix.yaml --strategy greedy --device cuda
phasematcher evaluate --config configs/phasemix.yaml --strategy beam --beam-size 10 --device cuda --batch-size 4
```

换成 `configs/rruff.yaml` 和 `ckpt/rruff/last.pt` 即可使用 RRUFF 模型。
快速测试可加 `--limit 8 --device cpu`。完整评测默认遍历全部测试样本。
`--history 12 34` 指定已知的库内编号；Greedy 和 Beam 均可继续预测。

预测 JSON 同时输出库内编号和 `entry.npy` 中的记录 ID。
可选的 `decomposition.npz` 包含贡献谱、残差和形变参数；贡献谱恢复到输入谱的强度尺度。
按 `phase_ids` 与 `valid_mask` 识别有效槽，不能把填充值 −1 当作物相。

多 GPU 评测用独立进程分片：每个进程设置自己的 `--device cuda:i --rank i --world-size G`。
输出文件包含 rank，指标可按 `metric_sums` 相加后除以样本数；**不要平均各卡百分比**。

## 三阶段训练

```bash
phasematcher train --config configs/phasemix.yaml --stage single --devices 7
phasematcher train --config configs/phasemix.yaml --stage phase --init ckpt/phasemix/single.pt --devices 7
phasematcher train --config configs/phasemix.yaml --stage stop --init ckpt/phasemix/phase.pt --devices 7
```

上面后两条假定已另行取得并放置阶段权重。若从头串联自己的三阶段训练，将 `--init` 指向前一阶段
新生成的 `outputs/.../last.pt`。
默认设备数为 1；原联合训练使用 7 卡，改变卡数会改变有效批量和训练轨迹。

每个新运行生成独立输出目录，保存配置、CSV 日志、精简 `last.pt` 和可恢复的
`checkpoints/last.ckpt`。中断恢复使用 `--resume outputs/.../checkpoints/last.ckpt`。
资产清单中的六个 `.pt` 是权重导出，不含优化器状态，不能用于无缝恢复原训练进度。

详细说明：[模型与训练](docs/architecture.md) · [数据字段与规模](docs/data.md) · [权重说明](ckpt/README.md)
