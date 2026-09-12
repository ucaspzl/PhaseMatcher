<div align="center">

# PhaseMatcher

### 面向粉末 X 射线衍射的自回归物相集合识别

**[English](README.md) · [简体中文](README.zh-CN.md)**

[快速开始](#快速开始) · [数据与权重](docs/resources.zh-CN.md) · [训练指南](docs/training.zh-CN.md) · [模型结构](docs/architecture.zh-CN.md)

</div>

PhaseMatcher 从一条粉末 XRD 谱中识别完整物相集合，无需预先提供相数。模型逐步选择参考库条目，根据原始观测与全部已选参考谱重建贡献谱和残差，并学习何时结束预测。

![从语言自回归到物相自回归](docs/assets/overview.png)

- **全库识别**：从完整参考库预测物相。
- **物理引导谱分解**：预测物理形变参数，将观测强度分配给已选物相和残差。
- **Greedy / Beam**：沿单条路径预测，或保留多个候选；也可从已知物相继续预测。

## 快速开始

### 1. 安装

需要 Python 3.10+、PyTorch 2.4+。使用 GPU 时，先安装与 CUDA 兼容的 PyTorch。

```bash
git clone https://github.com/ucaspzl/PhaseMatcher.git
cd PhaseMatcher
python -m venv .venv
```

Linux/macOS 使用 `source .venv/bin/activate` 激活环境；Windows PowerShell 使用 `.venv\Scripts\Activate.ps1`。然后执行：

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

测试使用临时生成的小样本，不依赖完整数据集或预训练权重。

### 2. 下载数据与权重

| 数据集 | 参考库条目数 | 数据与配置 | 最终权重 |
|---|---:|---|---|
| PhaseMix-135K | 135,258 | [Hugging Face](https://huggingface.co/datasets/pengzhonglong/PhaseMix-135k) | [下载 last](https://github.com/ucaspzl/PhaseMatcher/releases/download/checkpoints-v1/phasemix-last.pt) |
| 基于 RRUFF 的混合谱 | 740 | [数据放置说明](docs/resources.zh-CN.md#rruff-数据) | [下载 last](https://github.com/ucaspzl/PhaseMatcher/releases/download/checkpoints-v1/rruff-last.pt) |

下载对应数据集的权重，按照[资源指南](docs/resources.zh-CN.md)放置文件并校验完整性。

### 3. 跑一个测试集样本

放好对应数据集与最终 `last.pt` 权重后，执行：

```bash
python examples/predict_sample.py --config configs/phasemix.yaml --device cpu
```

示例按测试集配方构造一条留出的混合谱，预测物相并保存：

```text
outputs/example/
├── input.npy             实际送入模型的 XRD 谱
├── predictions.json      预测库内编号、来源记录 ID 与路径得分
└── decomposition.npz     物相贡献谱、残差与形变参数
```

模型只接收观测谱与参考库，**不接收真实标签或相数**。添加 `--strategy beam --beam-size 10` 使用 Beam，`--device cuda` 使用 GPU；换成 `--config configs/rruff.yaml` 使用 RRUFF。示例需要真实数据与权重，不使用随机模型代替。

## 使用自己的谱

输入为非负 NumPy 数组，形状 `[3501]` 或 `[B, 3501]`；2θ 网格为 10°–80°，间隔 0.02°。预处理要求、预测命令和已知物相续推见[使用指南](docs/usage.zh-CN.md)。

下载资源后，可先做一个小规模评测：

```bash
phasematcher evaluate --config configs/phasemix.yaml --limit 8 --device cpu
```

去掉 `--limit 8` 即评测完整测试集。三阶段训练和多 GPU 评测见[训练指南](docs/training.zh-CN.md)。

## 代码结构

```text
src/phasematcher/
├── models/          谱编码器、Phase/STOP 双头、谱分解
├── data/            参考库读取与混合谱构造
├── training/        训练目标、候选采样与优化
├── inference/       Greedy/Beam 与各路径的残差更新
├── checkpoint.py    权重加载与参考库顺序校验
├── metrics.py       集合识别与相数指标
└── cli.py           check / predict / evaluate / train 入口
```

建议从 [models/system.py](src/phasematcher/models/system.py) 看模型组成，再读 [inference/search.py](src/phasematcher/inference/search.py)。配置在 `configs/`，运行示例在 `examples/`，回归测试在 `tests/`。

本仓库只包含 PhaseMatcher 的源码、配置、示例、测试、说明和资源元数据，不打包各个 baseline 的实现。权重通过 Releases 下载，数据集单独托管。

| 文档 | 内容 |
|---|---|
| [资源指南](docs/resources.zh-CN.md) | 下载、放置与完整性校验 |
| [使用指南](docs/usage.zh-CN.md) | 输入预处理、输出字段、Python 接口 |
| [训练指南](docs/training.zh-CN.md) | 三阶段训练、初始化、中断恢复与评测 |
| [模型结构（中文）](docs/architecture.zh-CN.md) | 数据流、模块职责、损失与搜索 |
| [数据规范（中文）](docs/data.zh-CN.md) | 划分、字段形状、观测构造 |

## 许可

PhaseMatcher 自有代码与两个最终权重采用 [MIT 许可](LICENSE)。数据集和第三方内容保留各自条款，见[许可说明](docs/licensing.zh-CN.md)。
