# 数据与预训练权重

[中文首页](../README.zh-CN.md) · [English](resources.md)

## 数据与权重

| 资源 | 链接 | 内容 |
|---|---|---|
| PhaseMix-135K 数据集 | [Hugging Face](https://huggingface.co/datasets/pengzhonglong/PhaseMix-135k) | 参考谱、扰动谱与混合清单 |
| 基于 RRUFF 的混合谱数据 | [数据放置说明](#rruff-数据) | 所需目录与参考库顺序 |
| 最终推理权重 | [checkpoints-v1](https://github.com/ucaspzl/PhaseMatcher/releases/tag/checkpoints-v1) | PhaseMix 与 RRUFF 模型 |

| 权重 | 大小（字节） | 本地路径 |
|---|---:|---|
| [phasemix-last.pt](https://github.com/ucaspzl/PhaseMatcher/releases/download/checkpoints-v1/phasemix-last.pt) | 493,651,977 | `ckpt/phasemix/last.pt` |
| [rruff-last.pt](https://github.com/ucaspzl/PhaseMatcher/releases/download/checkpoints-v1/rruff-last.pt) | 218,159,113 | `ckpt/rruff/last.pt` |

## 需要下载哪些文件

| 用途 | 必需资源 |
|---|---|
| 预测自己的谱 | 对应数据集的 `reference/` 文件夹和最终 `last.pt` |
| 运行示例或评测 | 数据集的三个文件夹和最终 `last.pt` |
| 从头训练 | 数据集的三个文件夹；后续阶段使用前一阶段的训练输出 |
| 初始化后续训练阶段 | 前一阶段训练导出的 `last.pt` |

最终权重包含编码器、Phase/STOP 双头与谱分解模块，不含优化器状态。两个最终权重采用 [MIT 许可](../LICENSE)，数据集和参考库保留各自条款（[许可说明](licensing.zh-CN.md)）。

## 放置路径

将 Hugging Face 数据仓库中的内容下载到 `dataset/phasemix/`，不要再嵌套一层 `phasemix/`：

```bash
hf download pengzhonglong/PhaseMix-135k --repo-type dataset --local-dir dataset/phasemix
```

下载上表中的权重，分别放到对应目录，并重命名为 `last.pt`：

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

不要修改 `entry.npy` 或参考谱库的行顺序。加载权重时会校验参考库编号顺序。

## RRUFF 数据

RRUFF 模型使用 740 条参考谱。按上图将对应的 `reference/`、`observations/` 和 `manifest/` 放到 `dataset/rruff/`，并使用 `configs/rruff.yaml`。预测需要匹配的参考库；测试集示例和评测还需要扰动谱与混合清单。数组字段见[数据规范](data.zh-CN.md)，文件哈希见资源清单。

## 完整性校验

```bash
phasematcher check --config configs/phasemix.yaml --hash
phasematcher check --config configs/phasemix.yaml --scope dataset --hash
```

| 检查范围 | 检查内容 |
|---|---|
| 默认 `inference` | 最终权重与参考库 |
| `--scope dataset` | 在默认范围上增加扰动单相谱和全部混合清单 |
| `--scope all` | 再增加前两个训练阶段的权重 |

省略 `--hash` 时不逐个计算文件哈希，但仍检查大小、形状和模型与参考库的匹配关系。

[`assets.json`](../assets.json) 记录准确的文件大小、数组形状和 SHA-256。来源与字段说明见[权重元数据](../ckpt/README.md)和[数据规范](data.zh-CN.md)。
