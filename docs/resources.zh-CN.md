# 数据与预训练权重

[中文首页](../README.zh-CN.md) · [English](resources.md)

## 发布状态

| 资源 | 位置 | 状态 |
|---|---|---|
| PhaseMix-135K 数据集 | [Hugging Face](https://huggingface.co/datasets/pengzhonglong/PhaseMix-135k) | 已发布；15 个文件及 SHA-256 均已校验 |
| 基于 RRUFF 的混合谱数据集 | Hugging Face 仓库待建立 | 尚未上传 |
| 最终推理权重 | [GitHub Releases](https://github.com/ucaspzl/PhaseMatcher/releases) | RRUFF 已在草稿中校验通过；PhaseMix 正在重传 |

权重版本标签为 `checkpoints-v1`。GitHub 仓库当前为私有，下载需要仓库访问权限。数据仓库与代码仓库的公开设置相互独立。

## 需要下载哪些文件

| 用途 | 必需资源 |
|---|---|
| 预测自己的谱 | 对应数据集的 `reference/` 文件夹和最终 `last.pt` |
| 运行示例或评测 | 数据集的三个文件夹和最终 `last.pt` |
| 从头训练 | 数据集的三个文件夹；后续阶段使用前一阶段的训练输出 |
| 使用归档阶段权重初始化 | 对应的 `single.pt` 或 `phase.pt`，目前尚未发布 |

最终权重包含编码器、Phase/STOP 双头与谱分解模块，不含优化器状态。

## 放置路径

将 Hugging Face 数据仓库中的内容下载到 `dataset/phasemix/`，不要再嵌套一层 `phasemix/`：

```bash
hf download pengzhonglong/PhaseMix-135k --repo-type dataset --local-dir dataset/phasemix
```

权重 Release 发布后，下载 `phasemix-last.pt` 或 `rruff-last.pt`，分别放到对应目录，并重命名为 `last.pt`：

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
