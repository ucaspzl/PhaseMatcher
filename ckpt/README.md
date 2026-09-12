# 权重

本目录在 Git 中仅包含说明与 JSON 元数据，不包含 `.pt` 或 `.ckpt` 文件。
两个最终推理权重均已在 [checkpoints-v1](https://github.com/ucaspzl/PhaseMatcher/releases/tag/checkpoints-v1) 正式发布，文件大小和 SHA-256 与本地资产记录一致。
下载状态与放置方法：[中文](../docs/resources.zh-CN.md) · [English](../docs/resources.md)。
仓库当前为私有，下载需要有访问权限的 GitHub 账号。
两个最终权重采用 [MIT 许可](../LICENSE)，授权范围见[许可说明](../docs/licensing.zh-CN.md)。数据集和参考库不随权重改用 MIT。

| 下载附件 | 字节数 | 下载后放置路径 |
|---|---:|---|
| [phasemix-last.pt](https://github.com/ucaspzl/PhaseMatcher/releases/download/checkpoints-v1/phasemix-last.pt) | 493651977 | `ckpt/phasemix/last.pt` |
| [rruff-last.pt](https://github.com/ucaspzl/PhaseMatcher/releases/download/checkpoints-v1/rruff-last.pt) | 218159113 | `ckpt/rruff/last.pt` |

下载后按上表重命名为 `last.pt`。预测和评测只需要对应的最终权重与数据资源。
可用 `phasematcher check --config configs/phasemix.yaml --hash` 校验推理必需资源；无需前两阶段权重。
SHA-256 见对应的 `last.json` 或根目录 `assets.json`。
三阶段权重的含义如下；`single.pt` 和 `phase.pt` 尚未发布。

| 数据集 | single.pt | phase.pt | last.pt |
|---|---|---|---|
| PhaseMix-135K | 单相预训练权重 | 多相联合训练最终权重，120,000 步 | 加入 STOP 的联合训练最终权重，40,000 步 |
| RRUFF | 单相预训练权重 | 多相联合训练最终权重，3,715 步 | 加入 STOP 的联合训练最终权重，1,429 步 |

`last.pt` 是用于最终评测的阶段三训练结束权重。
每个文件旁的 JSON 包含来源 checkpoint 哈希、参考库顺序哈希、架构和训练进度。
`assets.json` 提供各文件的 SHA-256。

文件格式为 `format_version=1`，参数前缀为 `model.*` 和 `spectral_decomposition.*`；
单相文件为 `encoder.*` 和 `classifier.*`。

`.pt` 仅保存模型权重，可用于预测、评测或下一阶段初始化。
恢复中断训练需使用训练过程中保存的 `.ckpt`，其中包含优化器和调度器状态。
