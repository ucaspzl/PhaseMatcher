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
`single.pt` 和 `phase.pt` 尚未上传；以下保留本地三阶段资产的来源说明。

| 数据集 | single.pt | phase.pt | last.pt |
|---|---|---|---|
| phasemix | 既有单相初始化 | v2 多相联合训练 last，120000 步 | v2 STOP 联合训练 last，40000 步 |
| rruff | RRUFF 单相初始化 | 新扰动混合谱联合训练 last，3715 步 | 对应 STOP last，1429 步 |

`last.pt` 是当前用于最终测试的阶段三权重，不是按验证集筛选的 best。
每个文件旁的 JSON 包含来源 checkpoint 哈希、参考库顺序哈希、架构和训练进度。
`assets.json` 提供整理后文件的 SHA-256。

格式只有一个版本：`format_version=1`，参数前缀统一为 `model.*` 和 `spectral_decomposition.*`；
单相文件为 `encoder.*` 和 `classifier.*`。不兼容旧命名，也不自动猜测旧 checkpoint 格式。

这是精简的纯权重导出，未保存原优化器和调度器状态。可用于预测、评测或下一阶段初始化。
原始完整 checkpoint 未修改。新训练会另外生成自己的可恢复 `.ckpt`。
