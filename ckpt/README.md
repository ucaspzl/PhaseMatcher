# 权重

本目录在 Git 中仅包含说明与 JSON 元数据，不包含 `.pt` 或 `.ckpt` 文件。
下载地址尚未发布；取得权重后放入相应的 `phasemix/` 或 `rruff/` 子目录。

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
