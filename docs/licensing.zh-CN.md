# 许可说明

[中文首页](../README.zh-CN.md) · [English](licensing.md)

| 资源 | 许可 |
|---|---|
| PhaseMatcher 自有源码与说明文档 | [MIT](../LICENSE) |
| 下表列出的最终 PhaseMatcher 权重 | [MIT](../LICENSE) |
| PhaseMix-135K 数据集 | 按[数据仓库](https://huggingface.co/datasets/pengzhonglong/PhaseMix-135k)声明的 CC BY 4.0 |
| RRUFF 派生数据与参考谱 | 适用独立的数据权利与再分发条款，不包含在代码和权重的 MIT 授权中 |
| 第三方软件及其他第三方内容 | 各自的原有许可 |

## 最终权重

PhaseMatcher 作者将 `checkpoints-v1` 中的两个权重文件按根目录 [LICENSE](../LICENSE) 的 MIT 条款授权；本项授权覆盖权重张量。再分发权重或其主要部分时，须保留版权与许可声明。正式许可文本以英文 LICENSE 为准。

| 文件 | SHA-256 |
|---|---|
| `phasemix-last.pt` | `89b3017a4c24754df19822577c9c5994e99242b84830814dfe429b0e58df26a9` |
| `rruff-last.pt` | `146bbc324dfb60ca1e03560bff2ec2360adb9cf4b766a81e55e0999e32b20c77` |

权重采用 MIT 不代表其训练数据、推理所需参考库也改用 MIT。数据集的许可与署名要求仍独立适用；外部 baseline 的实现也不属于本项目的 MIT 授权范围。
