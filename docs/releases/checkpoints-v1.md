# PhaseMatcher final checkpoints v1

Final stage-three **last** checkpoints used for evaluation, not validation-selected best checkpoints. Both files are fully uploaded and verified against the asset manifest.

用于最终评测的阶段三 **last** 权重，并非按验证集筛选的 best。两个文件均已上传，大小与 SHA-256 已核验。

| Asset / 文件 | Dataset / 数据集 | Global step / 步数 | Local path / 放置路径 |
|---|---|---:|---|
| `phasemix-last.pt` | PhaseMix-135K | 40000 | `ckpt/phasemix/last.pt` |
| `rruff-last.pt` | RRUFF | 1429 | `ckpt/rruff/last.pt` |

Rename each downloaded checkpoint to `last.pt` in its corresponding directory. The files contain the complete inference weights, including spectral decomposition, without optimizer or scheduler state. Earlier-stage checkpoints and datasets are not included.

下载后放到上表目录并重命名为 `last.pt`。文件包含完整推理模型和谱分解模块，不含优化器、调度器状态，也不包含前两阶段权重或数据集。

## License / 许可

Original PhaseMatcher code and these two final checkpoints are licensed under [MIT](https://github.com/ucaspzl/PhaseMatcher/blob/main/LICENSE). Retain the copyright and license notice when redistributing. Datasets and third-party content retain separate terms; see [English](https://github.com/ucaspzl/PhaseMatcher/blob/main/docs/licensing.md) / [中文](https://github.com/ucaspzl/PhaseMatcher/blob/main/docs/licensing.zh-CN.md).

PhaseMatcher 自有代码及这两个最终权重采用 MIT；再分发时须保留版权与许可声明。数据集和第三方内容不随之改许可。

## Data and source / 数据与源码

- [PhaseMix-135K dataset](https://huggingface.co/datasets/pengzhonglong/PhaseMix-135k) is available on Hugging Face. RRUFF data are not yet published. / PhaseMix 数据已发布；RRUFF 数据尚未发布。
- [Current PhaseMatcher source](https://github.com/ucaspzl/PhaseMatcher/tree/main) contains no bundled baseline implementations. Data and checkpoint binaries are distributed separately. / 当前源码只包含 PhaseMatcher，不打包 baseline，数据和权重单独下载。
- [Download and setup guide / 下载与放置指南](https://github.com/ucaspzl/PhaseMatcher/blob/main/docs/resources.md) · [中文](https://github.com/ucaspzl/PhaseMatcher/blob/main/docs/resources.zh-CN.md).

The GitHub repository remains private; code and weight downloads require repository access. / GitHub 仓库仍为私有，下载代码与权重需要仓库访问权限。

## SHA-256

```text
89b3017a4c24754df19822577c9c5994e99242b84830814dfe429b0e58df26a9  phasemix-last.pt
146bbc324dfb60ca1e03560bff2ec2360adb9cf4b766a81e55e0999e32b20c77  rruff-last.pt
```
