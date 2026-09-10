# 推理与输出

[中文首页](../README.zh-CN.md) · [English](usage.md) · [资源指南](resources.zh-CN.md) · [模型结构](architecture.zh-CN.md)

## 输入要求

| 项目 | 要求 |
|---|---|
| 文件 | NumPy `.npy`，数值有限且非负 |
| 形状 | 单条谱 `[3501]`，批量谱 `[B,3501]` |
| 角度轴 | 2θ = 10–80°，间隔 0.02° |
| 归一化 | 预测入口将每条谱除以其最大值 |
| 预处理 | 不自动插值或扣背景 |

若输入角度网格不同，需先重采样到参考网格。网格匹配并不等于模型已具备对新仪器或新数据库的泛化能力。

## 命令行预测

下面使用 Bash 的反斜杠换行。在 Windows PowerShell 中，可以把整条命令写在一行，或将行末反斜杠换为反引号。

```bash
phasematcher predict \
  --checkpoint ckpt/phasemix/last.pt \
  --library dataset/phasemix/reference \
  --input outputs/example/input.npy \
  --strategy beam --beam-size 10 \
  --output outputs/example/beam.json \
  --spectra outputs/example/beam-spectra.npz
```

可以先用[运行示例](../examples/predict_sample.py)生成输入，也可以将 `--input` 替换为自己的 `.npy` 文件。添加 `--device cpu` 或 `--device cuda` 指定设备。

已知物相续推使用 `--history 12 34`；这里的数字仅作示例，应替换为当前参考库的有效编号。Greedy 和 Beam 均支持初始历史。推理会屏蔽已选物相，并在预测第一个物相前屏蔽 STOP。

## 预测 JSON

最外层列表对应输入谱，每条谱对应一个候选路径列表，按累计动作对数概率排序：

| 字段 | 含义 |
|---|---|
| `phase_ids` | 按预测顺序排列的参考库行编号 |
| `entries` | `entry.npy` 中对应的来源记录 ID |
| `log_probability` | 动作对数概率之和，不是经过校准的置信度 |

Greedy 返回一条路径，Beam 可以返回多条。同一无序集合的不同预测顺序不去重。物相编号为 `0..N-1`；STOP 不是参考库条目。

## 分解结果 NPZ

`--spectra` 对每条输入的最高分候选进行分解。设批量大小为 B，最大物相数 K=4，谱长度 L=3501：

| 字段 | 形状 | 含义 |
|---|---|---|
| `phase_ids` | `[B,K]` | 按编号升序排列的已选物相；空槽为 −1 |
| `valid_mask` | `[B,K]` | 各槽是否包含有效物相 |
| `contributions` | `[B,K,L]` | 输入强度尺度下的物相贡献谱 |
| `remainder` | `[B,L]` | 输入强度尺度下的残差谱 |
| `alignment_parameters` | `[B,K,5]` | 以网格点数为单位的偏移、无量纲应变和三个展宽权重 |
| `axis_two_theta` | `[L]` | 角度网格，单位为度 |

应根据 `phase_ids` 和 `valid_mask` 匹配贡献谱，不能直接套用 JSON 中的预测顺序。各贡献谱与残差逐点相加等于输入，仅存在浮点误差。形变操作作用于参考谱特征，不是改变原子坐标。

## Python 接口

`load_model(checkpoint, device, library)` 校验参考库兼容性，返回处于评估模式的模型及权重元数据。

`Search(model, library).predict(observations, ...)` 要求观测谱已按最大值归一化，返回由 `Prediction` 对象组成的候选列表。命令行入口会自动完成这一步归一化。

`Search.decompose(observations, histories)` 返回与所提供观测相同强度尺度的张量。空历史对应全零贡献谱和未改变的观测残差。完整参数见 [search.py](../src/phasematcher/inference/search.py)。
