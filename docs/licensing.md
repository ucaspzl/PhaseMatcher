# Licensing

[Home](../README.md) · [简体中文](licensing.zh-CN.md)

| Resource | License |
|---|---|
| Original PhaseMatcher source code and documentation | [MIT](../LICENSE) |
| Final PhaseMatcher checkpoints listed below | [MIT](../LICENSE) |
| PhaseMix-135K dataset | CC BY 4.0 as declared in the [dataset repository](https://huggingface.co/datasets/pengzhonglong/PhaseMix-135k) |
| RRUFF-derived data and reference patterns | Not covered by the code or checkpoint license; redistribution terms remain to be documented before publication |
| Third-party software and other third-party content | Their respective licenses |

## Final checkpoints

The PhaseMatcher authors license both `checkpoints-v1` weight files under the MIT terms in the root `LICENSE`. For this grant, the licensed work includes the checkpoint tensors. Retain the copyright and license notice when redistributing the weights or substantial portions of them.

| File | SHA-256 |
|---|---|
| `phasemix-last.pt` | `89b3017a4c24754df19822577c9c5994e99242b84830814dfe429b0e58df26a9` |
| `rruff-last.pt` | `146bbc324dfb60ca1e03560bff2ec2360adb9cf4b766a81e55e0999e32b20c77` |

Licensing the model weights does not relicense their training datasets or the reference libraries required for inference. Dataset licenses and attribution requirements remain separate. The MIT grant does not apply to external baseline implementations.
