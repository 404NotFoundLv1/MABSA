# 主力 Base 固定备份（2026-10-31）

本目录固定已经确认的主力：原 MACT + DeBERTa-v3-base 目标标记情感分类器。
保留 MACT 预测的方面边界；两路情感概率按 1:1 平均，两个温度均为 1。
主力模型需要两套 checkpoint 和各自的 tokenizer。

## 结果及选模身份

| Test Joint F1 | Twitter2015 | Twitter2017 |
|---|---:|---:|
| 15 seed 均值 ± 样本标准差 | 69.0954 ± 0.8395 | 69.4443 ± 0.8459 |
| 历史单 seed 最高分 | 70.6724（seed26） | 71.2681（seed44） |
| 对应目标分类器 epoch | 29 | 13 |

最高分仅为历史观察，不用于新实验选 seed。历史 Test 已被观察，历史种子亦经过选择，
不能将这些统计描述成未接触基准上的泛化证明。

原主力分类器按其单支在预测边界上的 Dev Joint F1 选模。
后续按融合 Dev 选模的 Base 对照不能覆盖本目录的 checkpoint 和结果。
详见 `BASE_IDENTITY_REVIEW.txt`。

## Git 保存的内容

- 仓库中已有的源码、训练/评价入口、环境依赖及配置。
- `manifest.json`：两组最高分模型的精确路径、SHA-256、epoch 和指标。
- `evidence/fixed_equal_fusion_test_20261018/`：全部 30 组评测的预测、logits、
  指标、来源指纹、锁定决策和汇总，保留原始字节与原始绝对路径。
- `configs/`：两套组件配置。
- `EVIDENCE_SHA256SUMS`：评测证据的校验清单。
- `verification-tests.log`：备份时工作区 195 项测试通过的记录。
  工作区还含后续实验的未提交改动；这些改动不属于本次主力备份提交。

本次核对确认，原评测记录中的全部源码指纹与当前对应文件一致。

## 权重与恢复

Git 中没有 4 个大权重文件。对应权重及 tokenizer/配置另以两个 tar 归档保存，
归档名为 `mact-main-base-twitter2015-seed26-20261031.tar` 和
`mact-main-base-twitter2017-seed44-20261031.tar`。
归档及 Git bundle 的本机备份目录为 `/home/whj/reproject/mact_backup_20261031/`。
**本机归档不等于异地备份；以实际远端上传和校验结果为准。**

先恢复 Git 仓库，再从仓库根目录解开两个 tar。归档中的相对路径与原仓库一致。
历史配置与证据保留 `/home/whj/reproject/mact` 绝对路径；在其他位置恢复时，
须明确重定向资源路径，不应修改原始证据文件来伪造相同的评测身份。
每个归档的文件校验清单位于本目录的 `archives/`。

两份归档保存选定的训练权重、tokenizer 和小型评测依赖文件。
它们不包含完整数据集、原始图像、视觉缓存、SigLIP2 预训练资源或 Python 环境；
执行推理前仍需恢复相应资源（见原 README 和 `reference/assets-manifest.json`）。
其余 13 个 seed 的权重也不在这两个归档中；全部 15 seed 的结果证据保存在 Git。

评测入口是 `scripts/evaluate_fixed_fusion.py`，融合规则是
`src/mact/calibrated_fusion.py`。完整历史入口依赖锁定的全部 15 seed 模型及资源；
仅恢复本目录两组最高分模型时，不要直接启动完整 15 seed 队列。

校验评测证据：

```bash
cd main_base
sha256sum -c EVIDENCE_SHA256SUMS
```

校验恢复的权重与 tokenizer（仓库根目录运行）：

```bash
sha256sum -c main_base/archives/twitter2015-seed26.files.sha256
sha256sum -c main_base/archives/twitter2017-seed44.files.sha256
```
