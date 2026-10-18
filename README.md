# MACT baseline

MACT = **Multimodal Attention CRF Tagger**。这是从已验证实验中分离的 MABSA baseline：DeBERTa-v3-base + 冻结 SigLIP2 视觉特征 + 跨模态注意力 + 五标签约束 CRF，默认 `fusion` 模式。

## 当前实验主线（2026-10-18 用户确认）

**主力候选：MACT＋目标上下文情感分类器的固定等权概率融合。**

| 实验角色 | 方案 | 情感预测规则 |
|---|---|---|
| 主力候选 | 固定等权融合 | 原 MACT 与目标分类器的情感概率按1:1平均，两个温度均为1 |
| 对照一 | 原 MACT | 原 MACT 的 CRF 解码结果 |
| 对照二 | 分类器单独替换 | 保留原 MACT 预测边界，由目标上下文分类器替换情感标签 |

三种方案使用同 seed 配对的原 MACT 预测方面边界。主力候选沿用已验证的
`src/mact/calibrated_fusion.py` 固定等权数值规则，评估入口为
`scripts/evaluate_fixed_fusion.py`。`configs/base.json` 代表原 MACT，
`configs/target_context.json` 代表配套分类器；主力候选需要两套模型及各自 tokenizer。
后续实验以固定等权融合为主线，同时报告两个对照，继续使用 Dev 选模。
15-seed Test 均值为 Twitter2015 **69.10 ± 0.84**、Twitter2017 **69.44 ± 0.85**；
完整对比与评估限制见下方。正式决策见[实验记录](docs/superpowers/plans/2026-10-18-calibrated-fusion.md#confirmed-main-candidate-2026-10-18)。

**上一版分类器单独替换方案（2026-10-17）：** 当时被指定为主力候选，现保留为
对照二。最初 seeds 42、43、44 的 Test Joint F1（均值 ± 样本标准差）：

| 数据集 | 原 MACT | 分类器单独替换 | 平均提升（百分点） |
|---|---:|---:|---:|
| Twitter2015 | 67.81 ± 0.90 | 68.19 ± 0.76 | +0.38 |
| Twitter2017 | 67.91 ± 0.32 | 69.51 ± 1.39 | +1.60 |

六组配对 Test 均提升；seed 44 的两个 Dev 均下降，三 seed 结果不等于统计显著性证明。
所有 checkpoint 均按 Dev 选取；补充复现实验开始前已观察 seed 42 Test。
详细决策与产物位置见[实验记录](docs/superpowers/plans/2026-10-17-target-context.md)。

**后续补充证据（2026-10-17）：** 原三组共12个历史 seed 已完成，Test 均值
在 Twitter2015 上为原 MACT 68.48、候选68.26（−0.22 pp），Twitter2017 为
68.76、68.55（−0.22 pp）；两个数据集均为5升7降。Dev 平均虽分别提升
0.35/0.89 pp，Test 未复现最初三个 seed 的一致增益。因此目前不支持将候选
称为稳定增强。这里的“候选”指当时的分类器单独替换方案。历史12 seed 按以往
结果选取，与42/43/44单独报告；当时的候选指定不表示扩大复现已通过。

**校准融合诊断（2026-10-18）：** 已复用上述15个 seed × 两个数据集，完整
重放两模型 Dev 预测后，比较固定等权概率融合与按句子五折拟合的温度校准融合。
以下均为 **Dev Joint F1 均值 ± 样本标准差**，不是 Test 成绩：

| 方案 | Twitter2015 | Twitter2017 |
|---|---:|---:|
| 原 MACT | 67.45 ± 0.42 | 67.14 ± 0.74 |
| 分类器单独替换 | 67.81 ± 0.57 | 67.95 ± 0.92 |
| 未校准固定等权融合 | 68.04 ± 0.57 | 68.13 ± 0.95 |
| 五折校准＋选权融合 | 67.59 ± 0.71 | 68.13 ± 1.00 |

校准版在2015上比分类器单独替换下降0.22 pp，未通过两个数据集均超过两个对照的
预设门槛，因此校准版未启动 Test。预先设置的等权对照在两个数据集相对原 MACT
均14升1降，相对分类器单独替换平均提升0.23/0.17 pp，成为后续固定等权 Test
验证的依据；该诊断阶段尚未替换主力候选。它与校准版的比较同时改变温度和权重选择，不单独证明温度
校准的因果贡献。checkpoint 已由同一 Dev 选择，五折不能消除该选择影响；
历史12 seed 单独统计。逐 seed、分批结果和复核见
`outputs/calibrated_fusion_20261018/SUMMARY.txt`、`per_seed.csv`、`postrun-audit.json`；
实现与复现实验命令见[实验记录](docs/superpowers/plans/2026-10-18-calibrated-fusion.md)。

**固定等权融合 Test 验证（2026-10-18）：** 用户授权继续后，单独锁定未校准的
1:1 概率融合（两个温度均为1），完成全部15个 seed × 两个数据集。保留原 MACT
预测边界，将原 MACT 与目标上下文分类器的情感概率等权平均；复用已有模型，
本轮未重新训练或使用 Test 选择权重。以下为 **Test Joint F1 均值 ± 样本标准差**：

| 方案 | Twitter2015 | Twitter2017 |
|---|---:|---:|
| 原 MACT | 68.34 ± 0.82 | 68.59 ± 0.88 |
| 分类器单独替换（对照二） | 68.25 ± 0.99 | 68.74 ± 0.98 |
| 固定等权融合（主力候选） | **69.10 ± 0.84** | **69.44 ± 0.85** |

融合相对原 MACT 平均提升 **0.75/0.85 pp**，相对分类器单独替换提升 **0.85/0.70 pp**。
相对分类器单独替换，2015为12升1平2降，2017为14升0平1降；相对原 MACT 分别为
12升1平2降和13升1平1降。最初3 seed 与历史12 seed 两批，在两个数据集上
均值均超过两个对照；历史12 seed 相对分类器单独替换提升0.90/0.74 pp。
2017 seed44 从分类器单独替换71.1075升至71.2681，但主结论依据上述配对均值。

用户查看完整对比后，已正式确认固定等权融合为主力候选，并保留原 MACT 与
分类器单独替换作对照。全部30组对照预测和指标精确重放，边界不变，输入与
源码指纹复核通过。校准版的 Dev 门槛仍为未通过。本轮固定参数评估没有消除此前
Test 已知及历史 seed 选择带来的限制，不宣称统计显著性或未接触基准上的泛化。
完整逐 seed 和分批对比：`outputs/fixed_equal_fusion_test_20261018/SUMMARY.txt`、
`paired_results.csv`；锁定配置与复核：`decision.json`、`postrun-audit.json`。

一套代码供所有数据集与随机种子使用。原 `mabsa_evidence`、`deqa_reproduction` 代码、实验和结果完整保留；本项目的日常训练与评估不读取它们。

## 目录

```text
mact/
├── src/mact/                 # 唯一模型/训练/评估源码
├── configs/base.json         # 35 epochs 等共同超参数
├── configs/datasets/         # 两套官方数据的本地资源路径
├── scripts/                  # 单次训练、评估、多种子调度及汇总
├── tests/                    # 模型、CRF、资源迁移和调度测试
├── assets/                   # 独立复制的数据、图像、预训练权重、视觉缓存
├── .runtime/                 # 独立 Python 3.11.3 和固定版本依赖
├── reference/                # 历史 21 种子结果与原始实验记录
│   └── runs/<dataset>/seedN/ # seed24/25 各两套数据保留 best.pt
├── outputs/                  # 后续新实验，每个实验只存配置/日志/权重/结果
├── verification/             # 迁移复核结果，不作为新实验成绩
└── docs/                     # 模型图、迁移说明与模块扩展说明
```

`assets`、`.runtime` 和大型 checkpoint 未放进 Git；它们是真实独立副本，不是指向旧项目的符号链接或硬链接。资源清单记录 SHA256。

## 单个种子训练

```bash
cd /home/whj/reproject/mact
CUDA_VISIBLE_DEVICES=0 .runtime/bin/python scripts/train.py \
  --config configs/base.json --dataset twitter2015 --seed 25
```

默认输出到 `outputs/twitter2015/seed25/`。更换种子只改 `--seed`；更换数据集只改 `--dataset`。相同输出目录不会覆盖。自定义实验可指定 `--output outputs/my_experiment/twitter2015/seed25`。

默认完整流程：35 epochs → 根据 Dev Joint F1 选 checkpoint → 重载并逐条核对 Dev → 对完整官方 Test 评估一次。日志使用 `DEV aesc_p/aesc_r/aesc_f` 与 `BEST DEV--TEST` 格式，每个 epoch 输出汇总。`--train-only` 可省略最终 Test。

## 多种子自动分组

```bash
.runtime/bin/python scripts/run_seeds.py \
  --seeds 20 21 22 23 24 25 26 27 28 29 30 31 \
  --group-size 4 --gpus 0 1 --output outputs/seeds20_31 --detach
```

每组 GPU0 跑 Twitter2015 四个种子，GPU1 跑 Twitter2017 四个种子；整组退出再进入下一组。所有任务使用同一份源码，没有每个实验复制源码。`--detach` 在后台运行；批次状态为 `outputs/seeds20_31/batch-status.json`，批次日志为 `outputs/seeds20_31.log`，单任务日志在批次目录的 `logs/`。

某个任务失败会明确记录，仍继续后续组。未完成的种子不会被偷偷排除后计算均值。更改正在运行批次的源码或配置会使后续组检查失败；在批次结束后再修改。脚本没有服务器重启自动恢复或 optimizer/RNG 断点续训功能。

手动重新汇总：

```bash
.runtime/bin/python scripts/collect_results.py --output outputs/seeds20_31 \
  --seeds 20 21 22 23 24 25 26 27 28 29 30 31
```

## 使用已保存的 checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 .runtime/bin/python scripts/evaluate.py \
  --dataset twitter2015 \
  --checkpoint reference/runs/twitter2015/seed25/best.pt \
  --output outputs/eval_twitter2015_seed25
```

`--split dev` 可评估 Dev，默认 Test；`--predict-only` 只输出预测。历史 checkpoint 保留原文件字节和原训练配置，评估入口仅将视觉模型、图片和缓存路径重定向到本项目，不修改权重或模型超参数。

## 已保存的结果

所有 21 个种子的配置、指标、预测及日志保存在 `reference/runs/`；汇总为 `reference/all_21_seed_results.json`。完整 checkpoint 保存 seed24/25 × 两套数据的四个 Dev 最优模型。

| Seed | 数据集 | Test JMASA P | R | F1 |
|---|---|---:|---:|---:|
| 24 | Twitter2015 | 67.55 | 68.66 | 68.10 |
| 25 | Twitter2015 | 69.68 | 70.68 | 70.18 |
| 24 | Twitter2017 | 70.19 | 70.58 | 70.38 |
| 25 | Twitter2017 | 67.88 | 69.69 | 68.77 |

两套数据的最高分来自不同种子。全部 21 种子 Test Joint F1：Twitter2015 **67.81 ± 1.14**，Twitter2017 **68.57 ± 0.83**（样本标准差）。追加种子的实验是在查看早期 Test 结果后进行，历史记录保留了全部种子；以后加模块应固定共同种子做对照。

## 验证与环境

```bash
PYTHONPATH=src .runtime/bin/python -m unittest discover -s tests -v
.runtime/bin/python scripts/verify_reference.py --device cuda:0 --output verification/recheck
```

第二条会完整重评四个历史 checkpoint，要求指标和逐条预测一致。首次迁移的报告在 `verification/reference-evaluation/verification.json`。

迁移验证还包括：25 项单元/集成测试；原始文件未变化及资源独立性校验；真实 DeBERTa + 视觉缓存的短训练对照。在仅验证脚本启用确定性 GPU 算法时，两边 224 个模型张量、损失和 Dev 预测逐位一致。正式训练配置保持原样；原配置下的 GPU 更新仍可能存在微小数值非确定性，不能据此承诺重新训练必定逐位复现同一 checkpoint。没有为迁移重新跑完整 35 epochs。

本地资源可以再次校验：`.runtime/bin/python scripts/verify_assets.py`。

环境版本为 Python 3.11.3、PyTorch 2.5.1+cu118、Transformers 4.49.0、NumPy 1.26.4；完整依赖见 `requirements.lock.txt`。使用 `.runtime/bin/python`，不依赖旧项目的环境。换机器仍需兼容的 Linux / NVIDIA 驱动；当前 `.runtime` 是本机运行环境副本，不是跨平台安装包。

## 在 baseline 上增加模块

见 `docs/extending.md`。Git 的 `baseline-v1` 固定经验证版本；新模块从该版本创建分支，并使用新的配置及输出目录。不要改变原有官方 Train/Dev/Test，也不要将 Test 用于 epoch 选择。

已添加 B（方面级情感辅助监督），默认关闭。专用配置只开启权重为 `0.1` 的辅助损失：

```bash
CUDA_VISIBLE_DEVICES=0 .runtime/bin/python scripts/train.py \
  --config configs/aspect_sentiment_aux.json --dataset twitter2015 --seed 42 \
  --output outputs/aspect_sentiment_aux/twitter2015/seed42 --train-only
```

Twitter2017 使用相同配置，将 `--dataset` 和输出路径改为 `twitter2017`。
训练后仍按 Dev 选 checkpoint 并重载核对预测；这里的 `--train-only` 省略 Test。
模型结构、损失归一化与日志字段见 `docs/extending.md` 的 B 节。

C（以空视觉为参照的逐词门控）使用独立配置，B 关闭：

```bash
CUDA_VISIBLE_DEVICES=0 .runtime/bin/python scripts/train.py \
  --config configs/visual_gate.json --dataset twitter2015 --seed 42 \
  --output outputs/visual_gate/twitter2015/seed42 --train-only
```

C 从原预训练骨干重新训练，保留原 CRF 目标、35 epochs 和每 epoch 日志；
详细公式、初始化与门控统计见 `docs/extending.md` 的 C 节。

七标签 CRF 对照使用 `configs/seven_tag.json`，B、C 均关闭：

```bash
CUDA_VISIBLE_DEVICES=0 .runtime/bin/python scripts/train.py \
  --config configs/seven_tag.json --dataset twitter2015 --seed 42 \
  --output outputs/seven_tag/twitter2015/seed42 --train-only
```

`model.label_scheme="seven"` 将共享 `I` 拆为 `I-POS/I-NEU/I-NEG`，
CRF 只允许同一方面内部延续相同情感。默认仍为 `"five"`，原 baseline 与历史权重保留。
七标签由官方方面三元组在内存中转换，数据文件和评估三元组定义不变。

融合后的词间上下文交互使用 `configs/word_context.json`，保留五标签，B、C 关闭：

```bash
CUDA_VISIBLE_DEVICES=0 .runtime/bin/python scripts/train.py \
  --config configs/word_context.json --dataset twitter2015 --seed 42 \
  --output outputs/word_context/twitter2015/seed42 --train-only
```

该模块在原融合头的隐藏特征上加入一层 64 维、4 头词间注意力和可学习相对距离偏置，
输出五标签分数的残差，仍由一次 CRF 解码。输出层零初始化，起始预测与 baseline 一致；
默认 `word_context=false`，历史配置与 checkpoint 可继续使用。细节见 `docs/extending.md`。

R-Drop 训练稳定性对照使用 `configs/rdrop.json`，从五标签 baseline 重新训练，其他实验模块关闭：

```bash
CUDA_VISIBLE_DEVICES=0 .runtime/bin/python scripts/train.py \
  --config configs/rdrop.json --dataset twitter2015 --seed 42 \
  --output outputs/rdrop/twitter2015/seed42 --train-only
```

固定 `rdrop_weight=1.0`，目标为两次 CRF NLL 的平均值加对称序列 KL，均按句子平均，
不再按词数归一化。两次前向共享整图 dropout，神经网络 dropout 独立采样。
KL 使用完整的受约束 CRF 分布，保留两侧边缘概率的梯度，不是逐词 softmax KL。
参数量和推断结构与 baseline 一致；训练需要两次前向，计算预算高于 baseline。
默认权重为零，保持原训练路径；checkpoint 记录训练权重，但推断只做一次前向和 CRF 解码。
参考 [R-Drop, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html)；
此处为适配 CRF 的实现，收益需由固定种子 Dev 对照验证。
# Optional OCR context experiment

`configs/ocr_context.json` enables a separate five-tag baseline control: append up to 64 automatically recognized image-text subwords after the original text, then a separator. Only original words receive labels and CRF predictions. It adds no model parameters and does not combine R-Drop, auxiliary CE, visual gates, or word context. `configs/base.json` keeps the original path.

OCR assets live in `assets/cache/ocr_v1/{dataset}`. They were exported from the frozen Train/Dev feasibility probe using confidence >=0.8 and at least two consecutive ASCII letters per line, with whitespace normalization and engine line order. No labels, target names, manual transcriptions or quality-review notes enter the cache. Missing records or mismatched image/content hashes fail explicitly. The raw official datasets are unchanged.

The first experiment uses seeds 42/43/44, 35 epochs, epoch-only logs, and Dev-only selection/comparison. Its detached launcher is `verification/ocr_context/launch.py`; audits and GPU checks are in the same directory. Outputs are `outputs/ocr_context_s42_43_44_20261014`. This is an exploratory comparison on previously examined seeds.

For a new split, pass `--ocr-cache assets/cache/ocr_v1_test/{dataset}` to the evaluation script. The evaluator checks the original checkpoint-pinned asset and requires identical OCR weights, recognition configuration, engine versions and filtering policy. The Test asset is generated separately; archived training caches and checkpoints are never edited. `force_null` disables the visual patch branch but leaves OCR visible, so that setting is not a text-only ablation.

# Frozen residual supervision controls

`scripts/train_ocr_residual.py` trains only a residual branch over a hash-pinned frozen five-tag baseline. The original `configs/ocr_residual.json` remains the OCR-memory, fused-loss-only control. Two optional configurations add direct sequence supervision:

- `configs/residual_supervised_ocr.json`: independent frozen OCR representations as memory.
- `configs/residual_supervised_text.json`: frozen original-text subword representations as memory; OCR content is never read by this branch.

Both use the same branch parameters, initialization and OCR-presence gate. Text memory selects the first 64 content subwords after encoding the full original sentence; OCR memory encodes its own truncated sequence. These match trainable capacity and active samples, not encoder FLOPs. Missing OCR bypasses either branch exactly.

The per-sentence objective is fused CRF NLL plus `standalone_weight * standalone CRF NLL`, with weight fixed to 1.0 in both new configurations. The standalone term uses residual emissions alone and the same frozen CRF, and is zero for no-OCR rows. Both terms are averaged over all sentences, including inactive ones. Baseline/CRF weights remain frozen, output projection is zero-initialized, and inference still decodes only the fused five-label sequence. No gold labels are supplied at inference.

```bash
CUDA_VISIBLE_DEVICES=0 .runtime/bin/python scripts/train_ocr_residual.py \
  --config configs/residual_supervised_ocr.json --dataset twitter2015 \
  --output outputs/residual_supervised_ocr/twitter2015/seed42 --device cuda:0
```

Use the text configuration for the matched control. The runner uses Train/Dev only, saves the first best Dev checkpoint among epochs 1–35, and reports total loss, fused NLL and standalone NLL once per epoch. Selected-checkpoint diagnostics check reload, OCR removal, unchanged no-OCR predictions and three OCR shuffles. Text-control predictions must remain identical under those content shuffles; removal disables its presence-gated branch. Old branch checkpoints default to OCR memory and zero standalone weight on load.

`configs/partial_freeze6.json` is a separate five-tag baseline training control. It starts
from the original pretrained text backbone and freezes embeddings, the bottom 6 of 12
DeBERTa layers, and shared relative-position embeddings/normalization. Upper layers and
the existing task heads remain trainable; dropout stays in its normal training mode.
The optional `text_freeze_layers` setting defaults to 0, preserving baseline training.
Frozen parameters are excluded from AdamW and checked by SHA256 before/after training.
Use `scripts/train.py --config configs/partial_freeze6.json --dataset twitter2015 --seed 42
--train-only` for Dev-only training; best checkpoints still use earliest maximum Dev Joint F1.

## LLRD × EMA 配对训练

`configs/ema_uniform.json` 保留原优化器学习率，`configs/ema_llrd.json` 使文本层学习率由顶层 `2e-5` 向底层逐层乘 `0.9`。两者均启用系数 `0.99` 的 EMA。`base.json` 不变，新功能默认关闭。

同一条训练轨迹每 epoch 分别评估普通权重和 EMA 权重，各自按 Dev Joint F1 选最早最优 checkpoint。普通输出路径不变，EMA 的权重、tokenizer、Dev 预测和重载验证位于 `ema/`。EMA 评估会恢复在线参数和 RNG；推理只加载一份普通模型，额外开销发生在训练的权重维护与 Dev 评估。

固定两数据集、seeds42/43/44、35 epochs 的四组对照共需要12次训练，产生24个选模结果。先执行四组真实 CUDA 验证，再启动 Dev-only 队列：

```bash
.runtime/bin/python scripts/verify_llrd_ema.py \
  --output verification/llrd_ema_20261016 --gpus 0 1
.runtime/bin/python scripts/run_llrd_ema.py \
  --output outputs/llrd_ema_s42_43_44_20261016 \
  --smoke-root verification/llrd_ema_20261016 --detach
.runtime/bin/python scripts/collect_llrd_ema.py \
  --output outputs/llrd_ema_s42_43_44_20261016
```

以上目录必须是新目录；验证后修改源码或配置会使启动检查失败，需重新验证。GPU0负责Twitter2015，GPU1负责Twitter2017，每卡最多一个本轮任务。队列状态为顶层 `status.json` 和各轨迹 `batch-status.json`，汇总为 `results.json`。只有完整24个结果才报告四组均值与配对差值，失败和未完成任务会保留。该入口不读取或评估Test，也不自动扩大种子集合。

独立使用 EMA checkpoint 时，把 `scripts/evaluate.py` 的 `--checkpoint` 指向某条轨迹的 `ema/best.pt`；本轮只使用 `--split dev`。三种子 Dev 改善属于探索性结果，不能直接宣称稳定 Test 提升。
