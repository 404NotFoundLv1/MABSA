# MACT baseline

MACT = **Multimodal Attention CRF Tagger**。这是从已验证实验中分离的 MABSA baseline：DeBERTa-v3-base + 冻结 SigLIP2 视觉特征 + 跨模态注意力 + 五标签约束 CRF，默认 `fusion` 模式。

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

Current OCR caches and evaluation support cover Train/Dev only. A future Test evaluation requires an explicit new cache entry point validating identical OCR weights/configuration/filtering; do not edit the archived cache or checkpoint to add Test. `force_null` disables the visual patch branch but leaves OCR visible, so that setting is not a text-only ablation.
