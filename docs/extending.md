# 模块扩展入口

基准模型的 forward、CRF、标签编码、损失、优化器和数据顺序沿用原实验。当前迁移的必要改动仅为资源路径解析、统一实验入口及调度，不改变 baseline 数值计算。

| 文件 | 职责 / 后续扩展位置 |
|---|---|
| `src/mact/model.py` | `ModelConfig`、`SequenceModel`；编码、跨模态交互、标签分数融合。新模块可在这里接入。 |
| `src/mact/crf.py` | 五标签 CRF 的 NLL 与 Viterbi；约束同时用于训练和解码。 |
| `src/mact/labels.py` | O/B-POS/B-NEU/B-NEG/I 和方面三元组互转。 |
| `src/mact/train.py` | 配置校验、随机数、优化器分组、学习率调度、Dev 选模。 |
| `src/mact/runtime.py` | 梯度累积、句子均值损失、梯度裁剪、预测与 checkpoint。 |
| `src/mact/experiment.py` | 项目路径解析、完整训练/重载/Test 流程、源码/Git 记录。 |
| `src/mact/common/cache.py` | 冻结视觉缓存和图像/模型 SHA256 校验。 |
| `scripts/run_seeds.py` | 使用同一套代码，以参数区分数据集与种子。 |

新模块从 Git `baseline-v1` 创建分支。复制 `configs/base.json` 为新配置，例如 `configs/my_module.json`，在 `ModelConfig` 中定义明确开关并默认关闭。默认关闭时应保持 baseline 的前向与损失一致；启用时需记录完整模型配置，并验证 checkpoint 重载。

已有 `text/fusion/evidence` 三个代码分支为历史实现；本 baseline 的正式配置为 `fusion`。保留旧分支有利于已有单元测试与 checkpoint 兼容，不表示 evidence 差分机制已参与这些成绩。

不要在新模块实验中覆盖 `reference/`。模型比较应使用相同数据、训练预算、种子集合和 Dev 选模方式，分别保存输出。

## B：方面情感辅助监督

配置：`configs/aspect_sentiment_aux.json`，仅比 `base.json` 多
`model.aspect_sentiment_weight = 0.1`。默认值为 0，关闭时不创建辅助参数，
保留原模型初始化、前向、CRF 损失与解码；旧 checkpoint 缺少此字段时按 0 加载。

辅助头平均池化每个真实训练方面的 `word_projection` 输出（256 维文本表示），
再用 `Linear(256, 3)` 预测 NEG/NEU/POS。该分支增加 771 个参数。
方面范围和情感从已有五标签训练序列恢复，不枚举候选，不引入外部标注。
O、特殊 token 和 padding 不参与方面池化。相邻 B 标签分别代表两个方面。

每句目标为 `CRF_NLL + 0.1 * mean(aspect_cross_entropy)`；无方面句子的辅助损失为零。
再沿用 baseline 的句子均值梯度累积。因此多个方面不会直接把整句辅助权重成倍放大。
辅助分支只监督共享文本表示，不改变视觉融合、标签定义或 CRF。
推理不传入真实标签，不计算辅助头，仍一次性解码方面和情感。

`history.json` 中 `sentence_loss` 是总目标，`sentence_nll` 是原 CRF NLL，
`aspect_sentiment_loss` 是未乘权重、包含无方面句子零值的句均辅助 CE。
每个 epoch 的可读日志同时显示三个量，`log_every=0` 保持不打印逐 step 日志。

首次试验固定 seed 42、两套数据各 35 epochs、lambda=0.1，仅 Dev 选模与比较。
使用 `scripts/train.py --train-only`，避免开发阶段反复查看 Test。结果需要训练完成后判断，
单种子结果不能视为稳定提升；有效后再以预先固定的共同种子集合验证。

## C：以空视觉为参照的逐词门控

配置：`configs/visual_gate.json`，`visual_gate=true`、`visual_gate_init=0.9`，
`aspect_sentiment_weight=0`。从原 baseline 的预训练骨干重新训练，不从 B checkpoint 继续。
B 的实现和结果完整保留，baseline 配置也不变；C 关闭时不创建门控参数。

同一次 forward 共用文本词表示、attention 和 visual_head，分别计算正常图像与仅保留
null token 的分数。对于每个词，`g = sigmoid(Linear([word, attention_image-attention_null]))`，
输出为 `E_null + g * (E_image-E_null)`。门值对该词的五类标签共用。
门控权重初始为零、bias 为 logit(0.9)，故初始门值固定为 0.9。
新增参数为 `512+1=513`，训练目标仍只有原 CRF NLL，不增加辅助标签或门控损失。

门值为 0 时保留完整空视觉路径中的文本计算，为 1 时恢复原 fusion（浮点运算允许微小舍入差）。
`force_null` 和原有的 15% 按样本视觉 dropout 同样作用于 C；实际视觉与空视觉路径相同时，
修正量为零。C 只适用于 `fusion` 模式，旧 checkpoint 缺少新配置字段时默认关闭 C。

每个 epoch 记录 `train.visual_gate` 的有效词数、均值、标准差及门值低于 0.1／高于 0.9 的比例，
忽略 padding；可读日志只增加一行 `TRAIN Gate ...`。这些统计包括视觉 dropout 的样本，
门值大小本身不等于图像对结果的实际贡献。Dev 指标定义及 checkpoint 选择规则保持原样。

首轮固定两套数据的共同种子 42、43、44，各 35 epochs，先运行 `--train-only` 比较 Dev，
不根据中途结果修改配置或剔除种子。源码通过 Git 和 SHA256 固定，输出另存。
