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

## 七标签 CRF 对照

配置：`configs/seven_tag.json`，`model.label_scheme="seven"`，B、C 关闭。
标签顺序为 `O/B-POS/B-NEU/B-NEG/I-POS/I-NEU/I-NEG`。
`I-p` 只能接在 `B-p` 或 `I-p` 后，开头和 O 后不能出现 I；
相邻 B 仍可表示相邻的独立方面。训练与解码共同使用这些硬约束。

同一种子先按原顺序初始化完整五标签模型，再把文本、视觉输出层的 I 行各复制为
三个独立可训练行，CRF 的起止及转移参数也从相应五标签参数复制。
扩展不消耗额外的全局随机数，公共参数与原 baseline 初始化相同。
合法五标签路径和七标签路径一一对应，在初始化时序列分数及分区函数等价
（数值计算允许浮点舍入差）；训练后 I 的三个极性参数独立更新。
fusion 配置增加 1,056 个可训练参数。

标签转换发生在读取官方方面三元组之后，原 Train/Dev/Test 文件不改写。
推理仍一次 CRF 解码并还原为原 `(start,end,polarity)` 三元组，没有金标输入或额外分类阶段。
五标签 checkpoint 继续使用 `mabsa_sequence_crf_v1`；七标签使用
`mabsa_sequence_crf_v2`，加载时校验版本、标签表与模型配置一致，
旧 v1 配置中缺少 `label_scheme` 时按五标签加载。

首次对照固定种子 42、43、44，Twitter2015/Twitter2017 各 35 epochs，
其余训练配置沿用 baseline。先比较 Dev JMASA、MATE，以及单词/多词方面的
边界匹配后情感准确率；使用 Dev JMASA F1 选 checkpoint。
多词方面重复累积情感分数可能改变边界偏好，需同时检查情感与抽取指标。

## 融合后的词间上下文交互

配置：`configs/word_context.json`。从原预训练骨干重新训练，`label_scheme="five"`、
`word_context=true`，B、C 关闭。所有历史模块和结果保留在共享源码中；本轮使用
`word-context` 分支，并以 `baseline-v1` 的数值行为作为禁用新模块时的校验基准。
首版只允许五标签 fusion 单独启用该模块，其他组合明确拒绝，以免混入未经验证的变量。

DeBERTa 已提供文本上下文。本模块进一步交换跨模态融合后的词特征，既不使用真实方面
位置，也不先解码方面。无观点标注，因此不能将注意力权重直接解释为正确的方面—观点配对。

复用原 `visual_head` 的隐藏层 `u_i = GELU(W[word_i;visual_attention_i] + b)`，维度256。
原有文本和视觉分数仍完整保留。在 `u` 上计算一次词间注意力：

```text
q, k, v = Linear(u)                       # 各 64 维，无 bias，4 个 head
d_ij = clip(j - i, -16, 16)
alpha_ij = softmax_j(q_i k_j / sqrt(16) + relative_bias[head, d_ij])
context_i = concat_heads(sum_j alpha_ij v_j)
delta_i = Linear(context_i)               # 64 -> 5，weight 和 bias 初始均为0
emissions_i = original_emissions_i + delta_i
loss = 原句子均值 CRF NLL
prediction = 原五标签 CRF 的一次 Viterbi 解码
```

每个 head 的相对偏置初始化为 `-abs(d_ij)/16`，然后随任务训练；超过16词的距离
仍可访问，只共享边缘桶。保留 self-attention。padding 不能作为被读取的 key，padding
位置的输出修正为零；每句至少一个有效词。首版不增加新 dropout，避免改变原随机流。
新增模块在局部 RNG 上下文中初始化，共享参数和之后的随机状态保持不变。
零输出层使首步只有该层获得非零梯度，后续更新后 Q/K/V 和距离偏置也能学习。

默认参数量增加49,609：Q/K/V为49,152，相对偏置132，输出层325。训练和推断都使用
该分支，无辅助情感损失；原视觉 dropout 和 force-null 仍在融合特征形成前生效。
关闭时不创建新参数，原前向、损失和梯度保持不变。模型配置随 checkpoint 保存；
仍使用五标签 `mabsa_sequence_crf_v1`，旧 checkpoint 缺少字段时默认关闭。

首轮固定两套数据、种子42/43/44、每组35 epochs，其余配置沿用 baseline。
只按 Dev JMASA F1 选模，先不评估 Test。日志保留每 epoch 汇总，启动时记录模块设置。
自动汇总包含 JMASA P/R/F1、MATE、匹配边界情感准确率、NEG/NEU/POS 混淆矩阵，
以及新旧模型共同识别正确边界上的情感准确率。对照中的共同比较子集由同一 seed 的
两套预测与 gold 边界交集确定，避免把边界集合变化误当作纯情感提升。
