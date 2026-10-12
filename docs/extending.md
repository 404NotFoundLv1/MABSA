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
