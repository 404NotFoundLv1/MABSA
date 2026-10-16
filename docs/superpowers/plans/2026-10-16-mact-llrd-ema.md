# MACT LLRD × EMA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 实现并启动十二条 Dev-only 训练轨迹，得到原 MACT、EMA、LLRD、LLRD+EMA 四组可配对结果。

**Architecture:** 复用共享训练入口。LLRD 仅改变优化器分组；EMA 通过实际优化器更新后的回调维护影子参数，用隔离的评估上下文独立选模。两份配置驱动现有调度组件，专用汇总只读取本轮 Train/Dev 结果。

**Tech Stack:** 项目 `.runtime/bin/python`，PyTorch 2.5.1、Transformers 4.49、unittest，无新增依赖。

**Spec:** `docs/superpowers/specs/2026-10-16-mact-llrd-ema-design.md`（用户已确认）。

## Global Constraints

- 保留共享源码目录；不复制源码、不覆盖 `configs/base.json`、历史输出或官方数据。现有分支 `ocr-context`，实现基线 `8c75ee7`，设计提交 `51f0e98`。
- `twitter2015/twitter2017`、种子 `42/43/44`、35 epochs、batch=1、accumulation=8、FP32、五标签 fusion；所有旧实验模块关闭。
- `layer_lr_decay` 默认 `1.0`，实验值 `0.9`；`ema_decay` 默认 `0.0`，实验值 `0.99`；不扫描超参数。
- 普通及 EMA 权重各有35次 Dev候选，分别最大化 Joint F1，并列选最早；不合并候选池。
- 本轮不读取、散列或评估 Test 样本、缓存、预测；所有正式训练命令显式 `--train-only`。
- 每卡最多一个本轮任务；GPU0负责Twitter2015，GPU1负责Twitter2017；不可终止其他任务或静默回退CPU。
- 原指标继续指普通权重；EMA独立保存于 `ema/`。新旧checkpoint可加载，不新增断点续训。

## Review Focus

- Dev DataLoader迭代本身会消耗RNG：EMA完整评估必须恢复随机状态，第二个训练epoch也应与关闭EMA完全匹配（任务2、3）。
- DeBERTa共享位置参数不属于某个encoder layer：显式分组，未知文本参数报错，默认分组顺序不变（任务1）。
- EMA评估、写checkpoint或重载发生异常：恢复在线参数、buffer、模式和Parameter身份；任务不得标为成功（任务2、3）。
- 现有调度日志名不含trajectory：两个trajectory使用不同调度根目录，禁止重名日志和跨轨迹读错结果（任务4）。
- 历史汇总默认读取Test：专用结果读取器必须保护Test路径，检查完整24个结果、预测与checkpoint身份，拒绝部分均值（任务4）。

---

### Task 1: LLRD分组与配置

**Files:** 修改 `src/mact/finetuning.py`、`src/mact/train.py`；新建 `tests/test_llrd.py`；复用 `tests/test_partial_freeze.py` 的 `tiny_deberta()`。

**Interfaces:** 新增 `optimizer_parameter_groups(model, learning_rate, head_learning_rate, weight_decay, layer_lr_decay=1.0) -> tuple[list[dict], list[dict]]`。第一项供AdamW使用，第二项为可JSON序列化的分组记录（参数名、元素数、基础lr、weight_decay）。`resolve_config` 增加两个默认字段及校验；非默认新设置仅允许独立五标签text/fusion，禁止与旧实验模块叠加。正式配置仍仅fusion。

- [x] 写 `test_llrd_covers_every_parameter_once_and_assigns_shared_weights`、`test_default_groups_and_optimizer_updates_match_original`、`test_unknown_text_parameter_and_invalid_config_fail`。独立构建原分组作为默认值对照，做一次真实AdamW更新；测试12层名称映射及实际tiny DeBERTa。

```python
assert by_name['text_encoder.encoder.layer.11.attention.self.query_proj.weight']['lr'] == 2e-5
assert math.isclose(by_name['text_encoder.embeddings.word_embeddings.weight']['lr'], 2e-5 * .9**12)
assert len(parameter_ids) == len(set(parameter_ids)) == len(trainable_parameters)
assert default_config['layer_lr_decay'] == 1.0 and default_config['ema_decay'] == 0.0
```

- [x] 运行 `PYTHONPATH=src:tests .runtime/bin/python -m unittest test_llrd -v`，确认因缺少接口或行为而失败。
- [x] 实现接口并替换train.py的分组构建；lr=1时逐项保留原顺序；共享embeddings/LayerNorm用embedding lr、可选conv用底层lr；拒绝布尔值、NaN、Inf及越界值。分组记录写入provenance，不改变模型参数。
- [x] 重跑上述测试及 `test_partial_freeze`、`test_runtime`，确认通过；提交本任务涉及文件，消息 `feat: add optional layer-wise learning rate decay`。

### Task 2: EMA状态、实际更新回调和隔离上下文

**Files:** 新建 `src/mact/averaging.py`、`tests/test_averaging.py`；修改 `src/mact/runtime.py`。

**Interfaces:** `WeightEMA(model, decay)`，公开 `updates: int`、`shadow: dict[str, Tensor]`；`update(model) -> None`；`average_parameters(model) -> ContextManager[None]`。为 `train_epoch` 末尾追加 `after_optimizer_step: Callable[[], None] | None = None`，每次成功optimizer.step后立即调用一次，默认路径不增加随机操作。

- [x] 写 `test_ema_formula_no_grad_and_partial_accumulation`、`test_context_restores_parameters_buffers_modes_rng_on_error`、`test_online_two_epoch_trajectory_is_identical_with_ema_evaluation`。最后一个使用真正的DataLoader、非零dropout、两轮AdamW，并比较模型、optimizer state和下一次随机抽样；上下文异常测试包括Python/NumPy RNG和混合子模块模式。

```python
assert ema.updates == stats['updates'] == 2  # 三个micro-batch、accumulation=2
torch.testing.assert_close(ema.shadow[name], .99 * before + .01 * after, rtol=0, atol=0)
assert all(not value.requires_grad for value in ema.shadow.values())
assert parameter_ids_after == parameter_ids_before
assert torch.equal(online_without_ema[name], online_with_ema[name])
```

- [x] 运行 `PYTHONPATH=src:tests .runtime/bin/python -m unittest test_averaging -v`，确认新增行为尚不存在。
- [x] 实现FP32影子参数、每次实际更新计数；上下文使用try/finally原位恢复在线参数/buffer/模式和CPU及实际CUDA设备RNG。上下文覆盖评估DataLoader创建迭代器到保存结果全过程，不平均buffer。异常不得吞掉。
- [x] 重跑 `test_averaging`、`test_runtime`、`test_rdrop`；提交消息 `feat: add isolated EMA weight evaluation`。

### Task 3: 双权重流选模、checkpoint与完整Dev重载

**Files:** 修改 `src/mact/train.py`、`runtime.py`、`experiment.py`；新建 `tests/test_llrd_ema_training.py`、`configs/ema_uniform.json`、`configs/ema_llrd.json`；复用现有tiny训练和本地视觉缓存fixture。

**Interfaces:** `save_checkpoint(..., *, weight_metadata: dict | None = None)`，未提供时保持原格式行为。新增metadata字段 `source`（online/ema）、`ema_decay`、`ema_updates`、`trajectory`、`selected_epoch`。`run_training` 仍返回普通summary，新增 `ema` 字段引用相对路径、best_epoch、best_dev和checkpoint hash；EMA完整summary存 `ema/summary.json`。

- [x] 写 `test_online_and_ema_choose_independent_earliest_maxima`、`test_both_checkpoint_streams_reload_exactly`、`test_full_training_ema_does_not_change_online_rng_or_weights`、`test_dev_only_never_opens_or_hashes_test`。选模测试注入不同的两组Dev曲线（包括并列），同时保留真实tiny模型训练/加载集成测试。Test读取保护同时拦截path open和hash入口。

```python
assert online_summary['best_epoch'] == 2  # online F1: .4, .6, .6
assert ema_summary['best_epoch'] == 3     # EMA F1: .3, .5, .7
assert loaded_online_predictions == saved_online_predictions
assert loaded_ema_predictions == saved_ema_predictions
assert ema_summary['ema_updates'] == online_summary['optimizer_updates']
assert forbidden_test_reads == []
```

- [x] 运行 `PYTHONPATH=src:tests .runtime/bin/python -m unittest test_llrd_ema_training -v`，确认失败点对应缺失行为。
- [x] 连接任务1/2接口；每epoch先原有online评估，再隔离EMA评估；两个best和last、预测、summary分别保存。online仍使用原best.pt及原summary语义，EMA启用才新增字段。独立记录 `DEV ONLINE/DEV EMA`，保留原默认日志行为。`run_experiment` 在标记完成前核对两组完整Dev重载，任一失败即整条轨迹失败。
- [x] 两份新配置复制base参数并添加EMA=.99，uniform的LLRD=1、llrd=.9；两份配置只有LLRD不同，所有其他字段完全相等。仅配置metadata新增不影响老checkpoint加载。
- [x] 重跑 `test_llrd_ema_training`、`test_project`、`test_logging`、`test_runtime` 和前两任务测试；提交消息 `feat: train and select paired online and EMA checkpoints`。

### Task 4: Dev-only调度及四组配对汇总

**Files:** 新建 `src/mact/llrd_ema_results.py`、`scripts/run_llrd_ema.py`、`scripts/collect_llrd_ema.py`、`tests/test_llrd_ema_schedule.py`、`tests/test_llrd_ema_results.py`；复用 `schedule.build_groups/run_schedule`。

**Interfaces:** `collect(root: Path, seeds: Sequence[int]=(42,43,44)) -> dict` 生成并原子写 `results.json`；报告状态、24个结果行、逐数据集A/B/C/D统计、相对A的配对差值、筛选候选、共同正确边界情感准确率。启动命令：`.runtime/bin/python scripts/run_llrd_ema.py --output outputs/llrd_ema_s42_43_44_20261016 --smoke-root verification/llrd_ema_20261016 --detach`。

- [x] 写 `test_twelve_jobs_have_isolated_logs_and_train_only`、`test_hash_change_or_failed_smoke_blocks_launch`、`test_detached_launcher_records_failure_without_orphans`、`test_incomplete_duplicate_nonfinite_or_mismatched_results_are_rejected`、`test_paired_deltas_sample_sd_and_common_boundary_metrics`。合成结果验证两数据集都平均为正且各至少2胜才筛入；并列按B/C/D顺序。

```python
assert len(jobs) == len({(j['dataset'], j['seed'], j['trajectory']) for j in jobs}) == 12
assert all('--train-only' in command(j) for j in jobs)
assert len(log_paths) == len(set(log_paths)) == 12
assert incomplete['state'] == 'incomplete' and incomplete['aggregate'] is None
assert complete['aggregate']['twitter2015']['B']['wins'] == 2
```

- [x] 运行 `PYTHONPATH=src:tests .runtime/bin/python -m unittest test_llrd_ema_schedule test_llrd_ema_results -v`，确认预期失败。
- [x] 启动器创建新顶层目录，先uniform、再llrd，各用独立调度目录和group_size=1，两GPU并行且组间等待；轨迹目录为 `<root>/<trajectory>/<dataset>/seedN`，复用原日志命名也不会冲突。顶层记录状态/PID/固定源码与配置hash，组前验证；正式启动要求四份匹配当前源配置的CUDA smoke通过。保留既有调度组件对普通任务失败的继续执行行为，顶层结果仍标失败/不完整；源文件漂移阻止后续任务。
- [x] 汇总只读取本轮普通/EMA summary、best Dev预测、checkpoint hash及Dev gold；核对dataset/seed/配置/权重来源、35轮完成及重载报告，重新计算预测指标。没有完整24行就不给完整均值；不读取历史Test，不按历史最佳种子筛选。每组及全部完成后自动汇总，结果文件自含计算单位（F1百分点）。
- [x] 重跑新增测试及 `test_schedule`、`test_multiseed`；提交消息 `feat: schedule and summarize Dev-only LLRD EMA experiment`。

### Task 5: 实际资源验证、文档与正式启动

**Files:** 新建 `scripts/verify_llrd_ema.py`、`tests/test_llrd_ema_verification.py`；更新 `README.md`、`docs/extending.md`；验证输出仅写 `verification/llrd_ema_20261016/`，正式输出仅写新目录。

**Interfaces:** `.runtime/bin/python scripts/verify_llrd_ema.py --output verification/llrd_ema_20261016 --gpus 0 1`。输出每个dataset/trajectory的 `integration_check.json`（passed、CUDA设备/峰值显存、更新次数、EMA次数、两份checkpoint重载一致性、源配置hash）、resolved smoke config和整体manifest。使用 `resolve_experiment` 后加smoke限制，调用 `run_training`；重载必须显式传入已验证本地缓存和资源路径，不能直接使用旧smoke的无缓存入口。

- [x] 写并先运行 `test_smoke_requires_cuda_updates_both_streams_and_current_hashes`：CPU、零更新、EMA次数不符、旧源hash或任意重载失败均拒绝正式启动。实现验证器，固定每组seed42、16 Train/8 Dev、1epoch、2次更新、log_every=0；保留原样本id。记录并要求任务头参数变化量为正，反向梯度有限，实际优化器学习率符合分组记录；四组在对应GPU串行执行，每卡最多一个。
- [x] 运行 `PYTHONPATH=src .runtime/bin/python -m unittest discover -s tests -v`，要求退出码0；记录实际测试数量和输出，不复用以前通过数。
- [x] 按选定执行方式完成代码审阅；重点检查Review Focus、默认路径与原实现比较、source/config冻结、退出清理。修复发现的问题并重跑受影响验证。
- [x] 更新文档说明默认关闭、新配置、开销、双流选模/加载、Dev-only启动/汇总命令与限制；确认 `git diff --check` 通过、基线配置与历史文件未改变，提交最终源码。
- [x] `nvidia-smi` 检查占用后运行上述四组真实CUDA验证；核对正显存、两次实际更新、两流预测及指标重载一致、源码hash未变。失败则诊断修复，相关测试及最终源码smoke重新通过后再启动；不声称CPU测试等同CUDA验证。
- [x] 执行任务4的正式后台启动命令，确认父调度PID存活、两个首任务在正确GPU运行、ready事件及首个完整epoch ONLINE/EMA日志。此时只报告“训练已启动”，不报告提升。
- [ ] 十二条轨迹全部结束后检查24份选模结果、完整重载验证及配对汇总，提交结果报告；长训练可后台继续并交代状态路径，但不得把启动当作效果验证完成。本轮不自动启动Test或扩展种子。

## 自审与执行交接

已覆盖设计全部章节：任务1落实LLRD规则，任务2落实EMA及隔离，任务3落实双流checkpoint和Dev重载，任务4落实完整配对/筛选与Dev-only队列，任务5落实真实资源验证及启动。额外实体先验诊断是设计中的可选项，本轮不新增该分析器。

推荐主助手在当前会话顺序实施：五个任务共享训练接口和状态，逐项由同一实施者集成便于维护一致性；最终做一次独立代码审阅。用户已选择当前会话顺序实施；2026-10-16 开始执行，任务1–4已实现并通过各自验证。

## 执行证据（2026-10-16）

- 实现及审查修复：`6db724c..b82c799`；基线配置未改动。
- 全套回归：138 项通过。独立审查无 Critical；两项 Important（checkpoint 内部身份、汇总读取前 Train/Dev 路径保护）已用先失败后通过的测试修复。
- 额外决定：将最后一组结束后的资源冻结复核提升为必须修复项；错误判断的成本仅为额外一次哈希检查。
- 暂缓小项：`test_detached_launcher_records_failure_without_orphans` 的名称超出其模拟异常测试覆盖；实际子进程清理由既有 scheduler 测试覆盖。
- 最终源码 CUDA 验证：`verification/llrd_ema_20261016`，四组均通过，各有2次 optimizer/EMA 更新，梯度有限、任务头改变、实际学习率匹配、两份权重重载预测与指标完全一致。峰值分配显存：uniform 4,902,796,288 bytes，LLRD 4,903,058,432 bytes。
- 正式训练结果仍待运行完成；短训练分数不计入效果比较。

- 正式后台队列：`outputs/llrd_ema_s42_43_44_20261016`；父 PID 5962，首组 Twitter2015/GPU0 PID 7031、Twitter2017/GPU1 PID 7032，均显式 `--train-only`。已核对父子进程存活、GPU 映射、ready 与首个完整 epoch ONLINE/EMA 日志。
- 首轮仅作启动证据：Twitter2015 2101 Train/727 Dev、263 次更新；Twitter2017 1746 Train/577 Dev、219 次更新，均非 smoke，EMA 更新数一致。两条轨迹各计划35轮，剩余任务由后台调度自动继续。
- `status.json` 记录调度状态，`results.json` 仅在完整24份有效结果齐备后给出配对统计。当前不得据首轮分数作效果结论；末项结果报告尚待训练完成。
