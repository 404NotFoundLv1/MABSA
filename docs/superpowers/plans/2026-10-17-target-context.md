# Target context pilot implementation

Spec: docs/superpowers/specs/2026-10-17-target-context.md

Use executing-plans inline. User already authorized implementation and the
shared current branch. Base: 6951492. No new worktree or approval handoff.

## Global constraints

Train/Dev only; original MACT spans and sources immutable; no Test tuning;
formal seed42, 35 epochs per dataset; exploratory stop rule in spec.

## Task 1: Marked examples and standalone classifier

Files: src/mact/target_context.py, tests/test_target_context.py.

Interfaces: encode_target(tokenizer, words, start, end, max_length) returns
input_ids/attention_mask/target_index; TargetDataset accepts examples plus
explicit span triples and supervised flag (inference discards polarity).
relabel preserves complete ID and ordered boundary lists. TargetClassifier
consumes batched tokens and target_index; save/load uses a distinct format.

1. Write tests for repeated/multiword targets, full context, overflow, malformed
   predictions, empty sentences, label-independent inference, encoder gradients,
   exact checkpoint reload. Run focused unittest; Expected: missing feature fails.
2. Implement minimal interfaces. Run focused unittest; Expected: all pass.
3. Commit task. Verify through task-done with focused unittest.

## Task 2: Train/Dev runner and pilot launch

Files: src/mact/target_context_train.py, scripts/train_target_context.py,
configs/target_context.json, tests/test_target_context_training.py.

Interfaces: run_training(config, output) consumes Task 1 dataset/classifier,
explicit Train/Dev files and frozen baseline predictions/summary, writes fresh
output, status/history/best checkpoint, full Dev reload verification and hashes.
CLI restricts datasets to official Train/Dev and fixed baseline cohort; smoke
uses few Train aspects/one epoch but complete Dev and is explicitly labelled.

1. Write real tiny-transformer training/reload tests, baseline provenance metric
   rejection, fresh-output protection, earliest-primary-only selection, and
   no Test access. Run tests; Expected: missing runner fails.
2. Implement runner, fixed config and CLI. Run focused tests; Expected: pass.
3. Run full unittest suite; Expected: all pass. Commit task and task-done.
4. Final fresh-context review; fix Important/Critical findings RED/GREEN.
5. Real GPU smoke; Expected: finite loss, full Dev reload exact, boundary invariant.
6. Launch two formal jobs using existing scheduler. Confirm first epoch/status;
   record process IDs, output path and measured ETA. Do not claim final scores.

## Review Focus

- Repeated target strings and token splitting must select the correct occurrence.
- Gold Dev labels or boundaries must never leak into primary inference inputs.
- Partial/duplicate predictions must not silently change extraction counts.
- Selection/reload must use predicted-boundary Joint F1, not diagnostic accuracy.
- Interrupted/changed-input runs must not appear completed; no Test entry point.

## Execution evidence

Implementation: a8afe13 (marked encoder), b98e2d1 (Train/Dev runner).
144 tests passed, including a real tiny-transformer training/reload cycle and
an audit hook confirming no Test-file opens. Exact checkpoint roundtrip exposed
SDPA/eager mismatch; persisting encoder attention implementation fixed it.

Independent gpt-6-astra review: no Critical/Important issues. Deferred minor:
fault-injection tests for interruption and input mutation during training;
the reviewed failure handling correctly prevents completed status.

Real DeBERTa GPU smoke passed on both datasets (16 Train aspects, two updates,
full Dev evaluation and exact checkpoint reload). Every boundary/input/source
invariant passed. Smoke scores are not learning-performance evidence.
Artifacts: outputs/target_context_smoke_20261017.

Full Train/Dev tokenization preflight: T15 3179/1122 aspects, T17 3562/1176;
maximum marked lengths 50/51 tokens, no truncation needed.

Formal launcher: outputs/target_context_control_20261017/run_pilot.py.
Output: outputs/target_context_s42_20261017. Two concurrent dataset jobs,
one seed each; no extra seed or Test stage. comparison.json records stop versus
eligibility to consider replication after both jobs pass reload verification.

Execution decisions: preserve the user-selected shared branch; GPU compatibility
was gated by smoke; improvement awaits completed formal results. Fixed CLI paths
define the pilot's Train/Dev boundary (arbitrary renamed data passed directly to
Python API is outside scope). No resume support; a hard-killed run remains
non-completed and needs a fresh output. This costs restart work if interrupted.

Launch confirmed 2026-10-17 12:07 Asia/Shanghai: scheduler PID 4616; T15/GPU0
PID 4705 and T17/GPU1 PID 4706. Both completed epoch 1/35, finite loss and
unchanged extraction metrics. First epochs took 34.09/36.93 seconds; estimated
completion around 12:30 including reload (estimate, not a completion claim).
Live status is outputs/target_context_s42_20261017/batch-status.json plus each
dataset's seed42.status.json. Execution ledger/test evidence archived under
outputs/target_context_control_20261017/implementation-audit.

## 完成结果与当前主力候选决策（2026-10-17）

后续状态：2026-10-18 用户已确认固定等权概率融合为主力候选；本节的分类器单独
替换方案现保留为对照二，原 MACT 为对照一。详见[后续正式决策](2026-10-18-calibrated-fusion.md#confirmed-main-candidate-2026-10-18)。
以下保留2026-10-17的决策及其当时依据。

用户在完成 seed42 Test 和 seeds43/44 配对复现后明确同意：
**“MACT＋目标上下文情感分类器”作为当前主力候选，保留原 MACT 作对照。**
该决策确定后续实验主线，不改变原 baseline 配置及历史实验定义。
候选是原 MACT 预测边界加独立目标情感重编码器，不包含 EMA、LLRD 或 OCR 分支。

三个 seed 的 Test Joint F1（均值 ± 样本标准差）：

| 数据集 | 原 MACT | 主力候选 | 平均配对提升 |
|---|---:|---:|---:|
| Twitter2015 | 67.81 ± 0.90 | 68.19 ± 0.76 | +0.38 pp |
| Twitter2017 | 67.91 ± 0.32 | 69.51 ± 1.39 | +1.60 pp |

Test 配对增益依 seed42/43/44 为 T15 +0.38/+0.19/+0.57 pp，
T17 +0.32/+1.28/+3.21 pp。Dev 平均增益分别 +0.38/+0.47 pp，
但 seed44 Dev 分别下降 0.80/0.67 pp。保留这种不一致及 T17 的较大波动，
不将三 seed 的一致 Test 方向表述为统计显著或普遍稳定提升。
后续实验继续保留同 seed 原 MACT 对照，使用 Dev 选模，不根据 Test 调参。
补充 seed43/44 是观察 seed42 Test 后、保持配置不变的复现实验。

全部训练各35轮，完整 Dev 重载验证及 Test 独立指标核验通过，方面边界保持一致。
补充实验于 2026-10-17 14:13:03 Asia/Shanghai 完成，耗时43.44分钟。

产物（相对项目根目录）：

- 原 MACT 配对权重：`outputs/llrd_ema_s42_43_44_20261016/uniform/{dataset}/seed{seed}/best.pt`。
- 候选 seed42 分类器：`outputs/target_context_s42_20261017/{dataset}/seed42/best.pt`。
- 候选 seeds43/44 分类器：`outputs/target_context_s43_44_20261017/{dataset}/seed{seed}/best.pt`。
- 各 seed Test：`outputs/target_context_test_s{seed}_20261017/`。
- 三 seed 汇总：`outputs/target_context_multiseed_control_20261017/comparison.json`。
- 独立核验：同目录 `postrun-audit.json`。

候选推理需要配套的原 MACT 权重、目标分类器权重及各自 tokenizer。
后续待办为推理成本测量和情感错误分析；本次确认未自动启动新训练或增加模块。

## 历史三组12 seed 扩展结果（2026-10-17）

用户随后授权按原顺序执行三组：24/25/17/16，31/22/27/26，30/28/23/19。
每组覆盖两个数据集，完成训练、Dev重载及固定checkpoint的Test后再进入下一组。
全部24次训练及24次Test于19:00:52（Asia/Shanghai）完成，总耗时243.90分钟。
候选配置保持不变，使用各seed配对的原MACT预测边界；缺失历史权重从原实验
复制到项目内并核验SHA256。完整Dev/Test指标独立复算、选模轮次、checkpoint
哈希、35轮预算及边界不变检查全部通过。

| 数据集 | 原MACT Test均值±样本标准差 | 候选Test均值±样本标准差 | 配对平均提升 | 胜/负 |
|---|---:|---:|---:|---:|
| Twitter2015 | 68.48 ± 0.79 | 68.26 ± 1.07 | −0.22 pp | 5/7 |
| Twitter2017 | 68.76 ± 0.90 | 68.55 ± 0.81 | −0.22 pp | 5/7 |

| 分组 | Twitter2015平均Test增益 | Twitter2017平均Test增益 |
|---|---:|---:|
| 第一组24/25/17/16 | −0.53 pp | −0.79 pp |
| 第二组31/22/27/26 | −0.50 pp | +0.16 pp |
| 第三组30/28/23/19 | +0.38 pp | −0.02 pp |

Dev平均分别由67.42→67.77（+0.35 pp）、67.17→68.06（+0.89 pp），
但收益没有稳定转化到Test。该轮不支持此前从三个seed推及广泛稳定改善；
也不据此断言该结构在所有配置上无效。历史seed本身按过往Dev/Test结果选取，
是有选择的历史复核，与42/43/44保持分开报告，不混成无偏的15-seed样本。
本次仅更新证据；未根据Test更改参数或启动新训练。

原始输出：`outputs/target_context_historical12_20261017/`。
`comparison.json`含所有24行分数；`group1/2/3-comparison.json`含各组结果；
`postrun-audit.json`为独立核验。可复运行的审计脚本位于
`outputs/target_context_historical12_control_20261017/audit_results.py`。
