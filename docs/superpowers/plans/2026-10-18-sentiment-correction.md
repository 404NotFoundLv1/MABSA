# Target-conditioned Sentiment Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline. User has already selected this execution method.

**Goal:** Implement and run the approved frozen-feature correction pilot on both datasets at seed42.
**Architecture:** Add independent feature extraction and residual-head training; reuse existing frozen MACT/target models and fixed fusion. Compare plain and bounded corrections under identical optimization.
**Tech Stack:** Existing Python3.11, PyTorch, transformers; no added runtime dependencies.
**Spec:** docs/superpowers/specs/2026-10-18-sentiment-correction.md

## Global Constraints

- Existing branch/source tree, no worktree, no changes to original numerical fusion or model weights.
- Only Train/Dev; seed42, two datasets; no Test, no seed expansion or tuning after results.
- Two identical 128-hidden heads; plain vs 2*tanh(raw/2); zero final linear initialization.
- 30 epochs, batch64, AdamW .001, decay .01, clip1, dropout .1, residual L2 .01.
- All exact-matched Train predictions get CE; no error-conditioned selection or targets.
- Keep the user's uncommitted model-diagram artifacts intact.

## Review Focus

- Gold annotations leaking into feature extraction or residual inference.
- False-positive and no-aspect sentences changing boundary sets or metric denominators.
- Identical initialization, epoch0 fallback, and score ties concealing real behavior.
- Stale/mixed Train/Dev caches or checkpoint identities bypassing provenance checks.
- Training outputs from in-sample representations being misreported as held-out error supervision.

### Task 1: Frozen features and correction head

Files: src/mact/sentiment_correction.py, tests/test_sentiment_correction.py.
Interfaces: CorrectionHead(base_dim,target_dim,variant,hidden=128,dropout=.1);
anchor_scores(base_logits,target_logits); corrected_labels(scores,fixed_labels);
extract_features(base_model,target_model,examples,base_tokenizer,target_tokenizer,base_config,features,device).

- [x] Add failing tests for zero initialization, bounded corrections, shared-error override and finite gradients.
- [x] Add real tiny multimodal encoder test: changed gold cannot change features/predictions; boundaries remain identical; hook cleanup.
- [x] Implement numerical head and frozen feature extraction without modifying old models.
- [x] Run tests and commit tested implementation.

### Task 2: Provenance-checked pilot runner

Files: src/mact/correction_experiment.py, scripts/run_sentiment_correction.py, tests/test_correction_experiment.py.
Interfaces: extract_pilot(dataset,seed,device,root); train_pair(train_cache,dev_cache,output,config).

- [x] Add failing tests using controlled feature caches: split rejection, disjoint IDs, stale hashes, full reload and unchanged boundaries.
- [x] Implement two-split extraction and Train-only paired head training with 30-epoch history/status and epoch0 fallback.
- [x] Validate archived Dev controls, input/source hashes, all metrics and head reloads.
- [x] Run full suite; get one fresh whole-change review and fix substantive findings before GPU runs.

### Task 3: Execute and report

- [x] Lock protocol, sources and checkpoints before extracting real features.
- [x] Extract seed42 Train/Dev on the two datasets, then train both paired heads on CPU.
- [x] Independently recompute saved predictions, gains/regressions and the advancement gate.
- [x] Record results and limitations; keep fixed fusion as main candidate pending successful validation.

## Execution record

Base: 7199803. User's “开始” authorizes this pilot; continue inline without repeated permission requests.
Ordinary Train CE does not require correctness-based OOF targets. Frozen encoders
already saw Train; report saturation and treat this as an exploratory head pilot.
The tested difference is a correction limit, not an established novel mechanism.


## Completed results (2026-10-18)

Implementation: 835ffbf and 1eafe41. Full suite: 165 tests passed. One fresh
independent code review reported no substantiated findings. Real seed42 extraction
and 30 epochs per arm completed sequentially on both datasets. Both archived Dev
predictions and metrics reproduced exactly; maximum base-score difference was
1.90735e-6, within the locked 1e-5 tolerance; target scores were identical.
Independent saved-head inference and tuple audit passed.

Dev Joint F1 (%), showing best **trained** epochs separately from fallback:

| Model | Twitter2015 | Twitter2017 |
|---|---:|---:|
| Original MACT | 67.3407 | 67.2537 |
| Target classifier replacement | 68.2198 | 68.5115 |
| Fixed equal fusion | 68.0440 | 69.2662 |
| Plain correction | 68.5714 (epoch30) | 68.7631 (epoch1) |
| Bounded correction | 68.4835 (epoch1) | 68.6792 (epoch1) |

Both 2017 arms select epoch0, exactly preserving fixed fusion. This is fallback,
not improvement. At best trained epochs, plain/bounded wrong-to-correct versus
correct-to-wrong counts are 20/14 and 21/16 on 2015; 14/20 and 14/21 on 2017.
Shared-error rescues are zero in all four selected trained heads.

The advancement gate FAILED: no 2017 gain, and bounded is worse than plain on
2015. Keep fixed equal fusion as the main candidate. No Test evaluation, seed
expansion or post-result hyperparameter tuning was performed.

Train fixed-fusion matched sentiment accuracy was 3178/3179 (99.9685%) on 2015
and 3553/3556 (99.9156%) on 2017. Saturation is a plausible limitation of fitting
correction heads on in-sample frozen features, not a proven cause of the result.
This ordinary-CE feasibility pilot does not establish OOF error supervision,
independent generalization, significance or novelty of residual limits.

Artifacts: outputs/sentiment_correction_s42_20261018/SUMMARY.txt,
comparison.json, independent-audit.json, independent_audit.py, and each dataset's
features/{protocol.json,dev_replay.json,train.pt,dev.pt}, fit/result.json,
fit/{plain,bounded}/{best.pt,best_trained.pt,history.json,predictions.json,
best_trained_predictions.json}. The output directory is local/ignored by Git.

Execution rulings: use all matched Train examples with ordinary CE (disclosed
in-sample limitation); retain the local ledger (small disk cost); resolve the
reviewer's deferred runtime/replay/empirical checks through the completed pilot;
leave generalization/significance/novelty/Test unclaimed (further evidence needed).
There are no deferred minor code findings. Existing branch and diagram artifacts
are preserved according to the user's prior choice.
