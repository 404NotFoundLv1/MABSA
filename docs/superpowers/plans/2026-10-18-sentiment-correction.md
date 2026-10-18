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

- [ ] Add failing tests for zero initialization, bounded corrections, shared-error override and finite gradients.
- [ ] Add real tiny multimodal encoder test: changed gold cannot change features/predictions; boundaries remain identical; hook cleanup.
- [ ] Implement numerical head and frozen feature extraction without modifying old models.
- [ ] Run tests and commit tested implementation.

### Task 2: Provenance-checked pilot runner

Files: src/mact/correction_experiment.py, scripts/run_sentiment_correction.py, tests/test_correction_experiment.py.
Interfaces: extract_pilot(dataset,seed,device,root); train_pair(train_cache,dev_cache,output,config).

- [ ] Add failing tests using controlled feature caches: split rejection, disjoint IDs, stale hashes, full reload and unchanged boundaries.
- [ ] Implement two-split extraction and Train-only paired head training with 30-epoch history/status and epoch0 fallback.
- [ ] Validate archived Dev controls, input/source hashes, all metrics and head reloads.
- [ ] Run full suite; get one fresh whole-change review and fix substantive findings before GPU runs.

### Task 3: Execute and report

- [ ] Lock protocol, sources and checkpoints before extracting real features.
- [ ] Extract seed42 Train/Dev on the two datasets, then train both paired heads on CPU.
- [ ] Independently recompute saved predictions, gains/regressions and the advancement gate.
- [ ] Record results and limitations; keep fixed fusion as main candidate pending successful validation.

## Execution record

Base: 7199803. User's “开始” authorizes this pilot; continue inline without repeated permission requests.
Ordinary Train CE does not require correctness-based OOF targets. Frozen encoders
already saw Train; report saturation and treat this as an exploratory head pilot.
The tested difference is a correction limit, not an established novel mechanism.
