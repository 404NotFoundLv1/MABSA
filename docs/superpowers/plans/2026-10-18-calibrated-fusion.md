# Calibrated fusion implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate calibrated two-expert fusion on all completed paired Dev runs.
**Architecture:** A pure score/calibration module, an extraction runner using
existing model loaders, and a grouped cross-fitting evaluator with fixed policy.
**Tech Stack:** Existing Python, PyTorch, Transformers, unittest; no dependencies.
**Spec:** docs/superpowers/specs/2026-10-18-calibrated-fusion.md

## Global Constraints

- Dev only, original boundaries and archives unchanged, no encoder training.
- Same five sentence folds across all 15 seeds; dataset-level parameters.
- Temperatures [0.25,32]; alpha [0,.25,.5,.75,1]; exact control endpoints.
- Report checkpoint-selection and historical-seed limitations explicitly.
- Existing current-tree inline execution is the user's preserved choice.

## Review Focus

- CRF endpoints and immediately adjacent aspects must include all transitions.
- Probability ties and no-aspect sentences must preserve control outputs.
- Held-out labels must not influence that fold's calibration or alpha.
- False positives remain in F1; unmatched gold remains in recall denominator.
- Missing, duplicated, corrupted or stale seed caches must fail closed.

### Task 1: Conditional scores and calibrated decisions

**Files:** Create src/mact/calibrated_fusion.py, tests/test_calibrated_fusion.py.
**Interfaces:** aspect_polarity_logits(crf, emissions, aspects) -> tensor [N,3];
fit_temperature(logits, labels) -> float; fuse_labels(base_logits,target_logits,
baseline_labels,target_labels,temperature_base,temperature_target,alpha) -> tensor.

- [ ] Write tests comparing conditional logits with exhaustive path scores,
  known temperature optimum, invalid inputs, and exact endpoint/tie outputs.
- [ ] Run unittest discovery for this file; expect missing implementation failure.
- [ ] Implement pure functions; run same tests, expect all pass.
- [ ] Commit task 1 sources and tests.

### Task 2: Grouped cross-fitting and verified extraction

**Files:** Create src/mact/fusion_experiment.py, scripts/run_calibrated_fusion.py,
tests/test_fusion_experiment.py.
**Interfaces:** sentence_fold(id) -> int; crossfit(runs) -> report and predictions;
extract(dataset,seed,device,output) -> verified Dev cache. Reuse task 1 functions.

- [ ] Test sentence grouping, held-out-label exclusion, full F1 denominators,
  strict identity/duplicate validation and extraction with tiny real models.
- [ ] Run file tests; expect missing implementation failure.
- [ ] Implement extraction and fixed cross-fold policy; run both test files.
- [ ] Run full unittest suite; expect all pass; commit implementation.

### Task 3: Review and execute the approved Dev experiment

**Files:** outputs/calibrated_fusion_20261018/ artifacts; README.md results note.
**Interfaces:** CLI extract --dataset --seed --device --output; CLI summarize
--root requiring all expected seed caches, producing locked parameters/report.

- [ ] Fresh whole-change reviewer checks spec, tests and experiment integrity.
- [ ] Resolve important findings with reproducing tests and verify suite.
- [ ] Run one extraction per GPU for each dataset, seeds sequentially, then
  aggregate all 30 verified caches and inspect paired cross-fitted results.
- [ ] Record results and gate decision; do not tune after seeing Test.
- [ ] Verify outputs, commit final documentation, report evidence and limits.
