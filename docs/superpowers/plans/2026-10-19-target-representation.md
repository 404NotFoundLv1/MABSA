# Target Representation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch a controlled six-job Train/Dev A/B/C representation pilot.

**Architecture:** Add isolated representation dataset/model/checkpoint code and a
fusion-selected training runner, reusing verified baseline inference and metrics.
The legacy model/trainer and current main checkpoints remain unchanged.

**Tech Stack:** Existing Python 3.11 runtime, PyTorch, Transformers, unittest/pytest.

**Spec:** docs/superpowers/specs/2026-10-19-target-representation.md

## Global Constraints

Seed42; 35 epochs; CE only; same initialization and training settings as spec;
frozen MACT; fixed 0.5 probability fusion; no Test reads; six sequential jobs;
earliest maximum full predicted-boundary fusion Dev F1; new checkpoint format.
Remain in current checkout per session preference. Preserve unrelated changes.

## Review Focus

1. Multi-subword/repeated targets, unknown words, target-only sentences: correct masks, zero empty context.
2. Gold/polarity contamination, empty sentences, false positives: inference independent of labels and immutable spans.
3. Marker control equivalence and gradient flow: A identical; B/C train encoder and new parameters.
4. Stale/mismatched score cache, checkpoint, source/config, partial batch restart: reject, never silently reuse.
5. Selection/reload/aggregation: fusion metric alone, earliest tie, full Dev replay and correct paired comparisons.

### Task 1: Representation model and masks

**Files:** Create src/mact/target_representation.py, tests/test_target_representation.py.
**Interfaces:** RepresentationDataset/Collator retain legacy keys and add span_weights/context_mask;
RepresentationClassifier(encoder, variant, dropout=.1, attention_dim=128),
save_representation(path,model,metadata), load_representation(path,device).
- [ ] Write tests for exact repeated-target word weights, context exclusions/UNK,
  zero empty context, gold invariance, padding, A equivalence, gradients, checkpoint replay.
- [ ] Run tests; expected FAIL on missing implementation.
- [ ] Implement isolated dataset/model/checkpoint APIs and validation.
- [ ] Run focused tests and existing target tests; expected PASS. Commit task.

### Task 2: Verified fusion-selected training and diagnostics

**Files:** Create src/mact/representation_experiment.py,
src/mact/representation_diagnostic.py, tests/test_representation_experiment.py.
**Interfaces:** run_training(config,output); evaluate_scores(cache,logits,gold);
pilot_config(dataset,variant,score_cache,smoke=False); diagnose(examples,cache).
- [ ] Write real tiny-transformer training/reload tests, fusion-selection/tie tests,
  Test open guard, cache/config identity rejection, error-count and diagnostic tests.
- [ ] Run tests; expected FAIL on missing implementation.
- [ ] Implement same optimizer schedule as old trainer, versioned checkpoints,
  verified cache ordering/data identity, masks, source/input hash checks, full-Dev
  prediction replay, complete status and input-safe descriptive error diagnostics.
- [ ] Run new and existing fusion/target tests; expected PASS. Commit task.

### Task 3: Sequential batch, review and launch

**Files:** Create scripts/run_target_representation.py and runner tests.
**Interfaces:** CLI --output --smoke --device; six sequential subprocess jobs,
fresh full-Dev score extraction, frozen decision manifest, completion aggregation.
- [ ] Write tests for schedule, incomplete/stale job refusal and paired summary.
- [ ] Observe FAIL; implement batch/aggregation; observe PASS.
- [ ] Run entire suite; expected PASS, no missing/unreported failures.
- [ ] Fresh independent review of task diff, spec and failure modes; fix important
  findings with regressions, run final suite. Commit implementation.
- [ ] Run all A/B/C real-backbone small Train smoke + full Dev reload; expected PASS.
- [ ] Inspect deterministic Dev shared-error examples and Train statistics. Save
  human-readable assessment without treating cue flags as causal labels.
- [ ] Launch six full sequential jobs with logs and status; verify real progress,
  estimate runtime from observed epoch times; report location and status.
