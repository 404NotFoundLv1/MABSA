# OOF Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans inline, as previously selected by the user.

**Goal:** Implement, verify and launch the three-fold correction pilot end to end.
**Architecture:** Grouped Train-only fold preparation, existing encoder trainers,
verified fold feature aggregation, shared correction fitting, sequential queue.
**Tech Stack:** Existing Python/PyTorch/transformers, no new dependencies.
**Spec:** docs/superpowers/specs/2026-10-18-oof-correction.md

## Global Constraints

- Existing branch; preserve original artifacts. Seed42, 3 folds, both datasets.
- No outer-fold checkpoint selection, full-Train warm starts, Test or auto new seeds.
- Original 35-epoch experts; identical 30-epoch correction arms and controls.
- Exact original IDs and grouped text/image separation. Fresh/hash-verified stages.

## Review Focus

- Outer-fold samples or duplicates leaking into either expert's training/selection.
- Fold models initialized from full-Train task checkpoints, or checkpoint mismatch.
- Mixed cache formats, dropped false positives/sentences or stale stage reuse.
- Feature-space drift and smaller fold training sets hidden in interpretation.
- Detached queue claiming completion after failed/partial stages or source mutation.

### Task 1: Fold membership and validation

Files: src/mact/oof_data.py, tests/test_oof_data.py.
Interfaces: make_partition(examples,image_hashes); prepare_dataset(dataset,root);
verify_partition(examples,partition,image_hashes); validate_oof_payload(cache).
- [ ] Write failing tests for deterministic transitive groups, label-independent
  assignment, held-out/inner/fit disjointness, preserved IDs and corrupted membership.
- [ ] Implement grouped partition, raw-record materialization and strict checks.
- [ ] Run focused tests; commit.

### Task 2: Verified fitting and executable queue

Files: src/mact/oof_correction.py, scripts/run_oof_correction.py,
src/mact/correction_experiment.py, tests/test_oof_correction.py.
Interfaces: run_fold, aggregate_oof, fit_oof_pair, run_pipeline.
- [ ] Add failing tests for explicit OOF validation, full tiny encoder integration,
  malformed/incomplete/stale folds and ordinary-cache regression.
- [ ] Factor the unchanged fitter behind validated inputs; keep old format strict.
- [ ] Implement fresh pretrained fold training, inner replay, outer extraction,
  aggregation, original control replay, paired fitting and independent tuple audit.
- [ ] Add detached sequential queue, locked protocol and fail-fast status.
- [ ] Run full suite and one fresh whole-change code review; fix substantive findings.

### Task 3: Launch and verify live execution

- [ ] Materialize both dataset partitions and lock all identities/configuration.
- [ ] Start queue on an available GPU, confirm real optimizer/epoch progress.
- [ ] Record PID, order, measured runtime estimate and exact remaining work.
- [ ] If the queue finishes in-session, inspect/audit results; otherwise report it
  accurately as running with automatic downstream stages and persistent status.

## Execution rulings

The user authorized the previously proposed OOF next step and inline execution;
routine fold count/inner split choices are made concrete here without repeating
approval. Three folds bound cost; inner validation prevents using outer labels for
selection. Costs: smaller fold training sets and possible feature-space mismatch.
