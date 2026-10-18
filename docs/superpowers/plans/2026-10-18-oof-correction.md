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
- [x] Write failing tests for deterministic transitive groups, label-independent
  assignment, held-out/inner/fit disjointness, preserved IDs and corrupted membership.
- [x] Implement grouped partition, raw-record materialization and strict checks.
- [x] Run focused tests; commit.

### Task 2: Verified fitting and executable queue

Files: src/mact/oof_correction.py, scripts/run_oof_correction.py,
src/mact/correction_experiment.py, tests/test_oof_correction.py.
Interfaces: run_fold, aggregate_oof, fit_oof_pair, run_pipeline.
- [x] Add failing tests for explicit OOF validation, full tiny encoder integration,
  malformed/incomplete/stale folds and ordinary-cache regression.
- [x] Factor the unchanged fitter behind validated inputs; keep old format strict.
- [x] Implement fresh pretrained fold training, inner replay, outer extraction,
  aggregation, original control replay, paired fitting and independent tuple audit.
- [x] Add detached sequential queue, locked protocol and fail-fast status.
- [x] Run full suite and one fresh whole-change code review; fix substantive findings.

### Task 3: Launch and verify live execution

- [x] Materialize both dataset partitions and lock all identities/configuration.
- [x] Start queue on an available GPU, confirm real optimizer/epoch progress.
- [x] Record PID, order, measured runtime estimate and exact remaining work.
- [x] If the queue finishes in-session, inspect/audit results; otherwise report it
  accurately as running with automatic downstream stages and persistent status.

## Execution rulings

The user authorized the previously proposed OOF next step and inline execution;
routine fold count/inner split choices are made concrete here without repeating
approval. Three folds bound cost; inner validation prevents using outer labels for
selection. Costs: smaller fold training sets and possible feature-space mismatch.


## Launch verification and remaining experiment work

Implementation and launch tasks completed, **experiment still running**.
Commits e0b3ad7, e2e7f43, 09d54ee; 173 tests passed. One fresh code review led to
three verified fixes: completed-fit provenance, discrete control replay instead
of exact diagnostic floats, and immutable prior-reference fingerprints. No deferred
minor findings. Grouping and initialization were independently reviewed.

Detached PID15390 on cuda:0, root outputs/oof_correction_s42_20261018.
First Twitter2015 fold0 MACT epoch completed:1260 examples,158 updates,99.997s
including startup. Now epoch2/35 at launch verification. Rough remaining estimate
5–7 hours, based on first epoch and historical target timings; not a deadline.
All six fold pairs (12 expert trainings), feature generation, original controls,
paired head fitting and audits are queued. Results are NOT yet available and
fixed equal fusion remains the main candidate. No Test or new seeds authorized
by this queue. Persistent status.json records completion/failure.

2015 fold fit/inner/held counts:1260/140/701,1261/140/700,1260/141/700.
2017:1047/117/582,1045/119/582,1047/117/582. Exact text/image grouping found
2099 groups for2101 sentences and1740 groups for1746 sentences, respectively.
Original IDs are retained.

Rulings:3fold+10% inner groups bounds cost but reduces expert training size and
can introduce representation drift; preserve existing workspace/ledger per user;
record diagnostic float drift while demanding exact selected predictions and
metrics (not a bitwise optimization replay); live launch verifies GPU execution,
while final GPU replay/results and statistical conclusions await actual completion
and later validation. No claims of significance, generalization or novelty.
