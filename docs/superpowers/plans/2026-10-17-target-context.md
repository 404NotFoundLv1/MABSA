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
