# Frozen OCR residual: paired seeds 43 and 44

User approved the next step: replicate the exact seed42 treatment on seeds 43/44, Twitter2015/2017, 35 epochs each, Dev only. Original results and official splits remain untouched. Four concurrent jobs, two per GPU; no extra model modules or hyperparameter changes.

Implementation: add an optional seed override to the shared trainer, and a seed-aware baseline path template in a new configuration. Generalize the existing detached pilot launcher to multiple seeds and an optional prior-results directory. Keep seed42 defaults and old outputs. Each job must first match its own archived full Dev predictions and baseline checkpoint hash. Retain epoch-only logging and existing removal/shuffle/no-OCR/frozen-weight/reload checks. No Test reads or evaluation.

Tasks:
1. Tests for seed overrides, per-seed baseline resolution, seed bounds, matching prior treatment settings, paired aggregation with missing/duplicate protection.
2. Shared trainer/launcher changes; independent read-only review, full suite, seed42 core computation parity against original commit 2d2a783.
3. Import only original weights whose hashes match archived seed43/44 summaries. Final-source GPU smoke for all four combinations. Commit source and launch four jobs automatically followed by paired three-seed mean/sample-SD, deltas and wins, with individual diagnostics retained.

Review focus: same-seed frozen weights, no accidental reuse of seed42 baseline, original seed42 behavior preserved, no tuning or epoch8 fixation, no Test access, no partial aggregate presented as complete, correct GPU and seed arguments, detached process survival.

Ledger: approved design inherited from prior recommendation. Existing feature branch/shared source folder satisfies user preference. Original checkpoint files found under mabsa_evidence/experiments/20261011_sequence_crf_35ep_multiseed/runs.

Progress:
- Shared trainer seed override, seed-aware configuration, generalized existing launcher and strict paired aggregation implemented. No source copy per experiment.
- 106 tests pass. Independent AST and file-hash comparison against 2d2a783 confirms original model computation, training loop and diagnostics are unchanged. Both prior seed42 runs pass metric recomputation and same-treatment verification.
- Four original baseline checkpoints copied only after source hashes match their archived summaries; copied hashes rechecked. Original files unchanged.
- Fresh independent review found no blocking issue; six targeted tests independently passed. Four-job GPU assignments, same-seed baseline checks and paired sample-SD computation verified.
- Review ruling: retain existing generic hash-failure text mentioning seed42 so the old training-loop AST stays unchanged; actual validation always uses the requested seed and its archive. Cost: an integrity failure may show stale wording, but it cannot accept the wrong checkpoint.
- Review ruling: preflight does not separately prohibit a CPU-generated smoke artifact. This launch will use four explicit CUDA smokes and verify their positive GPU-memory measurements before launching; that supplies the required evidence for this run. No CPU fallback is permitted when CUDA is requested.
- Remaining: final-source four-job GPU smoke, source commit, detached launch and actual full Dev identity checks before reporting training started. No Test evaluation.
- GPU verification correction: all four initial smokes completed with frozen/reload checks passing, but the old preflight assumed exactly three optimizer updates. Twitter2017 seed43 has OCR counts [3,0,6] in its three accumulation windows, so two updates are correct. Regression tests reproduced the rejection; validation now requires three completed windows, positive updates no greater than windows, one epoch, and positive CUDA memory. CPU smokes are now explicitly rejected. Training code remains unchanged. Preserve first smoke artifacts and rerun final-source smokes under final-v2.
