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

- [x] Write tests comparing conditional logits with exhaustive path scores,
  known temperature optimum, invalid inputs, and exact endpoint/tie outputs.
- [x] Run unittest discovery for this file; expect missing implementation failure.
- [x] Implement pure functions; run same tests, expect all pass.
- [x] Commit task 1 sources and tests.

### Task 2: Grouped cross-fitting and verified extraction

**Files:** Create src/mact/fusion_experiment.py, scripts/run_calibrated_fusion.py,
tests/test_fusion_experiment.py.
**Interfaces:** sentence_fold(id) -> int; crossfit(runs) -> report and predictions;
extract(dataset,seed,device,output) -> verified Dev cache. Reuse task 1 functions.

- [x] Test sentence grouping, held-out-label exclusion, full F1 denominators,
  strict identity/duplicate validation and extraction with tiny real models.
- [x] Run file tests; expect missing implementation failure.
- [x] Implement extraction and fixed cross-fold policy; run both test files.
- [x] Run full unittest suite; expect all pass; commit implementation.

### Task 3: Review and execute the approved Dev experiment

**Files:** outputs/calibrated_fusion_20261018/ artifacts; README.md results note.
**Interfaces:** CLI extract --dataset --seed --device --root; CLI summarize
--root requiring all expected seed caches, producing locked parameters/report.

- [x] Fresh whole-change reviewer checks spec, tests and experiment integrity.
- [x] Resolve important findings with reproducing tests and verify suite.
- [x] Run one extraction per GPU for each dataset, seeds sequentially, then
  aggregate all 30 verified caches and inspect paired cross-fitted results.
- [x] Record results and gate decision; do not tune after seeing Test.
- [x] Verify outputs, commit final documentation, report evidence and limits.

## Execution record

Inline in the existing `ocr-context` branch, as previously chosen by the user.
Core implementation commits: `5d9d77c`, `c608293`; review fixes: `237c7f7`.
TDD: 5 core tests and 4 initial integration tests first failed for missing
implementations, then passed. Fresh independent gpt-6-astra whole-change review
found no critical issues and two important issues. Regressions reproduced both
misplaced-dataset cache acceptance and incomplete Test-artifact blocking; fixes
passed the entire **154-test** suite. No baseline source or config was changed.

The CLI consistently uses `--root` for extraction and summary. The exact
commands for the approved run are:

```bash
.runtime/bin/python scripts/run_calibrated_fusion.py extract --root outputs/calibrated_fusion_20261018 --dataset twitter2015 --device cuda:0
.runtime/bin/python scripts/run_calibrated_fusion.py extract --root outputs/calibrated_fusion_20261018 --dataset twitter2017 --device cuda:1
.runtime/bin/python scripts/run_calibrated_fusion.py summarize --root outputs/calibrated_fusion_20261018
```

Review limitation deferred: the internal `extract_run(resources={})` multimodal
fallback does not fingerprint the effective visual cache. The supported CLI
used here always supplies its explicit visual-cache path and fingerprints both
manifest and features. Use the CLI for this experiment; the fallback path needs
fingerprint completion before separate use.

Decisions: preserve the current tree/branch; use the unified `--root` interface;
judge GPU behavior and measured gains from the completed run; retain the
exploratory limitation of Dev-selected checkpoints/historical seeds; treat
checksum sidecars as accidental-corruption checks, not authenticated provenance.
These do not authorize merge/push, create independent validation, or promise
generalization to Test. The review ledger is retained locally for audit.

## Completed Dev experiment

All 30 score extractions completed with exact baseline/target replays. Both GPU
processes and summary exited 0. Postrun audit recomputed all 30 full-Dev metric
triples (baseline/target/cross-fitted fusion) and preserved boundary sets.

| All15 Dev F1 (mean ± sample SD) | Twitter2015 | Twitter2017 |
|---|---:|---:|
| Baseline | 67.4491 ± 0.4221 | 67.1443 ± 0.7382 |
| Target only | 67.8078 ± 0.5699 | 67.9548 ± 0.9209 |
| Raw fixed half mixture | 68.0373 ± 0.5684 | 68.1259 ± 0.9480 |
| Cross-fitted calibrated selection | 67.5899 ± 0.7080 | 68.1316 ± 0.9958 |

Calibrated minus target: -0.2179/+0.1768 pp; 4/2/9 and 11/0/4 wins/ties/losses.
2015's fold-selected target weights vary .25/.5/.75; 2017 consistently selects .5.
Full-Dev refit is archived separately and not substituted for held-out scores.
The preregistered advancement gate is **false**; no Test run or new training.

Raw fixed .5 control minus target: +0.2295/+0.1711 pp, 9/1/5 and 11/0/4
wins/ties/losses. Versus baseline it improves 14/15 seeds on each dataset.
This is a simpler candidate for a subsequent explicitly fixed evaluation,
not a retroactive pass of the calibrated experiment's gate. No further rules
were searched after seeing these results. The primary candidate designation
remains unchanged.

Artifacts: outputs/calibrated_fusion_20261018/{comparison.json,SUMMARY.txt,
per_seed.csv,postrun-audit.json}; original raw reports, score caches, SHA sidecars,
and logs retained. The initial3 and historical12 cohorts are separately reported.

## Authorized follow-up: fixed equal-fusion Test evaluation

The user explicitly said “继续下一步” after the recommendation to evaluate the
raw fixed 1:1 mixture. This authorizes a separate Test evaluation; it does not
change the failed calibrated variant's gate or permit Test-based tuning.

Before the new Test run, outputs/fixed_equal_fusion_test_20261018/decision.json
locked both temperatures to 1, alpha to .5, both datasets and all 15 seeds,
30 paired checkpoint hashes, target config hashes, and the unchanged numerical
fusion module hash. Initial decision SHA256:
`ef6d90acf0abc1bafc0dfbb4cb48d8b875add0a3332d20d9312562c880bd64ba`.

Implementation is bounded: add src/mact/fusion_test.py and
scripts/evaluate_fixed_fusion.py, preserving the Dev entry point and access
guard. Remove all gold annotations from inference examples; use frozen predicted
boundaries. Verify both controls against their previously archived full Test
predictions and metrics. Generate fusion predictions with exactly the existing
raw-half numerical rule. Never fit temperatures, search weights, choose seeds,
train encoders or select checkpoints using Test. Hash the effective visual cache
in the new evaluator, including when inherited from checkpoint configuration.

Tests: initial four tests failed for missing implementation, then the full
158-test suite passed. Added real tiny-checkpoint evaluation, changed-gold
prediction invariance, locked-parameter rejection, replay failures, missing
cohorts and misplaced result identities. A synthetic perfectly symmetric score
fixture was made unambiguous after measuring a 5.55e-17 softmax permutation
difference; the frozen fusion algorithm was not changed.

Run one dataset per GPU, all seeds sequentially. Require all 30 pairs before
aggregation. Recompute saved prediction metrics and boundaries; report all15,
initial3 and historical12 means, sample SD and paired win/tie/loss counts.
Earlier Test results and historical seed selection are known, so this is not a
claim of performance on an untouched benchmark. Keep current branch; no push.

```bash
.runtime/bin/python scripts/evaluate_fixed_fusion.py run --root outputs/fixed_equal_fusion_test_20261018 --dataset twitter2015 --device cuda:0
.runtime/bin/python scripts/evaluate_fixed_fusion.py run --root outputs/fixed_equal_fusion_test_20261018 --dataset twitter2017 --device cuda:1
.runtime/bin/python scripts/evaluate_fixed_fusion.py summarize --root outputs/fixed_equal_fusion_test_20261018
```

Follow-up review: a fresh independent reviewer found no Critical or Important
issues, reran the four new tests, checked all 30 archive mappings and recomputed
their archived metrics. Minor coverage gap deferred: the new tiny fixtures use
text-only baselines, so effective visual-cache metadata fallback lacks its own
small regression test. The real experiment's postrun audit explicitly checks
manifest/features fingerprints for all 30 multimodal evaluations. Actual GPU
replay is judged from the run, while earlier training/seed selection remains
an unchanged upstream limitation. No existing Dev protection was weakened.

## Completed fixed equal-fusion Test evaluation

Both GPU processes, aggregation and the postrun report/audit exited 0. All 30
paired evaluations completed under the locked decision SHA above, with source
implementation frozen at af71feb. Both archived controls replayed exactly;
all aspect boundaries, input hashes and source hashes passed verification.
The full implementation suite passed 158 tests before these runs. Subsequent
changes are documentation only.

| All15 Test Joint F1 (mean ± sample SD) | Twitter2015 | Twitter2017 |
|---|---:|---:|
| Baseline | 68.3447 ± 0.8242 | 68.5923 ± 0.8768 |
| Target only | 68.2480 ± 0.9933 | 68.7411 ± 0.9752 |
| Fixed equal probability fusion | 69.0954 ± 0.8395 | 69.4443 ± 0.8459 |

Fusion minus baseline: +0.7506/+0.8519 pp; wins/ties/losses 12/1/2 and 13/1/1.
Fusion minus target: +0.8474/+0.7031 pp; wins/ties/losses 12/1/2 and 14/0/1.

| Test cohort | 2015 fusion | Δ baseline | Δ target | 2017 fusion | Δ baseline | Δ target |
|---|---:|---:|---:|---:|---:|---:|
| Initial3 | 68.8279 | +1.0165 | +0.6355 | 70.0527 | +2.1429 | +0.5386 |
| Historical12 | 69.1622 | +0.6842 | +0.9004 | 69.2921 | +0.5292 | +0.7443 |

Twitter2017 seed44 target-only 71.1075 becomes fusion 71.2681; this single seed
is supplementary, not the basis for the mean-gain conclusion. The results support
recommending fixed equal fusion as the next main candidate, retaining baseline
and target-only controls. They do not change the calibrated variant's failed
Dev gate or erase earlier Test exposure/historical seed selection. No Test
tuning, new training or additional experiments were performed in this step.

Artifacts in outputs/fixed_equal_fusion_test_20261018/: decision.json and SHA
sidecar, 30 paired result.json files and sidecars, GPU logs, comparison.json,
per_seed.csv, paired_results.csv, SUMMARY.txt, make_report.py, postrun-audit.json
and completed status.json. The audit confirms effective multimodal cache
fingerprints for all 30 pairs; the small dedicated fallback regression test
remains the documented minor coverage gap.

## Confirmed main candidate (2026-10-18)

After reviewing the full Dev/Test and per-seed comparisons, the user explicitly
confirmed: “将固定等权融合升级为主力候选，保留原 MACT 和分类器单独替换方案作对照。”

- Main candidate: fixed equal probability fusion of MACT and the target-context
  classifier, alpha=.5 and both temperatures=1, preserving MACT aspect boundaries
  and the existing verified fusion/tie-handling implementation.
- Control 1: original MACT, using its original CRF predictions.
- Control 2: target-context classifier alone replaces sentiment labels on the
  same MACT-predicted aspect boundaries.

This supersedes the 2026-10-17 target-only main-candidate designation. Subsequent
experiments should report all three schemes on paired seeds and retain Dev-based
model selection. configs/base.json and configs/target_context.json continue to
define their respective component models. The main candidate reuses both trained
models; its verified Test entry point is scripts/evaluate_fixed_fusion.py.

The evidence is the completed 15-seed Test comparison above (fusion means
69.0954/69.4443, gains +0.7506/+0.8519 pp over original MACT and
+0.8474/+0.7031 pp over target-only). This records the accepted research direction;
the failed calibrated Dev gate, historical selection limitations, locked
experiment decision, archived predictions and results remain as recorded.

## Follow-up Dev disagreement diagnostic (2026-10-18)

User authorized “开始下一步” after the proposal to diagnose expert disagreements
before adding a reliability module. Reused all 30 saved Dev score caches; no
training, fitted parameters or Test access. Fixed three untuned confidence
controls before reading scores: higher maximum probability, larger top-two
probability margin, and lower entropy (ties retain MACT).

| Mean Dev Joint F1 / diagnostic | Twitter2015 | Twitter2017 |
|---|---:|---:|
| Current fixed fusion | 68.0373 | 68.1259 |
| Higher maximum probability | 68.0609 | 68.0983 |
| Larger top-two margin | 68.0373 | 68.0871 |
| Lower entropy | 68.0609 | 68.1094 |
| Gold oracle choosing between experts | 73.3330 | 74.8826 |
| Ideal selectable gap above fixed fusion | 5.2957 | 6.7567 |

The oracle uses gold and is not a deployable result or a forecast. A separate
oracle retaining fixed fusion's possible third-class successes gives the same
values here (no such rescues observed); this equivalence is not assumed generally.

Among exact-boundary matched aspect-seed pairs, experts disagree on 14.00%/16.88%.
Of the fixed fusion's remaining sentiment errors, 898/3125 (28.74%) on 2015 and
1212/4373 (27.72%) on 2017 have at least one correct expert. The remaining
71.26%/72.28% are wrong in both experts. False-positive boundaries and missed
gold boundaries are counted separately; sentiment rerouting cannot repair them.
Aspect-seed totals repeat sentences across seeds and are not independent sample
counts. Dev-selected checkpoints remain an exploratory limitation.

All three original metric sets replayed exactly from cached predictions/scores;
count partitions and 30 paired oracle F1 calculations passed a separate stdlib
audit. Input/cache sidecars, current Dev gold and relevant numerical source
hashes were checked. This audit did not reload checkpoints or visual features.

Artifacts: outputs/disagreement_diagnostic_20261018/{protocol.json,diagnose.py,
comparison.json,per_seed.csv,matched_aspects.csv,SUMMARY.txt,audit.json,
independent_audit.py,independent-audit.json,*-unique-aspects.json}.

Interpretation: these three confidence-only rules do not improve both datasets.
This does not rule out a learned gate, but provides no basis to promote one.
Keep fixed fusion as main candidate and test additional evidence before choosing
a module; the majority shared errors also motivate better representations.

### Completed image-masking sensitivity probe

Ran all six Dev pairs for seeds42/43/44 with frozen MACT weights, original
predicted boundaries and original neighboring Viterbi tags. Compared real-image
emissions with `force_null=True`, which masks visual tokens but retains the
learned null token. Gold annotations were removed from model inputs. This is a
conditional polarity probe, not image-free end-to-end decoding or causal proof.

| Initial3 mean Dev / diagnostic | Twitter2015 | Twitter2017 |
|---|---:|---:|
| Fixed equal fusion | 68.0113 | 68.2173 |
| Masked-image conditional scores + target fusion | 68.0113 | 68.2452 |
| MACT with image, original boundaries | 67.5700 | 67.0481 |
| Masked-image conditional labels, original boundaries | 67.5115 | 66.4371 |
| Image changed conditional label (matched aspect-seed pairs) | 68/2940 | 125/3319 |
| Image helpful / harmful flips | 34 / 32 | 71 / 49 |
| Mean per-seed AUROC: raw expert margin difference | 0.5511 | 0.5631 |
| Mean per-seed AUROC: image-induced log-odds shift | 0.4704 | 0.4953 |

AUROC evaluates which expert is right only on matched, disagreeing aspects
where exactly one expert is correct; positive class is MACT correct. Scores
and directions were specified before summary, without fitting a gate or choosing
a threshold. These weak scalar results do not prove every image-based reliability
mechanism impossible, nor do they imply images are generally useless. They do
not support promoting the proposed simple image-sensitivity gate.

All six real-image predictions replayed exactly. Repeated FP32 scores differed
from the archive by at most 1.90735e-6; numerical checks used absolute tolerance
1e-5 while label checks remained exact. The initial attempt assumed a nonexistent
separate Dev visual cache; the corrected runner resolves and verifies the actual
train_dev cache from archived provenance. Failed attempts wrote no score results.
Independent scalar softmax/F1/flip-count recomputation passed for all six pairs.

Artifacts under outputs/disagreement_diagnostic_20261018/: null_probe.py,
summarize_null.py, audit_null.py and null_probe/{protocol.json,twitter*/seed*.json,
comparison.json,SUMMARY.txt,independent-audit.json,status.json}.

### Next design priority

Keep fixed fusion as the main candidate. Do not promote confidence-only or
image-log-odds routing based on these diagnostics. The next candidate design to
investigate is a small target-conditioned sentiment correction head, using the
target encoder representation and MACT aspect representation, anchored to fixed
fusion with zero initial correction. Unlike a pure expert selector, it must be
able to change an incorrect label shared by both experts. Track corrections and
regressions separately; unchanged aspect boundaries remain an invariant.

This is a proposed hypothesis, not an implemented or validated module and not
a novelty claim. Before training, compare nearest prior residual/correction
methods and specify the mechanism-specific contribution. Use Train-derived
held-out/out-of-fold examples where error-conditioned supervision is needed;
do not train a correction gate against the already-observed Test. Compare a plain
parameter-matched residual head against the proposed mechanism and the existing
three controls. Freeze the design before expanding seeds and final Test.
