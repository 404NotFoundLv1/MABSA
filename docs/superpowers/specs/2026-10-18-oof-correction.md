# Out-of-fold correction pilot

The user's “继续下一步” authorizes implementing and launching the previously
recommended out-of-fold (OOF) feature experiment. Work inline in the existing
branch, preserving current checkpoints, results and diagram artifacts. This is
a new experimental data pipeline, not a new architectural novelty claim.

## Fixed design

Seed42, Twitter2015 then Twitter2017, three outer folds. Group connected records
by identical case-folded word sequences OR identical image bytes; never split a
group. Deterministic greedy assignment balances sentence counts. Within each
outer training complement, reserve approximately 10% of groups as inner validation.
The outer held-out fold participates in neither gradient training nor checkpoint
selection. Preserve original sentence IDs in every derived JSON file. No Test
input or Test evaluation. Official Dev is used only after OOF feature generation
for a dataset, by the existing frozen full-Train experts and final head selection.

For each fold train MACT and target classifier from the original pretrained
backbone and seeded task heads, 35 epochs each with existing hyperparameters.
Choose earliest best inner-validation Joint F1, using MACT-predicted boundaries
for target selection. Keep the original online MACT selection (not its EMA).
Reload/check inner-validation predictions before extracting the held-out fold.
Both models must exclude every held-out group. No warm start from existing
full-Train task checkpoints. Train/inner/outer files, original Train, image/cache
identities, pretrained weights, configs, source and checkpoints are fingerprinted.

Concatenate held-out feature records, covering every original Train sentence
exactly once. Use the same 256+768 features, logits, zero-init residual heads,
plain versus 2*tanh(raw/2), CE on all exact-matched boundaries, 30 epochs and all
optimizer settings from the preceding pilot. Preserve all Dev false positives
and empty sentences. Final Dev features use unchanged full-Train main experts.
Re-extract the prior in-sample control under current source and reproduce its
two best-trained/selected results before interpreting OOF results.

The existing strict in-sample cache format remains strict. A separate verified
OOF format explicitly permits multiple checkpoint pairs for Train and the
original full-Train pair for Dev. Its validator checks fold data membership,
training/selection config paths, initial pretrained identity, actual checkpoint
metadata, hashes, exact coverage, tensor dimensions and gold-derived labels.
Do not relabel OOF caches as ordinary same-checkpoint caches.

## Reporting and screening

Record Train saturation before/after, controls, selected epoch including epoch0,
best trained epoch, wrong-to-correct, correct-to-wrong, shared-error rescues and
checkpoint reload verification. A paired candidate passes screening only if its
best selected Dev F1 exceeds fixed fusion by >=.20pp on BOTH datasets and is no
worse than its in-sample counterpart on either. Predefine this for both arms;
if both pass, retain both for later seed validation, without post-hoc retuning.
No automatic Test, new seeds or promotion of a model based on this pilot.

OOF fold experts have fewer training examples and their learned feature spaces
can drift relative to full-Train experts. Shared initialization reduces but does
not eliminate this mismatch; report it, not a pure causal saturation experiment.
All heads still select on previously-used Dev. This does not establish independent
generalization, significance or originality. Cross-split exact text/image overlap
between official Train and Dev is audited and disclosed, not silently removed.

## Execution

Train twelve fold models sequentially (two experts × three folds × two datasets).
Use a detached, logged local queue with PID/status, immutable launch protocol,
per-stage completion checks and fail-fast errors. Reuse only fully completed,
hash-verified stages; partial stage outputs require explicit diagnosis, never
automatic deletion. Do not label a launched multi-hour queue as completed.
The queue automatically extracts, trains both controls and OOF arms, audits saved
predictions and writes final comparisons after all required models finish.

Replay details: exact selected/best-trained epochs, metrics, error changes and
saved predictions are required. Train CE/residual RMS diagnostic differences are
reported separately rather than rejected bitwise; these diagnostics do not select
models. Prior result/prediction references are included in immutable launch hashes.
Completed head-stage reuse verifies Train/Dev cache hashes, format, source,
configuration, full epoch history, completion status and both saved checkpoints'
training provenance before accepting independently recomputed tuple metrics.
