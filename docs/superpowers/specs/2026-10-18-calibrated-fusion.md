# Calibrated MACT / target-context fusion

User approved implementing the proposed calibrated two-model fusion, inline in
the current source tree. Reuse all 15 completed paired seeds on both datasets;
no encoder training or new dependencies. Preserve original MACT and target-only
outputs. This stage uses Dev only and does not automatically evaluate Test.

For the five-tag MACT CRF, score each predicted aspect's three alternative B
labels with every other decoded tag fixed, including incoming/outgoing CRF
transitions and sequence endpoints. These are conditional polarity logits, not
unconditional token probabilities. Five-tag I labels do not encode polarity.
Verify original Viterbi predictions exactly against archives. Collect independent
target-model logits on the identical ordered predicted boundaries and verify its
argmax predictions exactly. Never supply labels to either model forward pass.

Fit one positive scalar temperature per expert by matched-boundary sentiment
NLL, bounded to [0.25,32], pooling the same seeds. False positive boundaries have
no polarity target and are excluded from temperature fitting, but remain in
Joint F1 evaluation. Calibrated probability mixture uses target weight alpha in
[0,.25,.5,.75,1]; select mean seed Joint F1, ties prefer smaller alpha. Endpoints
must reproduce the archived MACT / target predictions exactly, including ties.
Unchanged labels and boundaries remain unchanged; mixed-score ties prefer MACT.

Five deterministic SHA256 sentence-ID folds, shared across seeds, keep all
aspects of a sentence together. For each held-out fold, fit temperatures and
select alpha exclusively on the other four folds. Concatenate held-out
predictions before computing each seed's full Dev F1. Report per-seed values,
mean, sample SD and paired changes for baseline, target-only, raw .5 mixture,
and calibrated selected fusion. Raw .5 is an explanatory calibration ablation.
Also fit full-Dev deployment parameters, clearly separate from cross-fold
results. Dataset-specific parameters, never seed-specific parameter selection.
Require complete unique 15-seed cohorts before selection. Report initial3 and
historical12 separately as well as descriptive all15 aggregates.

The checkpoint selection already used this Dev set: cross-fitting reduces
fusion-parameter reuse but does not create an independent validation set.
Historical seeds were selected previously using known results. Do not claim
statistical independence or Test improvement. Advance to Test only if the
cross-fitted all15 mean beats both controls on each dataset; otherwise report
the failed gate without automatically searching more rules.

Cache provenance includes checkpoints, dataset, predictions, tokenizer/config,
visual cache manifest, source hashes, full replay checks and unchanged inputs.
Refuse stale or nonfresh outputs. Data access guard disallows Test data and
evaluation artifacts in this Dev-only workflow. Verify cache identity before
aggregation. Tests cover exhaustive CRF scores, calibration, endpoint behavior,
grouped folds, held-out-label exclusion and malformed/incomplete records.
