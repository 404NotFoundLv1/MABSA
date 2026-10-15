# Frozen baseline with independent OCR residual

Approved task: preserve MACT five-tag baseline and train only an independent OCR correction. No Test access. Existing source, results and official splits remain intact.

Design: frozen baseline in eval mode produces word queries and emissions. Its frozen DeBERTa independently encodes OCR (64 content subwords plus special tokens). Trainable projection and four-head attention read OCR; a zero-initialized, bias-free five-score output adds a residual to baseline emissions. Original frozen CRF computes NLL and decodes. Rows without OCR bypass the branch exactly. No label or gold aspect is an input.

Protocol: seed 42, Twitter2015/2017, 35 epochs, batch 1, accumulation 8, AdamW branch LR 1e-4, decay .01 on matrices, warmup .1, clip 1. Sentence-mean CRF loss includes unchanged no-OCR sentences as constants. All-empty windows skip backward and optimizer updates; scheduler advances on training windows. Epoch-only VLP-style logging. Epoch 0 is a measured control, excluded from best-trained-epoch selection, so negative outcomes remain visible. Report both trained-best and control. No hyperparameter sweep.

Baseline: original 35-epoch seed42 checkpoints were found in mabsa_evidence/experiments/20261011_sequence_crf_35ep/runs; their hashes exactly equal archived MACT summaries. Copy weights into MACT assets, never modify originals. Require full Dev predictions and metrics to reproduce the archived runs before training. Persist baseline hash and all input/source hashes. Branch checkpoints store branch weights and a pinned baseline dependency; reload fails if it changes.

Tasks:
1. Tests first: independent input alignment, zero-initialized exact equality, no-OCR equality after training, baseline frozen/eval, gradient flow, all-empty windows, roundtrip and hash rejection.
2. Implement separate residual module/trainer/config and command; default baseline behavior unchanged except optional word-feature return. Do not import the prior concatenated OCR pipeline.
3. Add fixed-checkpoint Dev diagnostics: all/has-OCR/no-OCR metrics, removed OCR, three fixed length-matched shuffled OCR permutations (17/43/97), original no-OCR identity. Validate reload and baseline unchanged.
4. Full suite, real-GPU smoke for both datasets, fresh independent review, commit source, launch two detached 35-epoch jobs (one per GPU) with automatic diagnostics and aggregate results.

Review focus: baseline eval survives wrapper.train; CRF frozen but residual gradients flow; all-masked OCR rows never enter attention; optimizer cannot mutate baseline; original tokens and labels unchanged; no Test reads; baseline checkpoint/asset hashes enforced; selected checkpoint reload exact; negative results not hidden by epoch-0 fallback.

Ledger: initial source 782211f on existing ocr-context feature branch; user prefers a shared source directory and config-based experiments. No extra worktree or source copy.

Implementation ledger:
- Tasks 1–3 complete: separate OCR inputs/attention, 461,824 trainable parameters, frozen eval baseline/CRF, trained-only checkpoint selection, automatic Dev subset/removal/shuffle diagnosis. Default baseline forward changes only by optional word-feature return.
- Unit verification: zero-init identity, unchanged original alignment, mixed/all-empty batches, frozen state and gradients, accumulation denominator, branch checkpoint roundtrip/hash guard, and length-matched derangements. Initial implementation tests exposed unsupported fast-tokenizer special-mask API; content positions now use tokenizer special-token layout without treating unknown content as padding.
- Ruling: exact emission identity is measured against the frozen eval baseline. Toggling requires_grad can change attention floating-point rounding on tiny Bert (~3e-8), so full archived Dev prediction equality is independently required before real training. This avoids mistaking numerical rounding for a learned correction; no tolerance is allowed for prediction differences.
- Both preliminary GPU smokes passed (24 Train/12 Dev, three updates): frozen-state SHA exact, archived subset prediction equality, initial emission equality, nonzero learned output, exact reload and removed/no-OCR equality. Peak allocated memory ~778 MB each.
- Fresh independent review: no blocking functional findings; verified all six full-Dev shuffle assignments (370/346 nonempty rows, max token-length mismatch 2).
- Review correction: tokenizer files were missing from integrity protection. Regression test failed on missing tokenizer.json, then tokenizer files and visual feature payload were added to protected hashes. Full final suite and final-source GPU smoke follow before launch.
- Review runtime checks (full archived Dev identity, all 35 epochs, detached jobs) remain required; no assumption that smoke proves full-run performance. No deferred minor issues.
