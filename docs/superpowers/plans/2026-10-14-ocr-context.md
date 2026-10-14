# OCR context implementation plan

> Execute inline with superpowers:executing-plans; use tests before implementation.

Goal: test whether automatically recognized image text improves the existing five-tag MACT baseline on Dev.

Design: append OCR token IDs and a separator after the unchanged original text encoding. OCR word IDs are -1, so only original words enter pooling, the five-tag heads, loss and CRF decoding. No model parameters are added. Empty OCR is exactly the baseline input. Disabled OCR performs no OCR I/O. Keep one source tree, using the ocr-context branch in the existing directory per user preference.

Fixed treatment: confidence >=0.8 and at least two adjacent ASCII letters per OCR line; collapse whitespace, preserve engine line order, no label/target filtering or manual correction. Append at most 64 subwords plus a final SEP. Original max_length stays 256, total <=321. R-Drop, auxiliary CE, visual gate and word context are off. Use only the completed frozen Train/Dev OCR probe to build versioned assets with image and content hashes; missing or mismatched OCR is an error, not silent empty text.

Evaluation: seeds 42/43/44, both datasets, 35 epochs, batch 1, accumulation 8, original learning rates, epoch-only logging. Best checkpoint selected exclusively by Dev JMASA F1. Pair with archived baseline runs after source-off parity and data/config checks. No Test access, including no Test hashing during train-only runs. Future Test requires an independently generated, verified cache using the same OCR pipeline; today's Train/Dev cache must fail on missing images.

Tasks:
- [x] Add OCR asset loader/filter and optional data suffix. Tests cover corrupted assets/images, missing images, disabled/empty parity, budget preservation, word alignment, no gold dependence, and gradients through context.
- [x] Wire optional run config into training/evaluation/checkpoint provenance; retain legacy checkpoints. Tests cover mutually exclusive treatment, reload parity, and train-only never opening Test.
- [x] Export immutable derived OCR assets; audit all 5151 original examples and token budgets. Run full suite, exact source-off parity, real GPU smoke for both datasets, and worst-length memory check.
- [ ] Review changes, fix any correctness findings, commit shared source. Start detached Dev-only scheduler and confirm actual training progress. Save launch configuration, hashes, paired Dev collector and logs.

Review focus: missing cache must not be treated as no text; original tokens/labels must not be truncated; OCR must never create decode positions; evaluation must use checkpoint OCR settings and hashes; no Test read in train-only mode.

Authorization: user approved starting this follow-up after the OCR feasibility recommendation. Routine implementation choices above are recorded here without adding another permission round.

Validation: 91 unit/integration tests pass. Exact baseline-v1 initialization, RNG, emissions, loss, gradients and optimizer updates match with OCR off. All 5,151 Train/Dev examples preserve token prefixes and tags; actual maximum combined length 110. Both datasets pass real CUDA 24-train/8-dev three-update smoke and exact checkpoint reload. A 321-token Adam stress test reserves 4,049,600,512 bytes. Independent review found no blocking issues in core code or launcher. Test cache extension and force_null semantics are documented limitations.
