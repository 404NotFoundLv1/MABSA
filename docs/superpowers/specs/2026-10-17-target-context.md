# Target context sentiment pilot

Approved scope: user's “开始做” implements the proposed target-marked context
reencoding pilot, sequentially in the existing source tree. Original MACT remains
the baseline. This is exploratory Dev evidence, not a replacement-base claim.

Train an independent local DeBERTa-v3-base encoder with two added target markers,
dropout 0.1 and a three-class linear classifier on the opening marker's contextual
hidden state. All encoder weights train. Each gold Train aspect is one example;
no-aspect sentences produce no classification examples. Mark by word offsets,
including repeated names; reject truncation or marker collisions. Maximum length
258 preserves the baseline's 256-token context plus two markers.

Use only official Train and Dev, seed 42 for twitter2015 and twitter2017. Freeze
the existing uniform online seed42 MACT best-Dev predictions from
outputs/llrd_ema_s42_43_44_20261016. Relabel every predicted span, including false
positives. Never add, remove, merge or adjust boundaries. Gold Dev spans are
only a secondary diagnostic and never classifier inputs for the primary score.

Budget: 35 epochs, batch 8, FP32, encoder LR 2e-5, head LR 1e-4, AdamW decay .01
(bias/LayerNorm excluded), linear warmup .1, clip 1. Earliest maximum full
predicted-boundary Dev Joint F1 selects checkpoint; no confidence/blending sweep.
Report gold-boundary accuracy and macro F1 separately. Exact-boundary metrics
must equal baseline at every epoch. Verify saved checkpoint predictions on full
Dev after reload, source/input hashes unchanged, and archive status/history.

Run both dataset jobs on separate available GPUs after a short real-backbone
smoke and tests/review. Do not evaluate Test or add seeds automatically. If either
dataset fails to improve primary Dev F1, stop expansion. Positive results on both
only justify considering replication; one-seed Dev selection cannot prove stability.

Use a separate checkpoint format and standalone script; no changes to baseline
training or inference. Preserve all existing data and outputs.
