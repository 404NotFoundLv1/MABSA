# Target representation A/B/C pilot

Authorization: user accepted the proposed A/B/C experiment and requested execution
in the current session. Work in the existing ocr-context checkout; retain prior
figures and README changes. No additional approval gate or worktree is needed.

Purpose: test whether explicit aspect pooling and aspect-conditioned context
reduce sentiment errors in the current fixed-equal-fusion candidate. This is a
performance hypothesis, not an originality or guaranteed improvement claim.

## Locked comparison

- A = opening-marker hidden state, identical to existing TargetClassifier.
- B = concatenate opening-marker state and mean aspect WORD representation.
  Each aspect word is the mean of its subwords; words receive equal weight.
- C = B plus a context vector queried by the aspect mean: learned linear query
  and key projections to 128 dimensions, scaled dot-product softmax, weighted sum
  of encoder hidden states. Context includes original text outside the marked
  aspect; excludes both markers, special tokens and padding. Empty context gives
  exactly zero. Unknown original words remain context candidates.
- A/B/C share CE loss, dropout 0.1, pretrained DeBERTa-v3-base initialization,
  35 epochs, batch/eval batch 8, encoder LR 2e-5, non-encoder LR 1e-4, AdamW
  decay 0.01 (exclude biases/LayerNorm), warmup 0.1, clipping 1, FP32, seed42.
- Train all target encoder weights. MACT is frozen; boundaries never change.
  Both temperatures remain 1, target probability weight remains 0.5, retain
  existing agreement/tie rules. No auxiliary loss, confidence gate or tuning.
- Retrain A as well: all three select the earliest highest FULL predicted-boundary
  Dev fixed-fusion Joint F1. Also report classifier-alone metrics at that epoch.
  The historical A selected on classifier-only Dev remains a separate reference.
- Six sequential jobs: twitter2015 A/B/C, then twitter2017 A/B/C. No Test reads,
  no automatic additional seeds. Resume only if source/config/input/output hashes
  agree and prior job has completed and passed reload checks; otherwise fail.

## Evidence and outputs

Before training inspect official Train statistics and Dev error slices and a
deterministic sample of shared errors. Multi-aspect, mixed-polarity, multiword,
negation-cue and contrast-cue slices are descriptive, overlapping; lexical cues
are not annotations of error causes. Do not use held-out labels as model inputs.

Replay existing MACT and historical target on full Dev into fresh, verified score
caches using existing extract_run. Lock data/model/cache/source hashes. Preserve
empty sentences, false-positive boundaries and exact aspect ordering. Strip gold
from all prediction inputs. Checkpoints have a new versioned format containing
variant and attention dimension. Never overwrite/load them as legacy checkpoints.

Each job saves config, provenance, history, best target/fused predictions, scores,
checkpoint, full-Dev reload verification and completion/failure status. Report
Joint P/R/F1, boundary metrics, matched sentiment accuracy, matched class confusion,
and corrected/regressed/shared-error rescues vs historical fusion and paired A.
Selection is based on fusion F1 alone, earliest tie. A change in random seed best
score is not evidence of repeatability. Small-real-backbone smoke precedes runs.

## Completion

Model and runner behavioral tests pass, full suite passes, fresh independent code
review is addressed, all three variants pass real-backbone smoke/reload, diagnostic
is reviewed, and the sequential batch is demonstrably running (or completed).
Training may outlive the turn; report observed status and measured ETA honestly.
