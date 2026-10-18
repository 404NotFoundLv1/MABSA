# Target-conditioned sentiment correction pilot

The user approved starting the preceding correction-head proposal, inline in
the existing branch. Keep fixed equal fusion as main candidate and retain both
original controls. This pilot is seed42 on twitter2015 and twitter2017, Train/Dev
only. No automatic Test, seed expansion, backbone finetuning or parameter sweep.

## Scientific question and prior work

Can frozen target-context and MACT aspect representations correct shared
sentiment errors without losing more correct predictions? Compare two heads
with identical parameters, initialization, batches and optimization: a plain
log-probability residual, and the same residual limited to [-2,2] per class by
2*tanh(raw/2). Both start at exactly zero residual. The plain head isolates
the effect of limiting correction. This is a feasibility/mechanism experiment,
not a claim that residual correction or clipping is novel.

Relevant prior examples include residual feature refinement in aspect sentiment
analysis (https://www.mdpi.com/2227-7390/13/15/2437), gated residual proxy correction
for incomplete multimodal sentiment (https://arxiv.org/abs/2608.19971), and
visual-acoustic residual compensation with correction limits
(https://www.ecice06.com/EN/10.19678/j.issn.1000-3428.0260838). Tasks and inputs
differ; a future novelty claim requires mechanism-specific comparisons.

## Features and data isolation

Freeze both existing seed42 models in eval mode. Decode MACT boundaries on
Train and Dev with all gold annotations removed from model inputs. For each
predicted aspect cache (a) mean of MACT visual-head hidden states over that span,
256 dimensions; (b) target encoder's opening-marker state, 768 dimensions;
(c) both polarity score triples; (d) the existing fixed-fusion decision.
Use a temporary forward hook on the existing MACT visual_head GELU; remove it
even on failure. No baseline source or checkpoint changes.

Supervise with ordinary 3-class CE on every exact-boundary matched Train
prediction. Gold affects targets and the training loss mask only. Do not sample
by correctness, use an oracle choice label, fit on Dev, or use Dev labels in
features. False-positive Train boundaries have no sentiment target and are
excluded from CE, but all predicted Dev boundaries remain in evaluation.
Train representations are in-sample to the frozen encoders: disclose this
limitation and measure Train saturation. This is supervised feature-head fitting,
not an out-of-fold error-routing experiment; OOF is required before adding
error-conditioned routing targets or claiming unbiased error supervision.

## Model and fixed optimization

Separate LayerNorm on the 256/768-dimensional vectors, concatenate, Linear to
128, GELU, Dropout .1, Linear to 3. Last linear weights/bias initialize to zero.
Corrected logits = log(.5*softmax(base)+.5*softmax(target)) + residual, using
float64 probability arithmetic and a 1e-300 lower clamp. A zero residual returns
the exact historical fixed decision, including its tie/consensus rules; learned
nonzero corrections may override unanimous wrong predictions. Exact ties in
corrected logits prefer the fixed decision if it is among the maxima.

30 epochs, batch64, AdamW lr .001, weight_decay .01, gradient clip1, seed42,
CE plus .01*mean(residual^2). Head training on CPU over cached features.
Identical trainable parameter counts for both variants. No other losses or
hyperparameter trials in this pilot. Record epoch0 and all trained epochs.
Select earliest best Dev Joint F1 including epoch0 as a fallback, and separately
report best trained epoch so selection fallback cannot masquerade as improvement.

## Verification and advancement

Dev feature extraction must reproduce both archived predictions/metrics and
fixed-fusion metrics; labels/boundaries exact, score tolerance at most 1e-5.
Record checkpoint, tokenizer, data, effective visual cache and relevant source
hashes; refuse stale, mixed-seed, incomplete or nonfresh caches. Require Train
and Dev identities, disjoint sentence IDs, matching feature dimensions and frozen
checkpoint provenance before training. Reload head checkpoint and reproduce all
Dev predictions/metrics. Report correct-to-wrong, wrong-to-correct and rescued
shared errors. Initial3 diagnostics are motivation, not validation data for fitting.

Advance the bounded mechanism only if seed42 improves fixed fusion by at least
.20 F1 points on BOTH datasets and is no worse than the plain residual on either.
This is a prespecified pilot screening threshold, not significance. A passing
pilot justifies multi-seed validation; it does not authorize automatic Test or
establish originality. On failure, report it without tuning another setting.
