"""Conditional five-tag CRF polarity scores and calibrated expert mixtures."""
import math

import torch

from .labels import aspects_to_tags


def _scores(logits):
    logits = torch.as_tensor(logits).detach().to(dtype=torch.float64, device='cpu')
    if logits.ndim != 2 or logits.shape[1] != 3 or not torch.isfinite(logits).all():
        raise ValueError('expected finite [N,3] polarity scores')
    return logits


def _labels(labels, n):
    labels = torch.as_tensor(labels, device='cpu')
    if labels.shape != (n,) or labels.dtype not in (torch.int32, torch.int64):
        raise ValueError('expected integer [N] labels')
    if ((labels < 0) | (labels > 2)).any():
        raise ValueError('polarity outside [0,2]')
    return labels.long()


def aspect_polarity_logits(crf, emissions, aspects):
    """log p(polarity | every other Viterbi tag), up to additive constants.

    Only the B tag varies in the five-tag scheme. Scores include both incident
    transitions or the corresponding sequence endpoint. Output order NEG/NEU/POS.
    """
    if crf.num_tags != 5 or emissions.ndim != 2 or emissions.shape[1] != 5:
        raise ValueError('conditional polarity requires five-tag CRF emissions')
    if not torch.isfinite(emissions).all():
        raise ValueError('nonfinite emissions')
    tags = aspects_to_tags(len(emissions), aspects, 'five')
    alternatives = torch.tensor([3,2,1], device=emissions.device)
    start_scores, transitions = crf._scores()
    rows = []
    for start, _, _ in aspects:
        left = start_scores[alternatives] if start == 0 else transitions[tags[start-1], alternatives]
        right = (crf.end_transitions[alternatives] if start == len(tags)-1
                 else transitions[alternatives, tags[start+1]])
        rows.append(emissions[start, alternatives] + left + right)
    return torch.stack(rows) if rows else emissions.new_empty((0,3))


def fit_temperature(logits, labels):
    """Bounded scalar NLL minimization; deterministic golden-section search."""
    z = _scores(logits)
    y = _labels(labels, len(z))
    if not len(z):
        raise ValueError('temperature calibration requires matched aspects')
    def nll(log_t):
        scores = z / math.exp(log_t)
        return float((scores.logsumexp(-1)-scores[torch.arange(len(y)),y]).mean())
    lo, hi = math.log(.25), math.log(32.)
    ratio = (math.sqrt(5)-1)/2
    c, d = hi-ratio*(hi-lo), lo+ratio*(hi-lo)
    fc, fd = nll(c), nll(d)
    for _ in range(64):
        if fc <= fd:
            hi, d, fd = d, c, fc
            c = hi-ratio*(hi-lo); fc = nll(c)
        else:
            lo, c, fc = c, d, fd
            d = lo+ratio*(hi-lo); fd = nll(d)
    candidates = [0., math.log(.25), math.log(32.), (lo+hi)/2]
    best = min(candidates, key=nll)
    return math.exp(best)


def fuse_labels(base_logits, target_logits, baseline_labels, target_labels,
                temperature_base, temperature_target, alpha):
    b, t = _scores(base_logits), _scores(target_logits)
    if b.shape != t.shape:
        raise ValueError('expert score shapes differ')
    old, new = _labels(baseline_labels,len(b)), _labels(target_labels,len(b))
    if (not all(math.isfinite(x) and x > 0 for x in (temperature_base,temperature_target))
            or not math.isfinite(alpha) or not 0 <= alpha <= 1):
        raise ValueError('invalid mixture temperature or weight')
    if alpha == 0:
        return old.clone()
    if alpha == 1:
        return new.clone()
    if not len(b):
        return old.clone()
    p = (1-alpha)*(b/temperature_base).softmax(-1) + alpha*(t/temperature_target).softmax(-1)
    result = p.argmax(-1)
    ties = p[torch.arange(len(old)),old] == p.max(-1).values
    result[(old == new) | ties] = old[(old == new) | ties]
    return result
