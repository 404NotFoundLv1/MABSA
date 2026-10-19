"""Descriptive Train/Dev slices; lexical cues are NOT causal error annotations."""
from collections import Counter
import re

from .common.metrics import evaluate_predictions


def matched_diagnostic(gold, predictions):
    confusion = [[0] * 3 for _ in range(3)]
    for key, values in predictions.items():
        truth = {(s, e): p for s, e, p in gold[key]}
        for s, e, p in values:
            if (s, e) in truth:
                confusion[truth[s, e]][p] += 1
    scores = []
    for p in range(3):
        denominator = sum(confusion[p]) + sum(row[p] for row in confusion)
        scores.append(2 * confusion[p][p] / denominator if denominator else 0.)
    return dict(confusion=confusion, class_order=['NEG', 'NEU', 'POS'], per_class_f1=scores,
                macro_f1=sum(scores)/3, count=sum(map(sum, confusion)))


def error_transitions(gold, before, after):
    result = dict(corrected=0, regressed=0, changed=0, still_wrong=0)
    if set(gold) != set(before) or set(before) != set(after):
        raise ValueError('sentence identity mismatch in error comparison')
    for key, old in before.items():
        new = after[key]
        if [(s, e) for s, e, _ in old] != [(s, e) for s, e, _ in new]:
            raise ValueError('cannot compare changed boundaries')
        truth = {(s, e): p for s, e, p in gold[key]}
        for (s, e, p), (_, _, q) in zip(old, new):
            if (s, e) not in truth: continue
            y = truth[s, e]
            result['corrected'] += int(p != y and q == y)
            result['regressed'] += int(p == y and q != y)
            result['changed'] += int(p != q)
            result['still_wrong'] += int(p != y and q != y)
    return result


def dataset_statistics(examples):
    labels = Counter(p for e in examples for _, _, p in e.aspects)
    return dict(sentences=len(examples), aspects=sum(labels.values()),
        negative=labels[0], neutral=labels[1], positive=labels[2],
        multi_aspect_sentences=sum(len(e.aspects) > 1 for e in examples),
        mixed_polarity_sentences=sum(len({p for _, _, p in e.aspects}) > 1 for e in examples),
        multiword_aspects=sum(end-start > 1 for e in examples for start, end, _ in e.aspects))


def diagnose(examples, cache):
    from .representation_experiment import evaluate_scores
    gold = {e.id: e.aspects for e in examples}
    metrics, predictions = evaluate_scores(cache, cache['target_logits'], gold)
    slices = {name: dict(matched=0, fusion_wrong=0, shared_wrong=0) for name in
        ('all', 'single_aspect', 'multi_aspect', 'mixed_polarity', 'multiword', 'negation_cue', 'contrast_cue')}
    sample = []
    shared = 0
    for example in sorted(examples, key=lambda e: e.id):
        truth = {(s, e): p for s, e, p in example.aspects}
        base = {(s, e): p for s, e, p in cache['baseline'][example.id]}
        target = {(s, e): p for s, e, p in cache['target'][example.id]}
        fusion = {(s, e): p for s, e, p in predictions['fusion'][example.id]}
        sentence = ' '.join(example.words).lower()
        neg = bool(re.search(r"\b(?:not|no|never|neither|hardly|without)\b|n't", sentence))
        contrast = bool(re.search(r'\b(?:but|however|although|though|yet|whereas)\b', sentence))
        for (s, e), y in truth.items():
            if (s, e) not in base: continue
            both_wrong = base[s, e] != y and target[s, e] != y
            wrong = fusion[s, e] != y
            shared += int(both_wrong and wrong)
            flags = ['all', 'multi_aspect' if len(truth) > 1 else 'single_aspect']
            if len(set(truth.values())) > 1: flags.append('mixed_polarity')
            if e-s > 1: flags.append('multiword')
            if neg: flags.append('negation_cue')
            if contrast: flags.append('contrast_cue')
            for name in flags:
                slices[name]['matched'] += 1
                slices[name]['fusion_wrong'] += int(wrong)
                slices[name]['shared_wrong'] += int(both_wrong and wrong)
            if both_wrong and wrong and len(sample) < 30:
                sample.append(dict(id=example.id, text=' '.join(example.words), aspect=' '.join(example.words[s:e]),
                    start=s, end=e, gold=y, base=base[s,e], target=target[s,e], fusion=fusion[s,e], flags=flags))
    return dict(statistics=dataset_statistics(examples), metrics=metrics, slices=slices,
        shared_errors=shared, shared_error_sample=sample,
        limitation='Descriptive overlapping slices; cue presence does not identify an error cause. Dev has been reused.')
