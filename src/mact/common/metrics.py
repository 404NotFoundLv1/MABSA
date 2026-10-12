"""Exact set-based MABSA evaluation metrics."""

from __future__ import annotations

from collections import defaultdict


def _rates(tp: int, predicted: int, gold: int) -> dict:
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "predicted": predicted, "gold": gold}


def evaluate_predictions(
    gold: dict[str, list[tuple[int, int, int]]],
    predicted: dict[str, list[tuple[int, int, int]]],
) -> dict:
    """Evaluate exact joint triples and sentiment-free aspect boundaries."""
    if set(gold) != set(predicted):
        raise ValueError("gold and prediction example ID sets must match")
    gold_triples = {(example_id, *triple) for example_id, triples in gold.items() for triple in triples}
    predicted_triples = {(example_id, *triple) for example_id, triples in predicted.items() for triple in triples}
    gold_boundaries = {(example_id, start, end) for example_id, start, end, _ in gold_triples}
    predicted_boundaries = {(example_id, start, end) for example_id, start, end, _ in predicted_triples}
    joint = _rates(len(gold_triples & predicted_triples), len(predicted_triples), len(gold_triples))
    mate = _rates(len(gold_boundaries & predicted_boundaries), len(predicted_boundaries), len(gold_boundaries))

    gold_sentiments: dict[tuple[str, int, int], set[int]] = defaultdict(set)
    predicted_sentiments: dict[tuple[str, int, int], set[int]] = defaultdict(set)
    for example_id, start, end, sentiment in gold_triples:
        gold_sentiments[(example_id, start, end)].add(sentiment)
    for example_id, start, end, sentiment in predicted_triples:
        predicted_sentiments[(example_id, start, end)].add(sentiment)
    matched = gold_boundaries & predicted_boundaries
    correct = sum(bool(gold_sentiments[boundary] & predicted_sentiments[boundary]) for boundary in matched)
    count = len(matched)
    return {
        "joint": joint,
        "mate": mate,
        "sentiment_on_matched": {"accuracy": correct / count if count else 0.0, "count": count, "correct": correct},
    }
