"""Dataset loading, word alignment, and batching for span MABSA."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


NEG, NEU, POS = 0, 1, 2
_POLARITIES = {
    "negative": NEG,
    "neg": NEG,
    "neutral": NEU,
    "neu": NEU,
    "positive": POS,
    "pos": POS,
}


@dataclass
class Example:
    id: str
    words: list[str]
    image_id: str
    aspects: list[tuple[int, int, int]]
    ignored_spans: list[tuple[int, int]] = field(default_factory=list)


def _valid_span(start: Any, end: Any, n_words: int) -> bool:
    return (
        isinstance(start, int) and not isinstance(start, bool)
        and isinstance(end, int) and not isinstance(end, bool)
        and 0 <= start < end <= n_words
    )


def _polarity(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value in (NEG, NEU, POS):
        return value
    if isinstance(value, str) and value.lower() in _POLARITIES:
        return _POLARITIES[value.lower()]
    raise ValueError(f"invalid aspect polarity: {value!r}")


def _overlap(left: tuple[int, int, int], right: tuple[int, int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _term_matches(term: Any, span_words: list[str]) -> bool:
    if isinstance(term, str):
        return not term or term == " ".join(span_words)
    if isinstance(term, list):
        return not term or (all(isinstance(word, str) for word in term) and term == span_words)
    return False


def load_examples(path: str | Path, conflict_policy: str = "error") -> tuple[list[Example], dict]:
    """Load and validate the original JSON-list annotation format."""
    if conflict_policy not in {"error", "drop"}:
        raise ValueError("conflict_policy must be 'error' or 'drop'")
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("dataset JSON must be a list")

    examples: list[Example] = []
    seen_ids: set[str] = set()
    audit = {
        "examples": 0,
        "aspects": 0,
        "conflicting_spans": 0,
        "deduplicated_aspects": 0,
    }
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"record {index} must be an object")
        words = record.get("words")
        image_id = record.get("image_id")
        if not isinstance(words, list) or not words or any(not isinstance(word, str) or not word for word in words):
            raise ValueError(f"record {index} has invalid words")
        if not isinstance(image_id, str) or not image_id:
            raise ValueError(f"record {index} has invalid image_id")
        raw_id = record.get("id", f"{source.stem}:{index}:{image_id}")
        if not isinstance(raw_id, str):
            raise ValueError(f"record {index} has non-string id")
        if raw_id in seen_ids:
            raise ValueError(f"duplicate example id: {raw_id}")
        seen_ids.add(raw_id)

        raw_aspects = record.get("aspects", [])
        if raw_aspects is None:
            raw_aspects = []
        if not isinstance(raw_aspects, list):
            raise ValueError(f"record {index} has invalid aspects")
        triples: list[tuple[int, int, int]] = []
        for annotation in raw_aspects:
            if not isinstance(annotation, dict):
                raise ValueError(f"record {index} has invalid aspect annotation")
            start, end = annotation.get("from"), annotation.get("to")
            if not _valid_span(start, end, len(words)):
                raise ValueError(f"record {index} has invalid aspect span")
            term = annotation.get("term")
            if term is not None and not _term_matches(term, words[start:end]):
                raise ValueError(f"record {index} has mismatched aspect term")
            triples.append((start, end, _polarity(annotation.get("polarity"))))

        unique = sorted(set(triples))
        audit["deduplicated_aspects"] += len(triples) - len(unique)
        labels_by_span: dict[tuple[int, int], set[int]] = {}
        for start, end, label in unique:
            labels_by_span.setdefault((start, end), set()).add(label)
        conflicts = sorted(span for span, labels in labels_by_span.items() if len(labels) > 1)
        if conflicts and conflict_policy == "error":
            raise ValueError(f"record {index} has contradictory labels at {conflicts}")
        if conflicts:
            audit["conflicting_spans"] += len(conflicts)
            conflict_set = set(conflicts)
            unique = [triple for triple in unique if triple[:2] not in conflict_set]

        for position, triple in enumerate(unique):
            if any(_overlap(triple, other) for other in unique[position + 1 :]):
                raise ValueError(f"record {index} has overlapping gold aspect spans")
        examples.append(Example(raw_id, words, image_id, unique, conflicts))
        audit["aspects"] += len(unique)
        audit["examples"] += 1
    return examples, audit


def enumerate_spans(n_words: int) -> list[tuple[int, int]]:
    if not isinstance(n_words, int) or isinstance(n_words, bool) or n_words < 0:
        raise ValueError("n_words must be a non-negative integer")
    return [(start, end) for start in range(n_words) for end in range(start + 1, n_words + 1)]


class SpanDataset(Dataset):
    """Pre-tokenized word-aligned examples; image loading remains lazy."""

    def __init__(
        self,
        examples: list[Example],
        tokenizer: Any,
        image_root: str | Path | None = None,
        image_processor: Any = None,
        max_length: int = 256,
    ) -> None:
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("SpanDataset requires a fast tokenizer")
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("max_length must be positive")
        self.examples = examples
        self.tokenizer = tokenizer
        self.image_root = Path(image_root) if image_root is not None else None
        self.image_processor = image_processor
        self.max_length = max_length
        self._items = [self._tokenize(example) for example in examples]

    def _tokenize(self, example: Example) -> dict[str, Any]:
        encoded = self.tokenizer(
            example.words,
            is_split_into_words=True,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=True,
        )
        input_ids = list(encoded["input_ids"])
        if len(input_ids) > self.max_length:
            raise ValueError(f"example {example.id} tokenized length exceeds max_length")
        raw_word_ids = encoded.word_ids(batch_index=0)
        present_words = {word_id for word_id in raw_word_ids if word_id is not None}
        expected_words = set(range(len(example.words)))
        if present_words != expected_words:
            raise ValueError(f"example {example.id} lost words during tokenization")
        spans = enumerate_spans(len(example.words))
        aspect_lookup = {(start, end): label for start, end, label in example.aspects}
        ignored = set(example.ignored_spans)
        aspect_labels = [
            -100 if span in ignored else int(span in aspect_lookup)
            for span in spans
        ]
        sentiment_labels = [
            aspect_lookup.get(span, -100) if span not in ignored else -100
            for span in spans
        ]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "length": len(input_ids),
            "word_ids": torch.tensor(
                [-1 if word_id is None else word_id for word_id in raw_word_ids], dtype=torch.long
            ),
            "spans": torch.tensor(spans, dtype=torch.long),
            "aspect_labels": torch.tensor(aspect_labels, dtype=torch.long),
            "sentiment_labels": torch.tensor(sentiment_labels, dtype=torch.long),
            "example": example,
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self._items[index])
        if self.image_processor is not None:
            from PIL import Image

            image_path = Path(item["example"].image_id)
            if self.image_root is not None:
                image_path = self.image_root / image_path
            if not image_path.is_file():
                raise FileNotFoundError(f"required image is missing: {image_path}")
            try:
                with Image.open(image_path) as image:
                    rgb_image = image.convert("RGB")
                    processed = self.image_processor(images=rgb_image, return_tensors="pt")
            except OSError as error:
                raise ValueError(f"required image is corrupt: {image_path}") from error
            pixels = processed["pixel_values"]
            if not isinstance(pixels, torch.Tensor):
                raise ValueError("image_processor did not return tensor pixel_values")
            item["pixel_values"] = pixels[0] if pixels.ndim == 4 and pixels.shape[0] == 1 else pixels
        return item


class PairedImageDataset(Dataset):
    """Attach a fixed Train-image derangement without changing any labels."""

    def __init__(self, dataset: SpanDataset, mismatched_examples: list[Example]):
        originals = dataset.examples
        if dataset.image_processor is None:
            raise ValueError("paired images require an image processor")
        if len(originals) != len(mismatched_examples) or not originals:
            raise ValueError("paired examples must match the original dataset")
        if any(replace(other, image_id=original.image_id) != original
               for original, other in zip(originals, mismatched_examples)):
            raise ValueError("paired examples may change only image IDs")
        if (Counter(x.image_id for x in originals) != Counter(x.image_id for x in mismatched_examples)
                or any(x.image_id == y.image_id for x, y in zip(originals, mismatched_examples))):
            raise ValueError("paired images must derange the original Train image multiset")
        self.dataset = dataset
        image_indices = {example.image_id: index for index, example in enumerate(originals)}
        self.mismatch_indices = [image_indices[example.image_id] for example in mismatched_examples]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        item["mismatch_pixel_values"] = self.dataset[self.mismatch_indices[index]]["pixel_values"]
        return item


class SpanCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            raise ValueError("cannot collate an empty batch")
        batch_size = len(items)
        max_tokens = max(item["input_ids"].numel() for item in items)
        max_words = max(len(item["example"].words) for item in items)
        max_spans = max(item["spans"].shape[0] for item in items)
        batch = {
            "input_ids": torch.full((batch_size, max_tokens), self.pad_token_id, dtype=torch.long),
            "attention_mask": torch.zeros((batch_size, max_tokens), dtype=torch.long),
            "word_ids": torch.full((batch_size, max_tokens), -1, dtype=torch.long),
            "word_mask": torch.zeros((batch_size, max_words), dtype=torch.bool),
            "spans": torch.zeros((batch_size, max_spans, 2), dtype=torch.long),
            "span_mask": torch.zeros((batch_size, max_spans), dtype=torch.bool),
            "aspect_labels": torch.full((batch_size, max_spans), -100, dtype=torch.long),
            "sentiment_labels": torch.full((batch_size, max_spans), -100, dtype=torch.long),
            "examples": [item["example"] for item in items],
        }
        pos_flags = ["word_pos_ids" in item for item in items]
        if any(pos_flags):
            from .pos_features import POS_VOCAB_SIZE

            if not all(pos_flags):
                raise ValueError("all batch items must consistently include word_pos_ids")
            batch["word_pos_ids"] = torch.zeros((batch_size, max_words), dtype=torch.long)
            for row, item in enumerate(items):
                pos_ids = item["word_pos_ids"]
                word_count = len(item["example"].words)
                if (not isinstance(pos_ids, torch.Tensor) or pos_ids.dtype != torch.long
                        or pos_ids.ndim != 1 or pos_ids.numel() != word_count):
                    raise ValueError("word_pos_ids must be a long vector aligned with words")
                if torch.any((pos_ids <= 0) | (pos_ids >= POS_VOCAB_SIZE)):
                    raise ValueError("word_pos_ids for real words must be valid non-padding POS IDs")
                batch["word_pos_ids"][row, :word_count] = pos_ids
        for row, item in enumerate(items):
            token_count = item["input_ids"].numel()
            span_count = item["spans"].shape[0]
            word_count = len(item["example"].words)
            batch["input_ids"][row, :token_count] = item["input_ids"]
            batch["attention_mask"][row, :token_count] = item["attention_mask"]
            batch["word_ids"][row, :token_count] = item["word_ids"]
            batch["word_mask"][row, :word_count] = True
            batch["spans"][row, :span_count] = item["spans"]
            batch["span_mask"][row, :span_count] = True
            batch["aspect_labels"][row, :span_count] = item["aspect_labels"]
            batch["sentiment_labels"][row, :span_count] = item["sentiment_labels"]
        image_flags = ["pixel_values" in item for item in items]
        feature_flags = ['visual_features' in item for item in items]
        if any(feature_flags):
            if not all(feature_flags) or any(image_flags):
                raise ValueError('all items must have exclusively cached visual features')
            batch['visual_features'] = torch.stack([item['visual_features'] for item in items])
        if any(image_flags):
            if not all(image_flags):
                raise ValueError("all batch items must consistently include pixel_values")
            batch["pixel_values"] = torch.stack([item["pixel_values"] for item in items])
        mismatch_flags = ["mismatch_pixel_values" in item for item in items]
        if any(mismatch_flags):
            if not all(mismatch_flags):
                raise ValueError("all batch items must consistently include mismatch_pixel_values")
            batch["mismatch_pixel_values"] = torch.stack([item["mismatch_pixel_values"] for item in items])
        return batch
