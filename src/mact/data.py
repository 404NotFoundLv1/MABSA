"""Word-aligned sequence data. No span candidates are constructed."""

import json
from pathlib import Path

import torch

from .common.data import SpanDataset, load_examples
from .labels import aspects_to_tags, get_tags


def load_labeled(path):
    records = json.loads(Path(path).read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("labeled dataset must be a nonempty list")
    if any(not isinstance(r, dict) or not isinstance(r.get('aspects'), list) for r in records):
        raise ValueError("training/evaluation requires explicit aspects lists; [] means no aspects")
    return load_examples(path, conflict_policy='error')


class SequenceDataset(SpanDataset):
    # Reuse only lazy image I/O and the constructor; override all tokenization
    # and annotation behavior, so the base never enumerates spans.
    def __init__(self, examples, tokenizer, image_root=None, image_processor=None,
                 max_length=256, label_scheme='five', ocr_texts=None, ocr_max_tokens=64):
        get_tags(label_scheme)
        self.label_scheme = label_scheme
        if type(ocr_max_tokens) is not int or ocr_max_tokens < 1:
            raise ValueError('ocr_max_tokens must be a positive integer')
        self.ocr_texts, self.ocr_max_tokens = ocr_texts, ocr_max_tokens
        super().__init__(examples, tokenizer, image_root, image_processor, max_length)

    def _tokenize(self, example):
        if example.ignored_spans:
            raise ValueError("sequence labeling does not permit ignored/conflicting spans")
        encoded = self.tokenizer(example.words, is_split_into_words=True,
                                 add_special_tokens=True, truncation=False,
                                 return_attention_mask=True)
        word_ids = encoded.word_ids()
        if len(encoded['input_ids']) > self.max_length:
            raise ValueError(f"example {example.id} tokenized length exceeds max_length")
        if {i for i in word_ids if i is not None} != set(range(len(example.words))):
            raise ValueError(f"example {example.id} lost words during tokenization")
        ids, mask = list(encoded['input_ids']), list(encoded['attention_mask'])
        if self.ocr_texts is not None:
            if example.image_id not in self.ocr_texts:
                raise ValueError(f'missing OCR image: {example.image_id}')
            text = self.ocr_texts[example.image_id]
            if not isinstance(text, str):
                raise ValueError('OCR text must be a string')
            if text.strip():
                extra = self.tokenizer(text, add_special_tokens=False, truncation=False)['input_ids'][:self.ocr_max_tokens]
                if extra:
                    if self.tokenizer.sep_token_id is None:
                        raise ValueError('OCR context requires a separator token')
                    extra = extra + [self.tokenizer.sep_token_id]
                    ids += extra; mask += [1] * len(extra); word_ids += [None] * len(extra)
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'word_ids': torch.tensor([-1 if i is None else i for i in word_ids], dtype=torch.long),
            'tags': torch.tensor(aspects_to_tags(len(example.words), example.aspects, self.label_scheme),
                                 dtype=torch.long),
            'example': example,
        }


class SequenceCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, items):
        if not items:
            raise ValueError("empty batch")
        size = len(items)
        tokens = max(len(x['input_ids']) for x in items)
        words = max(len(x['example'].words) for x in items)
        batch = {
            'input_ids': torch.full((size, tokens), self.pad_token_id, dtype=torch.long),
            'attention_mask': torch.zeros(size, tokens, dtype=torch.long),
            'word_ids': torch.full((size, tokens), -1, dtype=torch.long),
            'word_mask': torch.zeros(size, words, dtype=torch.bool),
            'tags': torch.full((size, words), -100, dtype=torch.long),
            'examples': [x['example'] for x in items],
        }
        for row, item in enumerate(items):
            for key in ('input_ids', 'attention_mask', 'word_ids', 'tags'):
                batch[key][row, :len(item[key])] = item[key]
            batch['word_mask'][row, :len(item['example'].words)] = True
        for key in ('visual_features', 'pixel_values'):
            if any(key in x for x in items):
                if not all(key in x for x in items):
                    raise ValueError(f"inconsistent batch {key}")
                batch[key] = torch.stack([x[key] for x in items])
        return batch
