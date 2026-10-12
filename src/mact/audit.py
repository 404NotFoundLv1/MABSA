"""Audit Train/Dev five-tag round trips and real tokenizer alignment."""

import argparse
from collections import Counter
import json
from pathlib import Path

from transformers import AutoTokenizer

from .data import load_labeled, SequenceDataset
from .labels import TAGS, tags_to_aspects
from .runtime import write_json
from .train import resolve_config


def audit_config(config):
    cfg = resolve_config(json.loads(Path(config).read_text()))
    tokenizer = AutoTokenizer.from_pretrained(cfg['text_model'], use_fast=True, local_files_only=True)
    result = {}
    for split in ('train', 'dev'):
        examples, audit = load_labeled(cfg[f'{split}_data'])
        dataset = SequenceDataset(examples, tokenizer, max_length=cfg['max_length'])
        counts = Counter()
        for item in dataset:
            tags = item['tags'].tolist()
            if tags_to_aspects(tags) != sorted(item['example'].aspects):
                raise AssertionError(f"roundtrip failed: {item['example'].id}")
            counts.update(TAGS[t] for t in tags)
        result[split] = {**audit, 'roundtrip_exact': True, 'tag_counts': dict(counts),
                         'max_words': max(len(ex.words) for ex in examples),
                         'max_tokens': max(len(x['input_ids']) for x in dataset)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    write_json(args.output, audit_config(args.config))


if __name__ == '__main__':
    main()
