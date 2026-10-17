"""Independent target-marked sentiment encoder; never predicts boundaries."""
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from .common.backbones import text_from_config

MARKERS = ('[TGT]', '[/TGT]')


def add_markers(tokenizer):
    return tokenizer.add_special_tokens({'additional_special_tokens': list(MARKERS)})


def encode_target(tokenizer, words, start, end, max_length=258):
    if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(words)):
        raise ValueError('invalid target boundary')
    if any(marker in word for word in words for marker in MARKERS):
        raise ValueError('target marker occurs in original sentence')
    marked = words[:start] + [MARKERS[0]] + words[start:end] + [MARKERS[1]] + words[end:]
    encoded = tokenizer(marked, is_split_into_words=True, truncation=False,
                        add_special_tokens=True, return_attention_mask=True)
    ids = encoded['input_ids']
    marker_ids = tokenizer.convert_tokens_to_ids(list(MARKERS))
    if any(ids.count(m) != 1 for m in marker_ids) or len(set(marker_ids)) != 2:
        raise ValueError('tokenizer must contain two distinct target markers')
    if len(ids) > max_length:
        raise ValueError('marked sentence exceeds max_length; refusing truncation')
    if {i for i in encoded.word_ids() if i is not None} != set(range(len(marked))):
        raise ValueError('tokenization lost words')
    return dict(input_ids=ids, attention_mask=encoded['attention_mask'],
                target_index=ids.index(marker_ids[0]))


class TargetDataset(Dataset):
    def __init__(self, examples, spans, tokenizer, supervised=False, max_length=258):
        if len({e.id for e in examples}) != len(examples) or set(spans) != {e.id for e in examples}:
            raise ValueError('prediction IDs must equal unique example IDs')
        self.items, self.keys = [], []
        for example in examples:
            seen = set()
            for start, end, polarity in spans[example.id]:
                if (start, end) in seen:
                    raise ValueError('duplicate predicted boundary')
                seen.add((start, end))
                if type(polarity) is not int or polarity not in (0, 1, 2):
                    raise ValueError('invalid polarity')
                item = encode_target(tokenizer, example.words, start, end, max_length)
                if supervised:
                    item['labels'] = polarity
                self.keys.append((example.id, start, end))
                self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class TargetCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, items):
        length = max(len(x['input_ids']) for x in items)
        result = {}
        for key, pad in (('input_ids', self.pad_token_id), ('attention_mask', 0)):
            result[key] = torch.tensor([x[key] + [pad] * (length - len(x[key])) for x in items])
        result['target_index'] = torch.tensor([x['target_index'] for x in items])
        if 'labels' in items[0]:
            result['labels'] = torch.tensor([x['labels'] for x in items])
        return result


class TargetClassifier(nn.Module):
    def __init__(self, encoder, dropout=.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder.config.hidden_size, 3)

    def forward(self, input_ids, attention_mask, target_index, labels=None):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        target = hidden[torch.arange(len(hidden), device=hidden.device), target_index]
        logits = self.classifier(self.dropout(target))
        result = dict(logits=logits)
        if labels is not None:
            result['loss'] = nn.functional.cross_entropy(logits, labels)
        return result


def relabel(spans, keys, polarities):
    expected = {(key, s, e) for key, values in spans.items() for s, e, _ in values}
    if (len(keys) != len(polarities) or len(set(keys)) != len(keys) or set(keys) != expected
            or any(type(p) is not int or p not in (0, 1, 2) for p in polarities)):
        raise ValueError('relabel requires exactly one prediction per existing boundary')
    values = dict(zip(keys, polarities))
    return {key: [(s, e, values[key, s, e]) for s, e, _ in triples] for key, triples in spans.items()}


@torch.no_grad()
def predict_targets(model, loader, device):
    model.eval()
    predictions = []
    for batch in loader:
        # Inference never forwards labels, even if a supervised loader is supplied.
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        logits = model(**inputs)['logits']
        if not torch.isfinite(logits).all():
            raise FloatingPointError('nonfinite target logits')
        predictions.extend(logits.argmax(-1).tolist())
    return predictions


def save_target(path, model, metadata):
    path = Path(path)
    content = dict(format='mact_target_context_v1', text_config=model.encoder.config.to_dict(),
        text_attention=model.encoder.config._attn_implementation,
        dropout=model.dropout.p, metadata=metadata,
        state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()})
    temporary = path.with_suffix('.tmp')
    torch.save(content, temporary)
    temporary.replace(path)


def load_target(path, device='cpu'):
    content = torch.load(path, map_location='cpu', weights_only=True)
    if content.get('format') != 'mact_target_context_v1':
        raise ValueError('unsupported target checkpoint')
    model = TargetClassifier(text_from_config(content['text_config'], content['text_attention']), content['dropout'])
    model.load_state_dict(content['state_dict'], strict=True)
    return model.to(device).eval(), content['metadata']
