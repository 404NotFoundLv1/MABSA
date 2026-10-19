"""Explicit aspect and target-conditioned context representations; immutable spans."""
from collections import Counter
from pathlib import Path
import math

import torch
from torch import nn

from .common.backbones import text_from_config
from .target_context import MARKERS, TargetDataset, TargetCollator

VARIANTS = ('marker', 'span', 'context')
FORMAT = 'mact_target_representation_v1'


class RepresentationDataset(TargetDataset):
    def __init__(self, examples, spans, tokenizer, supervised=False, max_length=258):
        super().__init__(examples, spans, tokenizer, supervised, max_length)
        by_id = {e.id: e for e in examples}
        for item, (identifier, start, end) in zip(self.items, self.keys):
            words = by_id[identifier].words
            marked = words[:start] + [MARKERS[0]] + words[start:end] + [MARKERS[1]] + words[end:]
            encoded = tokenizer(marked, is_split_into_words=True, truncation=False,
                                add_special_tokens=True, return_attention_mask=True)
            if encoded['input_ids'] != item['input_ids']:
                raise ValueError('target tokenization changed during alignment')
            word_ids = encoded.word_ids()
            counts = Counter(w for w in word_ids if w is not None)
            target_words = set(range(start + 1, end + 1))
            item['span_weights'] = [1. / ((end-start) * counts[w]) if w in target_words else 0.
                                    for w in word_ids]
            item['context_mask'] = [w is not None and w not in target_words and w not in (start, end+1)
                                    for w in word_ids]
            if not math.isclose(sum(item['span_weights']), 1., abs_tol=1e-6):
                raise ValueError('target word pooling requires a nonempty aligned span')


class RepresentationCollator(TargetCollator):
    def __call__(self, items):
        result = super().__call__(items)
        length = result['input_ids'].shape[1]
        for key, dtype, pad in [('span_weights', torch.float32, 0.), ('context_mask', torch.bool, False)]:
            result[key] = torch.tensor([x[key] + [pad] * (length-len(x[key])) for x in items], dtype=dtype)
        return result


class RepresentationClassifier(nn.Module):
    def __init__(self, encoder, variant, dropout=.1, attention_dim=128):
        super().__init__()
        if variant not in VARIANTS or type(attention_dim) is not int or attention_dim < 1:
            raise ValueError('invalid representation variant or attention dimension')
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError('invalid dropout')
        self.encoder, self.variant, self.attention_dim = encoder, variant, attention_dim
        self.dropout = nn.Dropout(dropout)
        hidden = encoder.config.hidden_size
        # Marker mode has exactly the legacy parameter names, order and RNG draws.
        self.classifier = nn.Linear(hidden * (VARIANTS.index(variant) + 1), 3)
        if variant == 'context':
            self.query = nn.Linear(hidden, attention_dim, bias=False)
            self.key = nn.Linear(hidden, attention_dim, bias=False)

    def attend_context(self, hidden, aspect, context_mask):
        if context_mask.dtype != torch.bool or context_mask.shape != hidden.shape[:2]:
            raise ValueError('context mask must be boolean [batch, tokens]')
        scores = torch.einsum('bd,btd->bt', self.query(aspect), self.key(hidden)) / math.sqrt(self.attention_dim)
        # Finite fill avoids NaNs when an entire sentence consists of its target.
        weights = scores.masked_fill(~context_mask, torch.finfo(scores.dtype).min).softmax(-1)
        weights = weights * context_mask
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        return torch.einsum('bt,bth->bh', weights, hidden), weights

    def forward(self, input_ids, attention_mask, target_index, span_weights=None, context_mask=None, labels=None):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        marker = hidden[torch.arange(len(hidden), device=hidden.device), target_index]
        parts = [marker]
        if self.variant != 'marker':
            if (span_weights is None or span_weights.shape != input_ids.shape
                    or not torch.isfinite(span_weights).all() or (span_weights < 0).any()
                    or not torch.allclose(span_weights.sum(-1), torch.ones_like(span_weights[:, 0]), atol=1e-6)
                    or (span_weights.masked_select(~attention_mask.bool()) != 0).any()):
                raise ValueError('invalid span pooling weights')
            aspect = torch.einsum('bt,bth->bh', span_weights.to(hidden.dtype), hidden)
            parts.append(aspect)
            if self.variant == 'context':
                if context_mask is None:
                    raise ValueError('context mask is required')
                if context_mask.dtype != torch.bool or context_mask.shape != input_ids.shape:
                    raise ValueError('invalid context mask')
                if (context_mask & (~attention_mask.bool() | (span_weights > 0))).any():
                    raise ValueError('context overlaps target or padding')
                context, _ = self.attend_context(hidden, aspect, context_mask)
                parts.append(context)
        logits = self.classifier(self.dropout(torch.cat(parts, -1)))
        result = dict(logits=logits)
        if labels is not None:
            result['loss'] = nn.functional.cross_entropy(logits, labels)
        return result


def save_representation(path, model, metadata):
    path = Path(path)
    content = dict(format=FORMAT, variant=model.variant, attention_dim=model.attention_dim,
        text_config=model.encoder.config.to_dict(), text_attention=model.encoder.config._attn_implementation,
        dropout=model.dropout.p, metadata=metadata,
        state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()})
    temporary = path.with_suffix('.tmp')
    torch.save(content, temporary)
    temporary.replace(path)


def load_representation(path, device='cpu'):
    content = torch.load(path, map_location='cpu', weights_only=True)
    if content.get('format') != FORMAT:
        raise ValueError('unsupported representation checkpoint')
    model = RepresentationClassifier(text_from_config(content['text_config'], content['text_attention']),
        content['variant'], content['dropout'], content['attention_dim'])
    model.load_state_dict(content['state_dict'], strict=True)
    return model.to(device).eval(), content['metadata']
