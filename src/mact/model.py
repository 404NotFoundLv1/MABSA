"""Word-level text/fusion/evidence emissions with joint CRF decoding."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .crf import ConstrainedCRF


@dataclass
class ModelConfig:
    mode: str = 'fusion'
    hidden_size: int = 256
    attention_heads: int = 4
    visual_size: int = 768
    dropout: float = .1
    visual_dropout: float = .15
    aspect_sentiment_weight: float = 0.

    def __post_init__(self):
        if self.mode not in ('text', 'fusion', 'evidence'):
            raise ValueError('mode must be text, fusion, or evidence')
        if (self.hidden_size < 1 or self.attention_heads < 1 or self.visual_size < 1
                or self.hidden_size % self.attention_heads):
            raise ValueError('invalid hidden/visual size or attention heads')
        if not 0 <= self.dropout < 1 or not 0 <= self.visual_dropout <= 1:
            raise ValueError('invalid dropout probability')
        if (type(self.aspect_sentiment_weight) not in (int, float)
                or not math.isfinite(self.aspect_sentiment_weight)
                or self.aspect_sentiment_weight < 0):
            raise ValueError('aspect_sentiment_weight must be finite and nonnegative')


def pool_words(tokens, word_ids, word_mask):
    valid = word_ids >= 0
    indices = word_ids.clamp_min(0)
    shape = (*word_mask.shape, tokens.shape[-1])
    pooled = tokens.new_zeros(shape)
    count = tokens.new_zeros(*word_mask.shape, 1)
    pooled.scatter_add_(1, indices.unsqueeze(-1).expand_as(tokens), tokens * valid.unsqueeze(-1))
    count.scatter_add_(1, indices.unsqueeze(-1), valid.unsqueeze(-1).to(tokens.dtype))
    return (pooled / count.clamp_min(1)) * word_mask.unsqueeze(-1)


class AspectSentimentHead(nn.Module):
    """Mean-pool gold training aspects; return mean aspect CE per sentence.

    Class order is NEG, NEU, POS. No gold spans or auxiliary predictions are
    consumed by the emission heads or CRF decoder.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, 3)

    def forward(self, words, tags, word_mask):
        begins = (tags >= 1) & (tags <= 3) & word_mask
        members = (begins | (tags == 4)) & word_mask
        indices = begins.long().cumsum(dim=1)
        batch, length, hidden = words.shape
        # Slot zero is the prefix before the first aspect. Mask O and padding
        # explicitly: cumsum alone would include inter-aspect/background words.
        pooled = words.new_zeros(batch, length + 1, hidden)
        count = words.new_zeros(batch, length + 1)
        pooled.scatter_add_(1, indices.unsqueeze(-1).expand_as(words),
                            words * members.unsqueeze(-1))
        count.scatter_add_(1, indices, members.to(words.dtype))
        pooled = pooled[:, 1:] / count[:, 1:].clamp_min(1).unsqueeze(-1)
        targets = tags.new_zeros(batch, length + 1)
        targets.scatter_add_(1, indices, (3 - tags).clamp(0, 2) * begins)
        logits = self.classifier(pooled)
        ce = F.cross_entropy(logits.reshape(-1, 3), targets[:, 1:].reshape(-1),
                             reduction='none').reshape(batch, length)
        valid = count[:, 1:] > 0
        return (ce * valid).sum(1) / valid.sum(1).clamp_min(1)


class SequenceModel(nn.Module):
    def __init__(self, text_encoder, config):
        super().__init__()
        self.text_encoder, self.config = text_encoder, config
        h = config.hidden_size
        self.word_projection = nn.Sequential(nn.Linear(text_encoder.config.hidden_size, h),
                                              nn.GELU(), nn.LayerNorm(h), nn.Dropout(config.dropout))
        self.text_head = nn.Linear(h, 5)
        if config.mode != 'text':
            self.visual_projection = nn.Sequential(nn.Linear(config.visual_size, h), nn.LayerNorm(h))
            self.null_evidence = nn.Parameter(torch.zeros(1, 1, h))
            self.query = nn.Linear(h * 2, h)
            # Deterministic paired scoring is needed for exact null cancellation.
            self.attention = nn.MultiheadAttention(h, config.attention_heads, dropout=0., batch_first=True)
            self.visual_head = nn.Sequential(nn.Linear(h * 2, h), nn.GELU(), nn.Linear(h, 5))
        self.crf = ConstrainedCRF()
        # Append after all baseline parameters; disabled models retain the
        # original parameter set and initialization RNG consumption exactly.
        if config.aspect_sentiment_weight > 0:
            self.aspect_sentiment_head = AspectSentimentHead(h)

    def forward(self, input_ids, attention_mask, word_ids, word_mask,
                visual_features=None, tags=None, force_null=False):
        hidden = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        words = self.word_projection(pool_words(hidden, word_ids, word_mask))
        text_emissions = self.text_head(words)
        emissions = text_emissions
        if self.config.mode != 'text':
            if visual_features is None or visual_features.ndim != 3 or visual_features.shape[0] != words.shape[0]:
                raise ValueError('multimodal tagging requires raw cached visual tokens [B,P,V]')
            visual = self.visual_projection(visual_features)
            null = self.null_evidence.expand(words.shape[0], -1, -1)
            memory = torch.cat([visual, null], dim=1)
            mask = torch.zeros(memory.shape[:2], dtype=torch.bool, device=words.device)
            drop = torch.zeros(words.shape[0], dtype=torch.bool, device=words.device)
            if force_null:
                drop[:] = True
            elif self.training and self.config.visual_dropout:
                drop = torch.rand(words.shape[0], device=words.device) < self.config.visual_dropout
            mask[:, :-1] = drop[:, None]
            context = (words * word_mask.unsqueeze(-1)).sum(1) / word_mask.sum(1, keepdim=True)
            query = self.query(torch.cat([words, context[:, None].expand_as(words)], dim=-1))
            actual = self.attention(query, memory, memory, key_padding_mask=mask, need_weights=False)[0]
            actual_score = self.visual_head(torch.cat([words, actual], dim=-1))
            if self.config.mode == 'evidence':
                # Same padded memory layout and scoring function, only the mask differs.
                null_mask = torch.ones_like(mask)
                null_mask[:, -1] = False
                reference = self.attention(query, memory, memory, key_padding_mask=null_mask, need_weights=False)[0]
                null_score = self.visual_head(torch.cat([words, reference], dim=-1))
                # Group subtraction before adding text preserves exact force-null identity.
                emissions = text_emissions + (actual_score - null_score)
            else:
                emissions = text_emissions + actual_score
        output = {'emissions': emissions, 'text_emissions': text_emissions}
        if tags is not None:
            output['losses'] = self.crf(emissions, tags, word_mask)
            if self.config.aspect_sentiment_weight > 0:
                output['crf_losses'] = output['losses']
                output['aspect_sentiment_losses'] = self.aspect_sentiment_head(words, tags, word_mask)
                output['losses'] = (output['crf_losses']
                                    + self.config.aspect_sentiment_weight * output['aspect_sentiment_losses'])
        return output
