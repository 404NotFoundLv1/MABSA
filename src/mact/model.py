"""Word-level text/fusion/evidence emissions with joint CRF decoding."""

from dataclasses import dataclass

import torch
from torch import nn

from .crf import ConstrainedCRF


@dataclass
class ModelConfig:
    mode: str = 'fusion'
    hidden_size: int = 256
    attention_heads: int = 4
    visual_size: int = 768
    dropout: float = .1
    visual_dropout: float = .15

    def __post_init__(self):
        if self.mode not in ('text', 'fusion', 'evidence'):
            raise ValueError('mode must be text, fusion, or evidence')
        if (self.hidden_size < 1 or self.attention_heads < 1 or self.visual_size < 1
                or self.hidden_size % self.attention_heads):
            raise ValueError('invalid hidden/visual size or attention heads')
        if not 0 <= self.dropout < 1 or not 0 <= self.visual_dropout <= 1:
            raise ValueError('invalid dropout probability')


def pool_words(tokens, word_ids, word_mask):
    valid = word_ids >= 0
    indices = word_ids.clamp_min(0)
    shape = (*word_mask.shape, tokens.shape[-1])
    pooled = tokens.new_zeros(shape)
    count = tokens.new_zeros(*word_mask.shape, 1)
    pooled.scatter_add_(1, indices.unsqueeze(-1).expand_as(tokens), tokens * valid.unsqueeze(-1))
    count.scatter_add_(1, indices.unsqueeze(-1), valid.unsqueeze(-1).to(tokens.dtype))
    return (pooled / count.clamp_min(1)) * word_mask.unsqueeze(-1)


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
        return output
