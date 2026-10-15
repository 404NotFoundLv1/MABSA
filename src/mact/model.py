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
    visual_gate: bool = False
    visual_gate_init: float = .9
    label_scheme: str = 'five'
    word_context: bool = False
    word_context_size: int = 64
    word_context_heads: int = 4
    word_context_max_distance: int = 16

    def __post_init__(self):
        if type(self.label_scheme) is not str or self.label_scheme not in ('five', 'seven'):
            raise ValueError('label_scheme must be five or seven')
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
        if type(self.visual_gate) is not bool or (self.visual_gate and self.mode != 'fusion'):
            raise ValueError('visual_gate must be boolean and requires fusion mode')
        if (type(self.visual_gate_init) not in (int, float)
                or not math.isfinite(self.visual_gate_init) or not 0 < self.visual_gate_init < 1):
            raise ValueError('visual_gate_init must be finite and strictly between 0 and 1')
        if type(self.word_context) is not bool:
            raise ValueError('word_context must be boolean')
        if (any(type(v) is not int or v < 1 for v in
                (self.word_context_size, self.word_context_heads, self.word_context_max_distance))
                or self.word_context_size % self.word_context_heads):
            raise ValueError('invalid word context size, heads or distance')
        if self.word_context and (self.mode != 'fusion' or self.label_scheme != 'five'
                                  or self.visual_gate or self.aspect_sentiment_weight > 0):
            raise ValueError('word_context requires five-label fusion with B and C disabled')


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
        members = (begins | (tags >= 4)) & word_mask
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


class WordContextResidual(nn.Module):
    """Read fused word features and return a zero-initialized five-score residual.

    Relative distance is key position minus query position, clipped at the
    configured radius. A mild local preference initializes a trainable bias
    per head; all valid words, including the query itself, remain accessible.
    No additional dropout is used, preserving the baseline's training RNG.
    """

    def __init__(self, hidden_size, context_size, heads, max_distance):
        super().__init__()
        self.heads, self.max_distance = heads, max_distance
        self.head_size = context_size // heads
        self.q = nn.Linear(hidden_size, context_size, bias=False)
        self.k = nn.Linear(hidden_size, context_size, bias=False)
        self.v = nn.Linear(hidden_size, context_size, bias=False)
        distances = torch.arange(-max_distance, max_distance + 1, dtype=torch.float32)
        self.relative_bias = nn.Parameter((-distances.abs() / max_distance).repeat(heads, 1))
        self.output = nn.Linear(context_size, 5)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, fused_words, word_mask):
        if not word_mask.any(dim=1).all():
            raise ValueError('word context requires at least one valid word per sentence')
        batch, length, _ = fused_words.shape
        def split(projected):
            return projected.reshape(batch, length, self.heads, self.head_size).transpose(1, 2)
        query, key, value = (split(layer(fused_words)) for layer in (self.q, self.k, self.v))
        positions = torch.arange(length, device=fused_words.device)
        distances = (positions[None, :] - positions[:, None]).clamp(
            -self.max_distance, self.max_distance) + self.max_distance
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_size)
        scores = scores + self.relative_bias[:, distances][None]
        scores = scores.masked_fill(~word_mask[:, None, None, :], -torch.inf)
        context = (scores.softmax(dim=-1) @ value).transpose(1, 2).reshape(batch, length, -1)
        return self.output(context) * word_mask.unsqueeze(-1)


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
        if config.visual_gate:
            self.visual_gate = nn.Linear(h * 2, 1)
            nn.init.zeros_(self.visual_gate.weight)
            nn.init.constant_(self.visual_gate.bias, math.log(config.visual_gate_init / (1 - config.visual_gate_init)))
        if config.label_scheme == 'seven':
            # Construct the original five-label model first, then untie its I
            # state. Common weights AND the post-init RNG state are preserved.
            self.text_head = expand_inside_emissions(self.text_head)
            if config.mode != 'text':
                self.visual_head[-1] = expand_inside_emissions(self.visual_head[-1])
            expanded = ConstrainedCRF('seven')
            mapping = torch.tensor([0, 1, 2, 3, 4, 4, 4])
            with torch.no_grad():
                expanded.start_transitions.copy_(self.crf.start_transitions[mapping])
                expanded.end_transitions.copy_(self.crf.end_transitions[mapping])
                expanded.transitions.copy_(self.crf.transitions[mapping][:, mapping])
            self.crf = expanded
        if config.word_context:
            # Append without perturbing common initialization or the subsequent
            # data/dropout RNG stream. All model parameters are constructed on CPU.
            with torch.random.fork_rng(devices=[]):
                self.word_context = WordContextResidual(h, config.word_context_size,
                    config.word_context_heads, config.word_context_max_distance)

    def forward(self, input_ids, attention_mask, word_ids, word_mask,
                visual_features=None, tags=None, force_null=False,
                visual_drop_mask=None, return_visual_drop_mask=False, return_words=False):
        if visual_drop_mask is not None:
            if (self.config.mode == 'text' or force_null
                    or visual_drop_mask.dtype != torch.bool
                    or visual_drop_mask.shape != (input_ids.shape[0],)
                    or visual_drop_mask.device != input_ids.device):
                raise ValueError('visual_drop_mask must be boolean [B] on the input device; fusion only, without force_null')
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
            elif visual_drop_mask is not None:
                drop = visual_drop_mask
            elif self.training and self.config.visual_dropout:
                drop = torch.rand(words.shape[0], device=words.device) < self.config.visual_dropout
            mask[:, :-1] = drop[:, None]
            context = (words * word_mask.unsqueeze(-1)).sum(1) / word_mask.sum(1, keepdim=True)
            query = self.query(torch.cat([words, context[:, None].expand_as(words)], dim=-1))
            actual = self.attention(query, memory, memory, key_padding_mask=mask, need_weights=False)[0]
            fused_input = torch.cat([words, actual], dim=-1)
            if self.config.word_context:
                fused_words = self.visual_head[1](self.visual_head[0](fused_input))
                actual_score = self.visual_head[2](fused_words)
            else:
                actual_score = self.visual_head(fused_input)
            if self.config.mode == 'evidence' or self.config.visual_gate:
                # Same padded memory layout and scoring function, only the mask differs.
                null_mask = torch.ones_like(mask)
                null_mask[:, -1] = False
                reference = self.attention(query, memory, memory, key_padding_mask=null_mask, need_weights=False)[0]
                null_score = self.visual_head(torch.cat([words, reference], dim=-1))
                if self.config.visual_gate:
                    # One scalar per word controls the image-dependent increment.
                    # The null path retains the text-dependent visual head.
                    gate = self.visual_gate(torch.cat([words, actual - reference], dim=-1)).sigmoid()
                    null_emissions = text_emissions + null_score
                    emissions = null_emissions + gate * (actual_score - null_score)
                else:
                    # Group subtraction before adding text preserves exact force-null identity.
                    emissions = text_emissions + (actual_score - null_score)
            else:
                emissions = text_emissions + actual_score
            if self.config.word_context:
                emissions = emissions + self.word_context(fused_words, word_mask)
        output = {'emissions': emissions, 'text_emissions': text_emissions}
        if return_words:
            output['words'] = words
        if return_visual_drop_mask and self.config.mode != 'text':
            output['visual_drop_mask'] = drop
        if self.config.visual_gate:
            output.update(visual_gate=gate, null_emissions=null_emissions)
        if tags is not None:
            output['losses'] = self.crf(emissions, tags, word_mask)
            if self.config.aspect_sentiment_weight > 0:
                output['crf_losses'] = output['losses']
                output['aspect_sentiment_losses'] = self.aspect_sentiment_head(words, tags, word_mask)
                output['losses'] = (output['crf_losses']
                                    + self.config.aspect_sentiment_weight * output['aspect_sentiment_losses'])
        return output


def expand_inside_emissions(layer):
    """Copy the shared I row into three independently trainable polarity rows."""
    devices = [layer.weight.device.index] if layer.weight.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        expanded = nn.Linear(layer.in_features, 7, bias=layer.bias is not None,
                             device=layer.weight.device, dtype=layer.weight.dtype)
    mapping = torch.tensor([0, 1, 2, 3, 4, 4, 4], device=layer.weight.device)
    with torch.no_grad():
        expanded.weight.copy_(layer.weight[mapping])
        if layer.bias is not None:
            expanded.bias.copy_(layer.bias[mapping])
    return expanded
