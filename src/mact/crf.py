"""Linear-chain CRF with exact hard five-tag constraints in NLL and Viterbi."""

import torch
from torch import nn


class ConstrainedCRF(nn.Module):
    def __init__(self):
        super().__init__()
        self.start_transitions = nn.Parameter(torch.zeros(5))
        self.end_transitions = nn.Parameter(torch.zeros(5))
        # [previous, next]
        self.transitions = nn.Parameter(torch.zeros(5, 5))
        allowed = torch.ones(5, 5, dtype=torch.bool)
        allowed[0, 4] = False
        self.register_buffer("allowed", allowed)
        self.register_buffer("allowed_start", torch.tensor([1, 1, 1, 1, 0], dtype=torch.bool))

    def _validate(self, emissions, mask, tags=None):
        if (emissions.ndim != 3 or emissions.shape[-1] != 5 or not emissions.is_floating_point()
                or mask.shape != emissions.shape[:2] or mask.dtype != torch.bool
                or emissions.shape[0] == 0 or emissions.shape[1] == 0):
            raise ValueError("expected emissions [B,W,5] and boolean mask [B,W]")
        if not mask[:, 0].all() or (mask[:, 1:] & ~mask[:, :-1]).any():
            raise ValueError("mask must contain nonempty contiguous word prefixes")
        if tags is not None:
            if tags.shape != mask.shape or tags.dtype != torch.long:
                raise ValueError("tags must be long [B,W]")
            if ((tags[mask] < 0) | (tags[mask] >= 5)).any():
                raise ValueError("invalid real-word tags")
            if (tags[:, 0] == 4).any() or (((tags[:, :-1] == 0) & (tags[:, 1:] == 4)) & mask[:, 1:]).any():
                raise ValueError("illegal gold tag transition")

    def _scores(self):
        return (self.start_transitions.masked_fill(~self.allowed_start, -torch.inf),
                self.transitions.masked_fill(~self.allowed, -torch.inf))

    def forward(self, emissions, tags, mask):
        """Return one negative log likelihood per sentence, without reduction."""
        # Keep partition math at least FP32 even if encoders use mixed precision.
        emissions = emissions.float() if emissions.dtype in (torch.float16, torch.bfloat16) else emissions
        self._validate(emissions, mask, tags)
        start, transitions = self._scores()
        alpha = emissions[:, 0] + start
        safe_tags = tags.masked_fill(~mask, 0)
        rows = torch.arange(emissions.shape[0], device=emissions.device)
        gold = start[safe_tags[:, 0]] + emissions[rows, 0, safe_tags[:, 0]]
        for t in range(1, emissions.shape[1]):
            next_alpha = torch.logsumexp(alpha.unsqueeze(2) + transitions.unsqueeze(0), dim=1) + emissions[:, t]
            alpha = torch.where(mask[:, t, None], next_alpha, alpha)
            step = transitions[safe_tags[:, t - 1], safe_tags[:, t]] + emissions[rows, t, safe_tags[:, t]]
            gold = gold + torch.where(mask[:, t], step, 0.)
        last = safe_tags.gather(1, (mask.sum(1) - 1).unsqueeze(1)).squeeze(1)
        gold = gold + self.end_transitions[last]
        return torch.logsumexp(alpha + self.end_transitions, dim=1) - gold

    @torch.no_grad()
    def decode(self, emissions, mask):
        # One device transfer, not a synchronization per word during traceback.
        emissions, mask = emissions.detach().cpu(), mask.cpu()
        self._validate(emissions, mask)
        start, transitions = (x.detach().cpu() for x in self._scores())
        end = self.end_transitions.detach().cpu()
        result = []
        for row, length in zip(emissions, mask.sum(1).tolist()):
            score = start + row[0]
            history = []
            for t in range(1, length):
                score, previous = (score[:, None] + transitions).max(0)
                score = score + row[t]
                history.append(previous)
            path = [int((score + end).argmax())]
            for previous in reversed(history):
                path.append(int(previous[path[-1]]))
            result.append(list(reversed(path)))
        return result
