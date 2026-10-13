"""Linear-chain CRF with exact hard tag constraints in NLL and Viterbi."""

import torch
from torch import nn

from .labels import get_tags


class ConstrainedCRF(nn.Module):
    def __init__(self, label_scheme="five"):
        super().__init__()
        self.num_tags = len(get_tags(label_scheme))
        self.label_scheme = label_scheme
        self.start_transitions = nn.Parameter(torch.zeros(self.num_tags))
        self.end_transitions = nn.Parameter(torch.zeros(self.num_tags))
        # [previous, next]
        self.transitions = nn.Parameter(torch.zeros(self.num_tags, self.num_tags))
        allowed = torch.ones(self.num_tags, self.num_tags, dtype=torch.bool)
        if label_scheme == "five":
            allowed[0, 4] = False
        else:
            allowed[:, 4:] = False
            for inside in range(4, self.num_tags):
                allowed[inside - 3, inside] = True
                allowed[inside, inside] = True
        self.register_buffer("allowed", allowed)
        self.register_buffer("allowed_start", torch.arange(self.num_tags) < 4)

    def _validate(self, emissions, mask, tags=None):
        if (emissions.ndim != 3 or emissions.shape[-1] != self.num_tags or not emissions.is_floating_point()
                or mask.shape != emissions.shape[:2] or mask.dtype != torch.bool
                or emissions.shape[0] == 0 or emissions.shape[1] == 0):
            raise ValueError(f"expected emissions [B,W,{self.num_tags}] and boolean mask [B,W]")
        if not mask[:, 0].all() or (mask[:, 1:] & ~mask[:, :-1]).any():
            raise ValueError("mask must contain nonempty contiguous word prefixes")
        if tags is not None:
            if tags.shape != mask.shape or tags.dtype != torch.long:
                raise ValueError("tags must be long [B,W]")
            if ((tags[mask] < 0) | (tags[mask] >= self.num_tags)).any():
                raise ValueError("invalid real-word tags")
            safe_tags = tags.masked_fill(~mask, 0)
            if ((~self.allowed_start[safe_tags[:, 0]]).any()
                    or ((~self.allowed[safe_tags[:, :-1], safe_tags[:, 1:]]) & mask[:, 1:]).any()):
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

    def marginals(self, emissions, mask):
        """Differentiable node marginals of the constrained sequence distribution."""
        emissions = emissions.float() if emissions.dtype in (torch.float16, torch.bfloat16) else emissions
        self._validate(emissions, mask)
        start, transitions = self._scores()
        alpha = emissions[:, 0] + start
        alphas = [alpha]
        for t in range(1, emissions.shape[1]):
            next_alpha = torch.logsumexp(alpha.unsqueeze(2) + transitions, dim=1) + emissions[:, t]
            alpha = torch.where(mask[:, t, None], next_alpha, alpha)
            alphas.append(alpha)
        log_z = torch.logsumexp(alpha + self.end_transitions, dim=1)
        beta = self.end_transitions.expand_as(alpha)
        betas = [beta]
        for t in range(emissions.shape[1] - 2, -1, -1):
            next_beta = torch.logsumexp(transitions + (emissions[:, t + 1] + beta).unsqueeze(1), dim=2)
            beta = torch.where(mask[:, t + 1, None], next_beta, beta)
            betas.append(beta)
        log_marginals = (torch.stack(alphas, dim=1) + torch.stack(betas[::-1], dim=1)
                         - log_z[:, None, None])
        return log_marginals.exp().masked_fill(~mask.unsqueeze(-1), 0.)

    def symmetric_kl(self, first, second, mask):
        """Half of KL(P||Q)+KL(Q||P), one value per sentence.

        P and Q share this CRF's transitions and hard support. Their transition
        and log-partition terms cancel in the symmetric sum, leaving differences
        of emissions weighted by differences of node marginals. Marginals stay
        attached so gradients through BOTH distributions and transitions remain.
        """
        if first.shape != second.shape or first.device != second.device:
            raise ValueError('paired emissions must have the same shape and device')
        first = first.float() if first.dtype in (torch.float16, torch.bfloat16) else first
        second = second.float() if second.dtype in (torch.float16, torch.bfloat16) else second
        delta = (first - second).masked_fill(~mask.unsqueeze(-1), 0.)
        return .5 * ((self.marginals(first, mask) - self.marginals(second, mask)) * delta).sum((1, 2))

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
