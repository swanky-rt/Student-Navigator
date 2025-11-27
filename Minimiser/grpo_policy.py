# grpo_policy.py
"""
Policy network for the GRPO Rule Agent (learned mask version):
- Shared MLP encoder over state
- Single head with 11 Bernoulli outputs (one per PII type)
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import NUM_PII


class RulePolicy(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, NUM_PII)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: (batch, state_dim) or (state_dim,)
        returns logits: (batch, NUM_PII)
        """
        if state.ndim == 1:
            state = state.unsqueeze(0)
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        logits = self.out(x)
        return logits  # no sigmoid here

    @torch.no_grad()
    def act(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[List[int], float]:
        """
        Sample or pick a deterministic binary share_mask for the given state.

        Returns:
          share_mask: list[int] of length NUM_PII (0 = hide, 1 = share)
          log_prob:  log-probability of the sampled action (0.0 in deterministic mode)
        """
        logits = self.forward(state)  # (1, NUM_PII)
        probs = torch.sigmoid(logits)  # (1, NUM_PII)
        dist = torch.distributions.Bernoulli(probs=probs)

        if deterministic:
            # Just take the most likely action per field
            actions = (probs >= 0.5).float()  # keep as float for consistency
            # We don't really need log_prob in eval, so set it to 0.0
            log_prob = 0.0
        else:
            # Sample stochastic mask during training
            actions = dist.sample()           # float 0/1
            log_prob = dist.log_prob(actions).sum(dim=-1)[0].item()

        # Convert to Python ints for convenience
        share_mask = actions.squeeze(0).round().tolist()
        share_mask = [int(x) for x in share_mask]

        return share_mask, float(log_prob)
