"""
Policy network for REAL GRPO:
- Shared encoder over state
- One policy head per PII group (3 actions: 0/1/2)
- One value head per PII group: V(s, group)
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import GROUPS, NUM_ACTIONS


class RulePolicy(nn.Module):
    """
    Multi-head policy for grouped RL:

    - Shared encoder over state
    - For each group g in GROUPS:
        * policy head: logits over NUM_ACTIONS
        * value head: scalar V(s, g)
    """

    def __init__(self, state_dim: int, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )

        # Group-specific heads
        self.group_heads = nn.ModuleDict()
        self.value_heads = nn.ModuleDict()
        for g in GROUPS.keys():
            self.group_heads[g] = nn.Linear(64, num_actions)
            self.value_heads[g] = nn.Linear(64, 1)

    def forward(self, state: torch.Tensor):
        """
        Forward pass.

        Args:
            state: Tensor of shape (batch, state_dim) or (state_dim,)

        Returns:
            logits_by_group: dict[group] -> (batch, num_actions)
            values_by_group: dict[group] -> (batch,)  (scalar V for each sample)
        """
        if state.ndim == 1:
            state = state.unsqueeze(0)  # (1, state_dim)

        emb = self.encoder(state)  # (batch, 64)

        logits_by_group: Dict[str, torch.Tensor] = {}
        values_by_group: Dict[str, torch.Tensor] = {}

        for g in self.group_heads.keys():
            logits = self.group_heads[g](emb)              # (batch, num_actions)
            value = self.value_heads[g](emb).squeeze(-1)   # (batch,)
            logits_by_group[g] = logits
            values_by_group[g] = value

        return logits_by_group, values_by_group

    @torch.no_grad()
    def act(self, state: torch.Tensor, deterministic: bool = True) -> Dict[str, int]:
        """
        Convenience method for inference / manual demo.

        Returns:
            actions_by_group: dict[group] -> int action (0/1/2)
        """
        logits_by_group, _ = self.forward(state)
        actions: Dict[str, int] = {}
        for g, logits in logits_by_group.items():
            probs = F.softmax(logits, dim=-1)
            if deterministic:
                a = int(probs.argmax(dim=-1).item())
            else:
                dist = torch.distributions.Categorical(probs=probs)
                a = int(dist.sample().item())
            actions[g] = a
        return actions
