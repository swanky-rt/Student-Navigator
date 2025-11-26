"""
Policy network for the GRPO Rule Agent:
- Shared MLP encoder over state
- One head per group producing logits over 3 actions
- One value head per group for GRPO / PPO advantage estimation
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import GROUPS, NUM_ACTIONS


class RulePolicy(nn.Module):
    """
    Multi-head policy:
      - shared encoder over state
      - one classification head per group (3 actions)
      - one scalar value head per group V(s, g)
    """

    def __init__(self, state_dim: int, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions

        # Shared encoder over the state
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )

        # One policy head and one value head per group
        self.group_heads = nn.ModuleDict()
        self.value_heads = nn.ModuleDict()
        for g in GROUPS.keys():
            self.group_heads[g] = nn.Linear(64, num_actions)
            self.value_heads[g] = nn.Linear(64, 1)

    def forward(self, state: torch.Tensor):
        """
        Forward pass.

        Args:
            state: (batch, state_dim) or (state_dim,)

        Returns:
            logits_by_group: dict[group_name] -> (batch, num_actions)
            values_by_group: dict[group_name] -> (batch,)
        """
        if state.ndim == 1:
            state = state.unsqueeze(0)  # (1, state_dim)

        emb = self.encoder(state)  # (batch, 64)

        logits_by_group: Dict[str, torch.Tensor] = {}
        values_by_group: Dict[str, torch.Tensor] = {}

        for g in self.group_heads.keys():
            logits = self.group_heads[g](emb)          # (batch, num_actions)
            value = self.value_heads[g](emb).squeeze(-1)  # (batch,)

            logits_by_group[g] = logits
            values_by_group[g] = value

        return logits_by_group, values_by_group

    @torch.no_grad()
    def act(self, state: torch.Tensor, deterministic: bool = True) -> Dict[str, int]:
        """
        Convenience helper for inference / manual decisions.
        Returns one discrete action (0/1/2) per group.
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
