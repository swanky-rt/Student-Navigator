"""
Policy network for the GRPO Rule Agent:
- Shared MLP encoder over state
- One head per group producing logits over 3 actions
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

        # One head per group
        self.group_heads = nn.ModuleDict()
        for g in GROUPS.keys():
            self.group_heads[g] = nn.Linear(64, num_actions)

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Input:
          state: (batch, state_dim)
        Output:
          logits_by_group: dict {group_name -> (batch, num_actions)}
        """
        emb = self.encoder(state)
        logits_by_group = {
            g: head(emb) for g, head in self.group_heads.items()
        }
        return logits_by_group

    @torch.no_grad()
    def act(self, state: torch.Tensor, deterministic: bool = True) -> Dict[str, int]:
        """
        Single-state action selection for all groups.
        Returns a dict {group_name -> action_index}.
        """
        if state.ndim == 1:
            state = state.unsqueeze(0)  # (1, state_dim)

        logits_by_group = self.forward(state)
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
