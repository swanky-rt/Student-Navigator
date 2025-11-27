# # GRPO/grpo_policy.py
# """
# Policy network for REAL GRPO.
# """

# from typing import Dict
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from utils.config import GROUPS, NUM_ACTIONS

# class RulePolicy(nn.Module):
#     def __init__(self, state_dim: int, num_actions: int = NUM_ACTIONS):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(state_dim, 64),
#             nn.Tanh(),
#             nn.Linear(64, 64),
#             nn.Tanh(),
#         )
#         self.group_heads = nn.ModuleDict()
#         self.value_heads = nn.ModuleDict()
#         for g in GROUPS.keys():
#             self.group_heads[g] = nn.Linear(64, num_actions)
#             self.value_heads[g] = nn.Linear(64, 1)

#     def forward(self, state):
#         if state.ndim == 1:
#             state = state.unsqueeze(0)
#         emb = self.encoder(state)
#         logits_by_group = {}
#         values_by_group = {}
#         for g in self.group_heads:
#             logits_by_group[g] = self.group_heads[g](emb)
#             values_by_group[g] = self.value_heads[g](emb).squeeze(-1)
#         return logits_by_group, values_by_group

#     @torch.no_grad()
#     def act(self, state, deterministic=True):
#         logits_by_group, _ = self.forward(state)
#         actions = {}
#         for g, logits in logits_by_group.items():
#             probs = F.softmax(logits, dim=-1)
#             if deterministic:
#                 a = int(probs.argmax(dim=-1).item())
#             else:
#                 dist = torch.distributions.Categorical(probs=probs)
#                 a = int(dist.sample().item())
#             actions[g] = a
#         return actions


# GRPO/grpo_policy.py

import torch
import torch.nn as nn
from utils.config import GROUPS

class RulePolicy(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        hidden = 128
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.heads = nn.ModuleDict({
            g: nn.Linear(hidden, 3) for g in GROUPS
        })

    def forward(self, state):
        h = self.net(state)
        return {g: self.heads[g](h) for g in self.heads}

    def act(self, state, deterministic=False):
        logits = self.forward(state)
        actions = {}

        for g, lg in logits.items():
            if deterministic:
                a = torch.argmax(lg).item()
            else:
                dist = torch.distributions.Categorical(logits=lg)
                a = dist.sample().item()
            actions[g] = a

        return actions