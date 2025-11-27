# grpo_train.py
"""
Training utilities for the GRPO Rule Agent (learned mask version).
"""

from dataclasses import dataclass
from typing import List
import random

import torch
import torch.nn.functional as F

from Minimiser.utils.dataset import load_dataset_from_excel, DatasetRow
from Minimiser.utils.state import build_state
from Minimiser.utils.reward import compute_episode_reward
from Minimiser.utils.config import SCENARIOS
from Minimiser.grpo_policy import RulePolicy


@dataclass
class TransitionBatch:
    states: List[torch.Tensor]
    share_masks: List[List[int]]
    log_probs: List[float]
    rewards: List[float]

    def __len__(self):
        return len(self.states)


def rollout_batch(
    policy: RulePolicy,
    dataset_rows: List[DatasetRow],
    batch_size: int = 64,
) -> TransitionBatch:
    """
    Sample a batch of one-step episodes from the dataset and current policy.
    """
    batch = TransitionBatch(states=[], share_masks=[], log_probs=[], rewards=[])

    for _ in range(batch_size):
        row = random.choice(dataset_rows)

        # pick scenario uniformly
        scenario_name = random.choice(SCENARIOS)
        if scenario_name == "restaurant":
            allowed_mask = row.allowed_mask_restaurant
        else:
            allowed_mask = row.allowed_mask_bank

        present_mask = row.present_mask

        # build state
        state = build_state(present_mask, scenario_name)

        # policy decides which fields to share
        share_mask, log_prob = policy.act(state, deterministic=False)

        # environment computes reward using hidden allowed_mask
        reward = compute_episode_reward(
            present_mask=present_mask,
            allowed_mask=allowed_mask,
            share_mask=share_mask,
            scenario_name=scenario_name,
        )

        batch.states.append(state)
        batch.share_masks.append(share_mask)
        batch.log_probs.append(log_prob)
        batch.rewards.append(reward)

    return batch


def policy_gradient_update(
    policy: RulePolicy,
    optimizer: torch.optim.Optimizer,
    batch: TransitionBatch,
    epochs: int = 1,
    clip_grad_norm: float = 1.0,
):
    """
    Simple REINFORCE / GRPO-style policy-gradient update with reward normalization.
    """
    if len(batch) == 0:
        return

    states = torch.stack(batch.states, dim=0)  # (B, state_dim)
    actions = torch.tensor(batch.share_masks, dtype=torch.float32)  # (B, NUM_PII)
    old_log_probs = torch.tensor(batch.log_probs, dtype=torch.float32)  # (B,)
    rewards = torch.tensor(batch.rewards, dtype=torch.float32)  # (B,)

    # Advantage = normalized reward
    advantages = rewards - rewards.mean()
    if rewards.std() > 1e-8:
        advantages = advantages / rewards.std()

    for _ in range(epochs):
        logits = policy(states)  # (B, NUM_PII)
        probs = torch.sigmoid(logits)
        dist = torch.distributions.Bernoulli(probs=probs)

        # log_prob of the *same* sampled actions
        log_probs_new = dist.log_prob(actions).sum(dim=-1)  # (B,)

        # REINFORCE / GRPO-style loss: - E[adv * log_prob]
        loss = -(advantages * log_probs_new).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), clip_grad_norm)
        optimizer.step()


def evaluate_average_reward(
    policy: RulePolicy,
    dataset_rows: List[DatasetRow],
    num_samples: int = 200,
) -> float:
    """
    Estimate the average episode reward under the current policy.
    """
    total_reward = 0.0
    for _ in range(num_samples):
        row = random.choice(dataset_rows)
        scenario_name = random.choice(SCENARIOS)
        if scenario_name == "restaurant":
            allowed_mask = row.allowed_mask_restaurant
        else:
            allowed_mask = row.allowed_mask_bank

        present_mask = row.present_mask
        state = build_state(present_mask, scenario_name)

        share_mask, _ = policy.act(state, deterministic=False)

        r = compute_episode_reward(
            present_mask=present_mask,
            allowed_mask=allowed_mask,
            share_mask=share_mask,
            scenario_name=scenario_name,
        )
        total_reward += r

    return float(total_reward / num_samples)
