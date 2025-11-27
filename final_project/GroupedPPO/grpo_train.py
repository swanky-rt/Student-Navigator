"""
Training utilities for the GRPO Rule Agent.

This version implements a grouped PPO / GRPO-style update:
- per-group rewards (identity/contact/financial/network)
- per-group value heads
- clipped policy objective
- entropy bonus
- optional KL regularization
"""

from dataclasses import dataclass
from typing import List, Dict
import random

import pandas as pd
import torch
import torch.nn.functional as F

from utils.config import SCENARIOS, GROUP2TYPEIDX
from utils.mdp import fields_to_mask, build_state, apply_group_action, compute_group_reward
from GroupedPPO.grpo_policy import RulePolicy


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

@dataclass
class DatasetRow:
    present_mask: List[int]
    allowed_mask_restaurant: List[int]
    allowed_mask_bank: List[int]


def parse_list_str(s) -> List[str]:
    if not isinstance(s, str):
        return []
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts


def load_dataset_from_excel(path: str) -> List[DatasetRow]:
    """
    Load the Excel dataset (sheet 'dataset') and produce masks for:
    - ground_truth       -> present_mask
    - allowed_restaurant -> allowed_mask_restaurant
    - allowed_bank       -> allowed_mask_bank

    We do a case-insensitive lookup of column names. For "bank", we fall back
    to "allowed_bank" if present, else we treat all PII types as allowed.
    """
    from utils.config import PII_TYPES

    df = pd.read_excel(path, sheet_name="dataset")
    rows: List[DatasetRow] = []

    # Case-insensitive column lookup
    col_names = {c.lower(): c for c in df.columns}

    gt_col = col_names.get("ground_truth")
    rest_col = col_names.get("allowed_restaurant")
    bank_col = col_names.get("allowed_bank")

    if gt_col is None:
        raise ValueError("Expected a 'ground_truth' column in the dataset sheet.")
    if rest_col is None:
        raise ValueError("Expected an 'allowed_restaurant' column in the dataset sheet.")

    for _, row in df.iterrows():
        gt = parse_list_str(row.get(gt_col, ""))

        # Start from what the dataset says
        allowed_rest = parse_list_str(row.get(rest_col, ""))

        # Allow explicit bank column if present, else default to 'all PII allowed'
        if bank_col is not None:
            allowed_bank = parse_list_str(row.get(bank_col, ""))
        else:
            allowed_bank = list(PII_TYPES)

        # ---- Override policy goals for training if needed ----
        # Example: ensure NAME is allowed in restaurant scenario
        if "NAME" not in allowed_rest and "NAME" in gt:
            allowed_rest.append("NAME")

        # Bank: treat *all* PII types as potentially allowed if no explicit col
        if bank_col is None:
            allowed_bank = list(PII_TYPES)

        present_mask = fields_to_mask(gt)
        allowed_mask_rest = fields_to_mask(allowed_rest)
        allowed_mask_bank = fields_to_mask(allowed_bank)

        rows.append(
            DatasetRow(
                present_mask=present_mask,
                allowed_mask_restaurant=allowed_mask_rest,
                allowed_mask_bank=allowed_mask_bank,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Transition buffer for GRPO
# ---------------------------------------------------------------------------

class Transition:
    """
    Container to hold transitions for training.

    Each element corresponds to one (state, group) decision:
        - state_vec: List[float]
        - group_name: str
        - action: int
        - reward: float
        - old_log_prob: float
        - old_value: float      (V_old(s, g))
        - old_probs: List[float] (full old action distribution for KL)
        - scenario_name: str
    """

    def __init__(self):
        self.states: List[List[float]] = []
        self.group_names: List[str] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.old_log_probs: List[float] = []
        self.old_values: List[float] = []
        self.old_probs: List[List[float]] = []
        self.scenario_names: List[str] = []

    def add(
        self,
        state: List[float],
        group_name: str,
        action: int,
        reward: float,
        old_log_prob: float,
        old_value: float,
        old_probs: List[float],
        scenario_name: str,
    ):
        self.states.append(state)
        self.group_names.append(group_name)
        self.actions.append(action)
        self.rewards.append(reward)
        self.old_log_probs.append(old_log_prob)
        self.old_values.append(old_value)
        self.old_probs.append(old_probs)
        self.scenario_names.append(scenario_name)

    def __len__(self) -> int:
        return len(self.states)


# ---------------------------------------------------------------------------
# GRPO rollout: collect per-group transitions
# ---------------------------------------------------------------------------

def rollout_batch_grpo(
    policy: RulePolicy,
    dataset_rows: List[DatasetRow],
    batch_size: int = 64,
) -> Transition:
    """
    Perform one batch of rollouts:

    - Sample a dataset row
    - Sample a scenario (restaurant or bank)
    - Build state
    - For each PII group with present fields:
        * sample action from current policy
        * compute per-group reward
        * store old log-prob, old value, old probs
    """
    trans = Transition()
    if not dataset_rows:
        return trans

    device = next(policy.parameters()).device

    for _ in range(batch_size):
        row = random.choice(dataset_rows)
        scenario_id = random.choice(list(SCENARIOS.keys()))
        scenario_name = SCENARIOS[scenario_id]

        present_mask = row.present_mask
        allowed_mask = (
            row.allowed_mask_restaurant if scenario_name == "restaurant" else row.allowed_mask_bank
        )

        # Build state tensor
        state_tensor = build_state(present_mask, scenario_id).to(device)  # (state_dim,)

        # Forward pass to get logits and values per group
        logits_by_group, values_by_group = policy(state_tensor.unsqueeze(0))

        for g, type_indices in GROUP2TYPEIDX.items():
            # Skip groups with no present fields
            if not any(present_mask[i] == 1 for i in type_indices):
                continue

            logits = logits_by_group[g]         # (1, num_actions)
            value = values_by_group[g][0]       # scalar
            probs = F.softmax(logits, dim=-1)   # (1, num_actions)

            dist = torch.distributions.Categorical(probs=probs)
            action_t = dist.sample()            # ()
            action = int(action_t.item())
            old_log_prob = float(dist.log_prob(action_t).item())
            old_probs = probs.squeeze(0).detach().cpu().tolist()

            # Apply action to compute which PII fields are shared
            shared_indices = apply_group_action(type_indices, present_mask, allowed_mask, action)
            reward = compute_group_reward(
                group_name=g,
                scenario_name=scenario_name,
                group_type_indices=type_indices,
                present_mask=present_mask,
                allowed_mask=allowed_mask,
                shared_indices=shared_indices,
                action=action,
            )

            trans.add(
                state=state_tensor.detach().cpu().tolist(),
                group_name=g,
                action=action,
                reward=float(reward),
                old_log_prob=old_log_prob,
                old_value=float(value.detach().cpu().item()),
                old_probs=old_probs,
                scenario_name=scenario_name,
            )
    return trans


# ---------------------------------------------------------------------------
# Advantage computation
# ---------------------------------------------------------------------------

def _compute_returns_and_advantages(trans: Transition):
    """
    For this 1-step MDP, the return is equal to the immediate reward.
    Advantage uses a simple baseline: A = R - V_old.
    """
    if len(trans) == 0:
        return None, None

    rewards = torch.tensor(trans.rewards, dtype=torch.float32)
    values_old = torch.tensor(trans.old_values, dtype=torch.float32)
    returns = rewards  # 1-step
    advantages = returns - values_old

    # Optional normalization for stability
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return returns, advantages


# ---------------------------------------------------------------------------
# GRPO / PPO-style update
# ---------------------------------------------------------------------------

def grpo_update(
    policy: RulePolicy,
    optimizer,
    transitions: Transition,
    ppo_epochs: int = 4,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    kl_coef: float = 0.1,
):
    """
    Grouped PPO / GRPO update using per-group rewards.

    For each stored (state, group):
        - compute ratio = π_new / π_old
        - clipped surrogate objective
        - value loss
        - entropy bonus
        - KL penalty between old and new group distributions
    """
    if len(transitions) == 0:
        return

    device = next(policy.parameters()).device

    returns, advantages = _compute_returns_and_advantages(transitions)
    returns = returns.to(device)
    advantages = advantages.to(device)

    states = torch.tensor(transitions.states, dtype=torch.float32).to(device)
    actions = torch.tensor(transitions.actions, dtype=torch.long).to(device)
    old_log_probs = torch.tensor(transitions.old_log_probs, dtype=torch.float32).to(device)
    old_values = torch.tensor(transitions.old_values, dtype=torch.float32).to(device)
    old_probs_all = [torch.tensor(p, dtype=torch.float32).to(device) for p in transitions.old_probs]
    group_names = transitions.group_names

    N = len(transitions)

    for _ in range(ppo_epochs):
        optimizer.zero_grad()

        policy_losses = []
        value_losses = []
        entropies = []
        kls = []

        for idx in range(N):
            s = states[idx].unsqueeze(0)      # (1, state_dim)
            a = actions[idx]
            adv = advantages[idx]
            ret = returns[idx]
            old_lp = old_log_probs[idx]
            old_probs = old_probs_all[idx]    # (num_actions,)
            g = group_names[idx]

            logits_by_group, values_by_group = policy(s)
            logits = logits_by_group[g]       # (1, num_actions)
            value = values_by_group[g][0]     # scalar

            probs = F.softmax(logits, dim=-1)  # (1, num_actions)
            dist = torch.distributions.Categorical(probs=probs)
            log_prob = dist.log_prob(a)
            entropy = dist.entropy()

            # PPO ratio
            ratio = torch.exp(log_prob - old_lp)

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2)

            # Value loss
            value_loss = F.mse_loss(value, ret)

            # KL between old and new distributions (per group)
            new_probs = probs.squeeze(0)  # (num_actions,)
            kl_term = torch.sum(
                old_probs * (torch.log(old_probs + 1e-8) - torch.log(new_probs + 1e-8))
            )

            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            entropies.append(entropy)
            kls.append(kl_term)

        policy_loss = torch.stack(policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        entropy = torch.stack(entropies).mean()
        kl = torch.stack(kls).mean()

        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy + kl_coef * kl
        loss.backward()
        optimizer.step()


# ---------------------------------------------------------------------------
# Backwards-compatible wrappers
# ---------------------------------------------------------------------------

def rollout_batch(policy: RulePolicy, dataset_rows: List[DatasetRow], batch_size: int = 64) -> Transition:
    """
    Backwards-compatible name: now uses GRPO-style rollout.
    """
    return rollout_batch_grpo(policy, dataset_rows, batch_size=batch_size)


def policy_gradient_update(
    policy: RulePolicy,
    optimizer,
    transitions: Transition,
    epochs: int = 3,
):
    """
    Backwards-compatible name: now wraps the GRPO / PPO-style update.

    `epochs` is mapped to `ppo_epochs`.
    """
    return grpo_update(policy, optimizer, transitions, ppo_epochs=epochs)


# ---------------------------------------------------------------------------
# Evaluation: average reward under greedy policy
# ---------------------------------------------------------------------------

def evaluate_average_reward(
    policy: RulePolicy,
    dataset_rows: List[DatasetRow],
    num_samples: int = 200,
) -> float:
    """
    Roughly estimate the average reward under the current policy by sampling.
    """
    if not dataset_rows:
        return 0.0

    device = next(policy.parameters()).device

    total_reward = 0.0
    count = 0

    for _ in range(num_samples):
        row = random.choice(dataset_rows)
        scenario_id = random.choice(list(SCENARIOS.keys()))
        scenario_name = SCENARIOS[scenario_id]

        present_mask = row.present_mask
        allowed_mask = (
            row.allowed_mask_restaurant if scenario_name == "restaurant" else row.allowed_mask_bank
        )

        state_tensor = build_state(present_mask, scenario_id).to(device)
        logits_by_group, _ = policy(state_tensor.unsqueeze(0))

        for g, type_indices in GROUP2TYPEIDX.items():
            if not any(present_mask[i] == 1 for i in type_indices):
                continue

            logits = logits_by_group[g]
            probs = F.softmax(logits, dim=-1)
            action = int(probs.argmax(dim=-1).item())

            shared_indices = apply_group_action(type_indices, present_mask, allowed_mask, action)
            r = compute_group_reward(
                group_name=g,
                scenario_name=scenario_name,
                group_type_indices=type_indices,
                present_mask=present_mask,
                allowed_mask=allowed_mask,
                shared_indices=shared_indices,
                action=action,
            )
            total_reward += r
            count += 1

    if count == 0:
        return 0.0
    return float(total_reward / count)