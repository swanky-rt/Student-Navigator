"""
REAL GRPO training for the RulePolicy:

- Per-group reward (identity/contact/financial/network)
- Per-group value heads
- Per-group advantage normalization (grouped)
- Per-group KL regularization
"""

from dataclasses import dataclass
from typing import List, Dict
import random

import pandas as pd
import torch
import torch.nn.functional as F

from utils.config import SCENARIOS, GROUP2TYPEIDX, PII_TYPES, GROUPS
from utils.mdp import fields_to_mask, build_state, apply_group_action, compute_group_reward
from GRPO.grpo_policy import RulePolicy


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
    Load Excel dataset (sheet 'dataset') and produce masks for:

      - ground_truth       -> present_mask
      - allowed_restaurant -> allowed_mask_restaurant
      - allowed_bank       -> allowed_mask_bank

    Column names are matched case-insensitively.
    """
    df = pd.read_excel(path, sheet_name="dataset")
    rows: List[DatasetRow] = []

    # Case-insensitive column lookup
    col_names = {c.lower(): c for c in df.columns}

    gt_col = col_names.get("ground_truth")
    rest_col = col_names.get("allowed_restaurant")
    
    bank_col = None
    for c in df.columns:
        if c not in [gt_col, rest_col] and not df[c].isna().all():
            bank_col = c
            break
    if bank_col is None:
        bank_col = rest_col  # fallback

    if gt_col is None:
        raise ValueError("Expected a 'ground_truth' column in 'dataset' sheet.")
    if rest_col is None:
        raise ValueError("Expected an 'allowed_restaurant' column in 'dataset' sheet.")

    for _, row in df.iterrows():
        gt = parse_list_str(row.get(gt_col, ""))

        # Start from dataset annotation
        allowed_rest = parse_list_str(row.get(rest_col, ""))

        # Bank allowed: either explicit column or default "all PII allowed"
        if bank_col is not None:
            allowed_bank = parse_list_str(row.get(bank_col, ""))
        else:
            allowed_bank = list(PII_TYPES)

        # Optionally enforce some training-side policies
        if "NAME" not in allowed_rest and "NAME" in gt:
            allowed_rest.append("NAME")

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
    Transition buffer for grouped RL.

    Each element is *one* (state, group) decision:

        - state_vec: List[float]
        - group_name: str
        - action: int
        - reward: float
        - old_log_prob: float        (log π_old(a|s,g))
        - old_value: float           (V_old(s,g))
        - old_probs: List[float]     (π_old(.|s,g) for KL)
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
# GRPO rollout: per-group transitions
# ---------------------------------------------------------------------------

def rollout_batch_grpo(
    policy: RulePolicy,
    dataset_rows: List[DatasetRow],
    batch_size: int = 64,
) -> Transition:
    """
    Perform one batch of GRPO rollouts.

    For each of batch_size iterations:
      - Sample a dataset row
      - Sample a scenario (restaurant/bank)
      - Build state
      - For each group with present PII:
          * sample action from π_old(.|s,g)
          * compute shared fields via apply_group_action
          * compute group reward
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

        # Build state vector
        state_tensor = build_state(present_mask, scenario_id).to(device)  # (state_dim,)

        # Forward through policy to get π_old and V_old
        logits_by_group, values_by_group = policy(state_tensor.unsqueeze(0))

        for g, type_indices in GROUP2TYPEIDX.items():
            # Skip groups with no present fields
            if not any(present_mask[i] == 1 for i in type_indices):
                continue

            logits = logits_by_group[g]       # (1, num_actions)
            value = values_by_group[g][0]     # scalar V_old(s,g)
            probs = F.softmax(logits, dim=-1) # (1, num_actions)

            dist = torch.distributions.Categorical(probs=probs)
            action_t = dist.sample()          # ()
            action = int(action_t.item())
            old_log_prob = float(dist.log_prob(action_t).item())
            old_probs = probs.squeeze(0).detach().cpu().tolist()

            # Convert group action into shared PII indices
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
# GRPO: grouped advantages
# ---------------------------------------------------------------------------

def _compute_grouped_advantages(trans: Transition):
    """
    Compute per-group advantages with group-wise normalization.

    A_g = R_g - V_old(s,g),
    then normalize A_g within each group separately.
    """
    if len(trans) == 0:
        return None, None

    rewards = torch.tensor(trans.rewards, dtype=torch.float32)
    values_old = torch.tensor(trans.old_values, dtype=torch.float32)
    group_names = trans.group_names

    returns = rewards.clone()  # 1-step MDP: return = immediate reward
    raw_adv = returns - values_old  # (N,)

    # Group-wise normalization: A_hat_g = (A_g - mean_g) / std_g
    adv_norm = torch.zeros_like(raw_adv)
    group_to_indices: Dict[str, List[int]] = {}
    for idx, g in enumerate(group_names):
        group_to_indices.setdefault(g, []).append(idx)

    for g, idxs in group_to_indices.items():
        idx_tensor = torch.tensor(idxs, dtype=torch.long)
        a_g = raw_adv[idx_tensor]
        mean_g = a_g.mean()
        std_g = a_g.std() + 1e-8
        adv_norm[idx_tensor] = (a_g - mean_g) / std_g

    return returns, adv_norm


# ---------------------------------------------------------------------------
# REAL GRPO update (no PPO ratio/clipping)
# ---------------------------------------------------------------------------

def grpo_update(
    policy: RulePolicy,
    optimizer,
    transitions: Transition,
    grpo_epochs: int = 4,
    value_coef: float = 0.5,
    kl_coef: float = 0.1,
):
    """
    REAL GRPO-style update:

    For each stored (state, group):

        - Advantage: A_g = R_g - V_old(s,g) (group-wise normalized)
        - Policy loss:   L_policy_g = -log π(a|s,g) * A_hat_g
        - Value loss:    L_value_g  = (V(s,g) - R_g)^2
        - KL penalty:    L_kl_g     = β * KL(π_old(.|s,g) || π(.|s,g))

    Total loss = sum_g (L_policy_g + value_coef * L_value_g + L_kl_g),
    averaged over all transitions.
    """
    if len(transitions) == 0:
        return

    device = next(policy.parameters()).device

    returns, advantages = _compute_grouped_advantages(transitions)
    returns = returns.to(device)
    advantages = advantages.to(device)

    states = torch.tensor(transitions.states, dtype=torch.float32).to(device)
    actions = torch.tensor(transitions.actions, dtype=torch.long).to(device)
    old_log_probs = torch.tensor(transitions.old_log_probs, dtype=torch.float32).to(device)
    old_values = torch.tensor(transitions.old_values, dtype=torch.float32).to(device)
    old_probs_all = [torch.tensor(p, dtype=torch.float32).to(device) for p in transitions.old_probs]
    group_names = transitions.group_names

    N = len(transitions)

    for _ in range(grpo_epochs):
        optimizer.zero_grad()

        policy_losses = []
        value_losses = []
        kl_losses = []

        for idx in range(N):
            s = states[idx].unsqueeze(0)     # (1, state_dim)
            a = actions[idx]
            ret = returns[idx]
            adv = advantages[idx]
            old_probs = old_probs_all[idx]   # (num_actions,)
            g = group_names[idx]

            logits_by_group, values_by_group = policy(s)
            logits = logits_by_group[g]      # (1, num_actions)
            value = values_by_group[g][0]    # scalar V(s,g)

            probs = torch.softmax(logits, dim=-1)  # (1, num_actions)
            log_probs = torch.log(probs + 1e-8)
            log_prob = log_probs[0, a]

            # GRPO policy loss: no PPO ratio, just - log π(a|s,g) * A_hat_g
            policy_loss = -log_prob * adv

            # Value loss
            value_loss = (value - ret).pow(2)

            # KL penalty between old and new distributions for this group
            new_probs = probs.squeeze(0)  # (num_actions,)
            kl_g = torch.sum(
                old_probs * (torch.log(old_probs + 1e-8) - torch.log(new_probs + 1e-8))
            )

            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            kl_losses.append(kl_g)

        policy_loss = torch.stack(policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        kl_loss = torch.stack(kl_losses).mean()

        loss = policy_loss + value_coef * value_loss + kl_coef * kl_loss
        loss.backward()
        optimizer.step()


# ---------------------------------------------------------------------------
# Backwards-compatible wrappers (names used by your demo)
# ---------------------------------------------------------------------------

def rollout_batch(policy: RulePolicy, dataset_rows: List[DatasetRow], batch_size: int = 64) -> Transition:
    """
    Backwards-compatible wrapper: uses GRPO-style rollout.
    """
    return rollout_batch_grpo(policy, dataset_rows, batch_size=batch_size)


def policy_gradient_update(
    policy: RulePolicy,
    optimizer,
    transitions: Transition,
    epochs: int = 3,
):
    """
    Backwards-compatible wrapper: now does REAL GRPO updates.
    """
    return grpo_update(policy, optimizer, transitions, grpo_epochs=epochs)


# ---------------------------------------------------------------------------
# Evaluation: average reward under greedy policy
# ---------------------------------------------------------------------------

def evaluate_average_reward(
    policy: RulePolicy,
    dataset_rows: List[DatasetRow],
    num_samples: int = 200,
) -> float:
    """
    Estimate average per-group reward under the current policy using greedy actions.
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
            probs = torch.softmax(logits, dim=-1)
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
