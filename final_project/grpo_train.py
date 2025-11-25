"""
Training utilities for the GRPO Rule Agent.
"""

from dataclasses import dataclass
from typing import List
import random

import pandas as pd
import torch
import torch.nn.functional as F

from grpo_config import SCENARIOS, GROUP2TYPEIDX
from grpo_mdp import fields_to_mask, build_state, apply_group_action, compute_group_reward
from grpo_policy import RulePolicy


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
    from grpo_config import PII_TYPES
    """
    Load the Excel dataset (sheet 'dataset') and produce masks for:
    - ground_truth       -> present_mask
    - allowed_restaurant -> allowed_mask_restaurant
    - bank/other set     -> allowed_mask_bank (we heuristically pick a non-empty extra column)
    """
    df = pd.read_excel(path, sheet_name="dataset")
    rows: List[DatasetRow] = []

    col_names = {c.lower(): c for c in df.columns}
    gt_col = col_names.get("ground_truth", "ground_truth")
    rest_col = col_names.get("allowed_restaurant", "allowed_restaurant")

    bank_col = None
    for c in df.columns:
        if c not in [gt_col, rest_col] and not df[c].isna().all():
            bank_col = c
            break
    if bank_col is None:
        bank_col = rest_col  # fallback

    for _, row in df.iterrows():
        gt = parse_list_str(row.get(gt_col, ""))

        # Start from what the dataset says
        allowed_rest = parse_list_str(row.get(rest_col, ""))
        allowed_bank = parse_list_str(row.get(bank_col, ""))

        # ---- Override policy goals for training ----
        # Restaurant: treat NAME as allowed (if present)
        if "NAME" not in allowed_rest:
            allowed_rest.append("NAME")

        # Bank: treat *all* PII types as potentially allowed
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


class Transition:
    """
    Simple container to hold transitions for training.
    Each element is (state_vec, group_name, action, log_prob, reward, scenario_name).
    """

    def __init__(self):
        self.states = []
        self.group_names = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.scenario_names = []

    def add(self, state, group_name, action, log_prob, reward, scenario_name):
        self.states.append(state)
        self.group_names.append(group_name)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.scenario_names.append(scenario_name)

    def __len__(self):
        return len(self.states)


def rollout_batch(policy: RulePolicy, dataset_rows: List[DatasetRow], batch_size: int = 64) -> Transition:
    """
    Offline rollout from dataset:
    - Sample a dataset row
    - Sample a scenario (restaurant or bank)
    - Build state
    - For each group with present fields, sample action and compute reward
    """
    trans = Transition()
    for _ in range(batch_size):
        row = random.choice(dataset_rows)
        scenario_id = random.choice(list(SCENARIOS.keys()))
        scenario_name = SCENARIOS[scenario_id]

        present_mask = row.present_mask
        allowed_mask = (
            row.allowed_mask_restaurant if scenario_name == "restaurant" else row.allowed_mask_bank
        )

        # Build state tensor
        state_tensor = build_state(present_mask, scenario_id)

        # Forward pass to get logits per group
        logits_by_group = policy(state_tensor.unsqueeze(0))

        for g, type_indices in GROUP2TYPEIDX.items():
            # Skip groups with no present fields
            if not any(present_mask[i] == 1 for i in type_indices):
                continue

            logits = logits_by_group[g]  # (1, num_actions)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            action_t = dist.sample()
            action = int(action_t.item())
            log_prob = float(dist.log_prob(action_t).item())

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
                state=state_tensor.tolist(),
                group_name=g,
                action=action,
                log_prob=log_prob,
                reward=reward,
                scenario_name=scenario_name,
            )
    return trans


def policy_gradient_update(policy: RulePolicy, optimizer, transitions: Transition, epochs: int = 3):
    """
    Simple REINFORCE-style policy gradient update with normalized rewards as advantages.
    """
    if len(transitions) == 0:
        return

    rewards = torch.tensor(transitions.rewards, dtype=torch.float32)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    for _ in range(epochs):
        optimizer.zero_grad()
        total_loss = 0.0
        for idx in range(len(transitions)):
            state_vec = transitions.states[idx]
            group_name = transitions.group_names[idx]
            action = transitions.actions[idx]

            state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
            logits_by_group = policy(state_tensor)
            logits = logits_by_group[group_name]
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            action_t = torch.tensor([action])
            log_prob = dist.log_prob(action_t)[0]

            advantage = advantages[idx]
            total_loss += -log_prob * advantage
        total_loss = total_loss / len(transitions)
        total_loss.backward()
        optimizer.step()


def evaluate_average_reward(policy: RulePolicy, dataset_rows: List[DatasetRow], num_samples: int = 200) -> float:
    """
    Roughly estimate the average reward under the current policy by sampling.
    """
    if not dataset_rows:
        return 0.0
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

        state_tensor = build_state(present_mask, scenario_id)
        logits_by_group = policy(state_tensor.unsqueeze(0))

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
