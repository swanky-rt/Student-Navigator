# # GRPO/grpo_train.py
# """
# REAL GRPO training for RulePolicy.
# """

# from dataclasses import dataclass
# from typing import List, Dict
# import random

# import pandas as pd
# import torch
# import torch.nn.functional as F

# from utils.config import SCENARIOS, GROUP2TYPEIDX, PII_TYPES, GROUPS
# from utils.mdp import fields_to_mask, build_state, apply_group_action, compute_group_reward
# from GRPO.grpo_policy import RulePolicy


# @dataclass
# class DatasetRow:
#     present_mask: List[int]
#     allowed_mask_restaurant: List[int]
#     allowed_mask_bank: List[int]


# def parse_list_str(s):
#     """
#     Parse strings like:
#         [EMAIL, PHONE, DATE/DOB]
#     """
#     if not isinstance(s, str):
#         return []

#     s = s.strip()
#     if s == "" or s == "[]" or s.lower() == "nan":
#         return []

#     # remove leading/trailing brackets
#     if s.startswith("[") and s.endswith("]"):
#         s = s[1:-1]

#     # split by comma
#     parts = [p.strip() for p in s.split(",")]

#     # filter empty
#     return [p for p in parts if p]



# def load_dataset_from_excel(path: str) -> List[DatasetRow]:
#     """
#     Load dataset from CSV or Excel automatically.
#     Requires columns:
#         - ground_truth
#         - allowed_restaurant
#         - allowed_bank
#     """

#     lower = path.lower()

#     # Auto-detect file type
#     if lower.endswith(".csv"):
#         df = pd.read_csv(path)
#     elif lower.endswith(".xlsx") or lower.endswith(".xls"):
#         df = pd.read_excel(path, sheet_name="dataset")
#     else:
#         raise ValueError(f"Unsupported file type: {path}")

#     rows: List[DatasetRow] = []

#     # Normalize column names
#     col_names = {c.lower(): c for c in df.columns}

#     required = ["ground_truth", "allowed_restaurant", "allowed_bank"]
#     missing = [col for col in required if col not in col_names]
#     if missing:
#         raise ValueError(f"Missing required columns: {missing}")

#     gt_col = col_names["ground_truth"]
#     rest_col = col_names["allowed_restaurant"]
#     bank_col = col_names["allowed_bank"]

#     for _, row in df.iterrows():
#         gt = parse_list_str(row.get(gt_col, ""))
#         allowed_rest = parse_list_str(row.get(rest_col, ""))
#         allowed_bank = parse_list_str(row.get(bank_col, ""))

#         present_mask = fields_to_mask(gt)
#         allowed_mask_rest = fields_to_mask(allowed_rest)
#         allowed_mask_bank = fields_to_mask(allowed_bank)

#         rows.append(
#             DatasetRow(
#                 present_mask=present_mask,
#                 allowed_mask_restaurant=allowed_mask_rest,
#                 allowed_mask_bank=allowed_mask_bank,
#             )
#         )

#     return rows


# class Transition:
#     def __init__(self):
#         self.states = []
#         self.group_names = []
#         self.actions = []
#         self.rewards = []
#         self.old_log_probs = []
#         self.old_values = []
#         self.old_probs = []
#         self.scenario_names = []

#     def add(self, state, group_name, action, reward, old_log_prob, old_value, old_probs, scenario_name):
#         self.states.append(state)
#         self.group_names.append(group_name)
#         self.actions.append(action)
#         self.rewards.append(reward)
#         self.old_log_probs.append(old_log_prob)
#         self.old_values.append(old_value)
#         self.old_probs.append(old_probs)
#         self.scenario_names.append(scenario_name)

#     def __len__(self):
#         return len(self.states)


# def rollout_batch_grpo(policy, dataset_rows, batch_size=64):
#     trans = Transition()
#     device = next(policy.parameters()).device

#     for _ in range(batch_size):
#         row = random.choice(dataset_rows)
#         scenario_id = random.choice(list(SCENARIOS.keys()))
#         scenario_name = SCENARIOS[scenario_id]

#         present_mask = row.present_mask
#         allowed_mask = (
#             row.allowed_mask_restaurant if scenario_name == "restaurant"
#             else row.allowed_mask_bank
#         )

#         state_tensor = build_state(present_mask, scenario_id).to(device)
#         logits_by_group, values_by_group = policy(state_tensor.unsqueeze(0))

#         for g, type_indices in GROUP2TYPEIDX.items():
#             if not any(present_mask[i] == 1 for i in type_indices):
#                 continue

#             logits = logits_by_group[g]
#             value = values_by_group[g][0]
#             probs = F.softmax(logits, dim=-1)

#             dist = torch.distributions.Categorical(probs=probs)
#             action_t = dist.sample()
#             action = int(action_t.item())

#             old_log_prob = float(dist.log_prob(action_t).item())
#             old_probs = probs.squeeze(0).detach().cpu().tolist()

#             # FIXED — no allowed_mask leak
#             shared_indices = apply_group_action(type_indices, present_mask, action)

#             reward = compute_group_reward(
#                 g, scenario_name, type_indices, present_mask,
#                 allowed_mask, shared_indices, action
#             )

#             trans.add(
#                 state_tensor.detach().cpu().tolist(),
#                 g, action, float(reward), old_log_prob,
#                 float(value.detach().cpu().item()), old_probs, scenario_name
#             )

#     return trans


# def rollout_batch(policy, dataset_rows, batch_size=64):
#     return rollout_batch_grpo(policy, dataset_rows, batch_size)


# def _compute_grouped_advantages(trans):
#     rewards = torch.tensor(trans.rewards)
#     values_old = torch.tensor(trans.old_values)

#     group_names = trans.group_names
#     returns = rewards.clone()
#     raw_adv = returns - values_old

#     adv_norm = torch.zeros_like(raw_adv)
#     group_to_indices = {}

#     for idx, g in enumerate(group_names):
#         group_to_indices.setdefault(g, []).append(idx)

#     for g, idxs in group_to_indices.items():
#         idxs_t = torch.tensor(idxs)
#         a = raw_adv[idxs_t]
#         adv_norm[idxs_t] = (a - a.mean()) / (a.std() + 1e-8)

#     return returns, adv_norm


# def grpo_update(policy, optimizer, transitions, grpo_epochs=4, value_coef=0.5, kl_coef=0.1):
#     device = next(policy.parameters()).device

#     returns, advantages = _compute_grouped_advantages(transitions)
#     returns, advantages = returns.to(device), advantages.to(device)

#     states = torch.tensor(transitions.states).float().to(device)
#     actions = torch.tensor(transitions.actions).to(device)
#     old_values = torch.tensor(transitions.old_values).float().to(device)
#     old_probs_all = [torch.tensor(p).to(device) for p in transitions.old_probs]
#     group_names = transitions.group_names

#     N = len(transitions)

#     for _ in range(grpo_epochs):
#         optimizer.zero_grad()

#         policy_losses, value_losses, kl_losses = [], [], []

#         for idx in range(N):
#             s = states[idx].unsqueeze(0)
#             a = actions[idx]
#             ret = returns[idx]
#             adv = advantages[idx]
#             old_probs = old_probs_all[idx]
#             g = group_names[idx]

#             logits_by_group, values_by_group = policy(s)
#             logits = logits_by_group[g]
#             value = values_by_group[g][0]

#             probs = torch.softmax(logits, dim=-1)
#             log_probs = torch.log(probs + 1e-8)
#             log_prob = log_probs[0, a]

#             policy_loss = -log_prob * adv
#             value_loss = (value - ret)**2

#             new_probs = probs.squeeze(0)
#             kl_g = torch.sum(old_probs * (torch.log(old_probs+1e-8) - torch.log(new_probs+1e-8)))

#             policy_losses.append(policy_loss)
#             value_losses.append(value_loss)
#             kl_losses.append(kl_g)

#         loss = (torch.stack(policy_losses).mean() +
#                 value_coef * torch.stack(value_losses).mean() +
#                 kl_coef * torch.stack(kl_losses).mean())

#         loss.backward()
#         optimizer.step()


# def policy_gradient_update(policy, optimizer, transitions, epochs=3):
#     return grpo_update(policy, optimizer, transitions, grpo_epochs=epochs)


# def evaluate_average_reward(policy, dataset_rows, num_samples=200):
#     device = next(policy.parameters()).device

#     total_reward, count = 0, 0
#     for _ in range(num_samples):
#         row = random.choice(dataset_rows)
#         scenario_id = random.choice(list(SCENARIOS.keys()))
#         scenario_name = SCENARIOS[scenario_id]

#         present_mask = row.present_mask
#         allowed_mask = (
#             row.allowed_mask_restaurant if scenario_name=="restaurant"
#             else row.allowed_mask_bank
#         )

#         state_tensor = build_state(present_mask, scenario_id).to(device)
#         logits_by_group, _ = policy(state_tensor.unsqueeze(0))

#         for g, type_indices in GROUP2TYPEIDX.items():
#             if not any(present_mask[i]==1 for i in type_indices):
#                 continue

#             probs = torch.softmax(logits_by_group[g], dim=-1)
#             action = int(probs.argmax(dim=-1).item())

#             shared_indices = apply_group_action(type_indices, present_mask, action)
#             r = compute_group_reward(
#                 g, scenario_name, type_indices, present_mask,
#                 allowed_mask, shared_indices, action
#             )
#             total_reward += r
#             count += 1

#     return total_reward / count if count else 0.0


# GRPO/grpo_train.py

import pandas as pd
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List
from utils.mdp import fields_to_mask, compute_group_reward
from utils.config import PII_TYPES, GROUP2TYPEIDX, SCENARIO_NAME2ID
import random

@dataclass
class DatasetRow:
    present_mask: List[int]
    allowed_mask_restaurant: List[int]
    allowed_mask_bank: List[int]

def parse_list_str(s: str) -> List[str]:
    if not isinstance(s, str) or len(s.strip()) == 0:
        return []
    s = s.strip().lstrip("[").rstrip("]")
    items = [x.strip() for x in s.split(",")]
    return [x for x in items if x]

def load_dataset_from_excel(path: str) -> List[DatasetRow]:
    lower = path.lower()
    if lower.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, sheet_name="dataset")

    rows = []

    for _, row in df.iterrows():
        gt = parse_list_str(row["ground_truth"])
        rest = parse_list_str(row["allowed_restaurant"])
        bank = parse_list_str(row["allowed_bank"])

        rows.append(
            DatasetRow(
                present_mask=fields_to_mask(gt),
                allowed_mask_restaurant=fields_to_mask(rest),
                allowed_mask_bank=fields_to_mask(bank),
            )
        )

    return rows

def rollout_batch(policy, rows, batch_size=32):
    batch = random.sample(rows, batch_size)
    transitions = []

    for r in batch:
        for scenario in ["restaurant", "bank"]:
            present = r.present_mask
            allowed = (
                r.allowed_mask_restaurant if scenario == "restaurant"
                else r.allowed_mask_bank
            )
            scenario_id = SCENARIO_NAME2ID[scenario]

            import utils.mdp as mdp
            state = mdp.build_state(present, scenario_id)
            actions = policy.act(state, deterministic=False)

            shared = {}
            for g, idxs in GROUP2TYPEIDX.items():
                shared[g] = mdp.apply_group_action(idxs, present, actions[g])

            reward = 0.0
            for g, idxs in GROUP2TYPEIDX.items():
                reward += mdp.compute_group_reward(
                    g, scenario, idxs,
                    present, allowed,
                    shared[g], actions[g]
                )

            transitions.append((state, actions, reward))

    return transitions

def policy_gradient_update(policy, optimizer, transitions, epochs=1):
    for _ in range(epochs):
        for state, actions, reward in transitions:
            logits = policy.forward(state)

            logp = 0
            for g, a in actions.items():
                logp += F.log_softmax(logits[g], dim=-1)[a]

            loss = -logp * reward

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def evaluate_average_reward(policy, rows, num_samples=200):
    total = 0.0
    for _ in range(num_samples):
        r = random.choice(rows)
        scenario = random.choice(["restaurant", "bank"])
        present = r.present_mask
        allowed = (
            r.allowed_mask_restaurant if scenario == "restaurant"
            else r.allowed_mask_bank
        )
        scenario_id = SCENARIO_NAME2ID[scenario]

        import utils.mdp as mdp
        state = mdp.build_state(present, scenario_id)
        actions = policy.act(state, deterministic=True)

        reward = 0.0
        for g, idxs in GROUP2TYPEIDX.items():
            shared = mdp.apply_group_action(idxs, present, actions[g])
            reward += mdp.compute_group_reward(
                g, scenario, idxs, present, allowed, shared, actions[g]
            )

        total += reward

    return total / num_samples