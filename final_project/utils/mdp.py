# # utils/mdp.py
# """
# MDP helpers for GRPO Rule Agent.
# """

# from dataclasses import dataclass
# from typing import Dict, List
# import random
# import torch

# from utils.config import (
#     PII_TYPES,
#     TYPE2IDX,
#     GROUPS,
#     GROUP2TYPEIDX,
#     SCENARIO_NAME2ID,
#     NUM_PII,
#     NUM_SCENARIOS,
#     GROUP_WEIGHTS,
#     SCENARIO_WEIGHTS,
#     LAMBDA_COMPLEXITY,
#     ACTION_COMPLEXITY,
# )

# @dataclass
# class ManualInput:
#     present_fields: List[str]
#     scenario_name: str
#     allowed_fields_restaurant: List[str]
#     allowed_fields_bank: List[str]

# @dataclass
# class DecisionOutput:
#     scenario_name: str
#     actions_by_group: Dict[str, int]
#     shared_fields_by_group: Dict[str, List[str]]
#     present_fields: List[str]

# def fields_to_mask(fields: List[str]) -> List[int]:
#     mask = [0] * NUM_PII
#     for f in fields:
#         if f in TYPE2IDX:
#             mask[TYPE2IDX[f]] = 1
#     return mask

# def mask_to_fields(mask: List[int]) -> List[str]:
#     return [PII_TYPES[i] for i, v in enumerate(mask) if v == 1]

# def build_state(present_mask: List[int], scenario_id: int) -> torch.Tensor:
#     scenario_oh = [0] * NUM_SCENARIOS
#     scenario_oh[scenario_id] = 1
#     state_vec = present_mask + scenario_oh
#     return torch.tensor(state_vec, dtype=torch.float32)

# # ============ ONLY CHANGED FUNCTION (NO allowed-mask leak) ============
# def apply_group_action(group_type_indices, present_mask, action):
#     present_indices = [i for i in group_type_indices if present_mask[i] == 1]

#     if not present_indices:
#         return []

#     if action == 0:
#         return []

#     if action == 1:
#         return present_indices

#     if action == 2:
#         if len(present_indices) <= 1:
#             return present_indices
#         k = max(1, len(present_indices)//2)
#         return random.sample(present_indices, k)

#     raise ValueError(f"Invalid action {action}")
# # ======================================================================

# def compute_group_reward(
#     group_name, scenario_name, group_type_indices,
#     present_mask, allowed_mask, shared_indices, action
# ):
#     present = [i for i in group_type_indices if present_mask[i] == 1]
#     if not present:
#         return 0.0

#     allowed = [i for i in present if allowed_mask[i] == 1]
#     disallowed = [i for i in present if allowed_mask[i] == 0]

#     shared_set = set(shared_indices)
#     shared_allowed = [i for i in allowed if i in shared_set]
#     shared_disallowed = [i for i in disallowed if i in shared_set]

#     utility = len(shared_allowed)/len(allowed) if allowed else 0.0
#     privacy = 1.0 - (len(shared_disallowed)/len(disallowed)) if disallowed else 1.0

#     beta = SCENARIO_WEIGHTS[scenario_name]["beta"]
#     alpha = SCENARIO_WEIGHTS[scenario_name]["alpha"]
#     w_g = GROUP_WEIGHTS[group_name]
#     complexity = ACTION_COMPLEXITY[action]

#     return float(w_g * (beta*privacy + alpha*utility - LAMBDA_COMPLEXITY*complexity))

# def decide_sharing_for_manual_input(policy, manual, deterministic=True):
#     present_mask = fields_to_mask(manual.present_fields)
#     scenario_id = SCENARIO_NAME2ID[manual.scenario_name]

#     allowed_mask = fields_to_mask(
#         manual.allowed_fields_restaurant
#         if manual.scenario_name == "restaurant"
#         else manual.allowed_fields_bank
#     )

#     state = build_state(present_mask, scenario_id)
#     actions_by_group = policy.act(state, deterministic=deterministic)

#     shared_fields_by_group = {}
#     for g, type_indices in GROUP2TYPEIDX.items():
#         action = actions_by_group[g]
#         shared_indices = apply_group_action(type_indices, present_mask, action)
#         shared_fields_by_group[g] = [PII_TYPES[i] for i in shared_indices]

#     return DecisionOutput(
#         scenario_name=manual.scenario_name,
#         actions_by_group=actions_by_group,
#         shared_fields_by_group=shared_fields_by_group,
#         present_fields=manual.present_fields,
#     )



# utils/mdp.py

from dataclasses import dataclass
from typing import Dict, List
import random
import torch

from utils.config import (
    PII_TYPES,
    TYPE2IDX,
    GROUP2TYPEIDX,
    SCENARIO_NAME2ID,
    NUM_PII,
    NUM_SCENARIOS,
    GROUP_WEIGHTS,
    SCENARIO_WEIGHTS,
    LAMBDA_COMPLEXITY,
    ACTION_COMPLEXITY,
)

@dataclass
class ManualInput:
    present_fields: List[str]
    scenario_name: str

@dataclass
class DecisionOutput:
    scenario_name: str
    actions_by_group: Dict[str, int]
    shared_fields_by_group: Dict[str, List[str]]
    present_fields: List[str]

def fields_to_mask(fields: List[str]) -> List[int]:
    mask = [0] * NUM_PII
    for f in fields:
        if f in TYPE2IDX:
            mask[TYPE2IDX[f]] = 1
    return mask

def build_state(present_mask: List[int], scenario_id: int) -> torch.Tensor:
    scenario_vec = [0] * NUM_SCENARIOS
    scenario_vec[scenario_id] = 1
    return torch.tensor(present_mask + scenario_vec, dtype=torch.float32)

def apply_group_action(type_indices, present_mask, action):
    present = [i for i in type_indices if present_mask[i] == 1]

    if not present:
        return []

    if action == 0:
        return []

    if action == 1:
        return present

    if action == 2:
        if len(present) <= 1:
            return present
        k = max(1, len(present) // 2)
        return random.sample(present, k)

    return []

def compute_group_reward(
    group_name, scenario_name, type_indices,
    present_mask, allowed_mask, shared_indices, action
):
    present = [i for i in type_indices if present_mask[i] == 1]
    if not present:
        return 0.0

    allowed = [i for i in present if allowed_mask[i] == 1]
    disallowed = [i for i in present if allowed_mask[i] == 0]

    shared = set(shared_indices)

    shared_allowed = [i for i in allowed if i in shared]
    shared_disallowed = [i for i in disallowed if i in shared]

    utility = len(shared_allowed) / len(allowed) if allowed else 0.0
    privacy = 1 - (len(shared_disallowed) / len(disallowed)) if disallowed else 1.0

    alpha = SCENARIO_WEIGHTS[scenario_name]["alpha"]
    beta = SCENARIO_WEIGHTS[scenario_name]["beta"]
    w_g = GROUP_WEIGHTS[group_name]
    comp = ACTION_COMPLEXITY[action]

    return float(w_g * (beta * privacy + alpha * utility - LAMBDA_COMPLEXITY * comp))

def decide_sharing_for_manual_input(policy, manual, deterministic=True):
    present_mask = fields_to_mask(manual.present_fields)
    scenario_id = SCENARIO_NAME2ID[manual.scenario_name]

    # allowed mask used ONLY for reward training — not for inference
    state = build_state(present_mask, scenario_id)
    actions = policy.act(state, deterministic=deterministic)

    shared_by_group = {}

    for g, idxs in GROUP2TYPEIDX.items():
        act = actions[g]
        shared = apply_group_action(idxs, present_mask, act)
        shared_by_group[g] = [PII_TYPES[i] for i in shared]

    return DecisionOutput(
        scenario_name=manual.scenario_name,
        actions_by_group=actions,
        shared_fields_by_group=shared_by_group,
        present_fields=manual.present_fields,
    )