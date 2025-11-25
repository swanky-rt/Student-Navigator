"""
MDP helpers for the GRPO Rule Agent:
- State construction
- Action semantics
- Reward computation
- Manual input -> decision API
"""

from dataclasses import dataclass
from typing import Dict, List

import torch

from grpo_config import (
    PII_TYPES,
    TYPE2IDX,
    GROUPS,
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
    """
    Input for manual demo:
    - present_fields: which PII types appear in this conversation
    - scenario_name: "restaurant" or "bank"
    - allowed_fields_restaurant / allowed_fields_bank: normative allowed sets
      (in  dataset these come from allowed_restaurant and Unnamed: 3).
    """
    present_fields: List[str]
    scenario_name: str
    allowed_fields_restaurant: List[str]
    allowed_fields_bank: List[str]


@dataclass
class DecisionOutput:
    """
    Output of policy decision for a single manual input.
    """
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


def mask_to_fields(mask: List[int]) -> List[str]:
    return [PII_TYPES[i] for i, v in enumerate(mask) if v == 1]


def build_state(present_mask: List[int], scenario_id: int) -> torch.Tensor:
    """
    State = [present_mask || scenario_one_hot]
    present_mask: length NUM_PII
    scenario_one_hot: length NUM_SCENARIOS
    """
    scenario_oh = [0] * NUM_SCENARIOS
    scenario_oh[scenario_id] = 1
    state_vec = present_mask + scenario_oh
    return torch.tensor(state_vec, dtype=torch.float32)


def apply_group_action(
    group_type_indices: List[int],
    present_mask: List[int],
    allowed_mask: List[int],
    action: int,
) -> List[int]:
    """
    Returns indices of fields actually shared in this group.
    action:
      0: share nothing
      1: share only allowed
      2: share all present
    """
    present_indices = [i for i in group_type_indices if present_mask[i] == 1]
    if not present_indices:
        return []

    if action == 0:
        return []
    elif action == 1:
        return [i for i in present_indices if allowed_mask[i] == 1]
    elif action == 2:
        return present_indices
    else:
        raise ValueError(f"Invalid action {action}")


def compute_group_reward(
    group_name: str,
    scenario_name: str,
    group_type_indices: List[int],
    present_mask: List[int],
    allowed_mask: List[int],
    shared_indices: List[int],
    action: int,
) -> float:
    """
    Compute privacy/utility reward for one group under a specific scenario.
    r_g = w_g * (beta_c * privacy_g + alpha_c * utility_g - lambda * complexity(a_g))
    """
    present = [i for i in group_type_indices if present_mask[i] == 1]
    if not present:
        # No fields in this group -> neutral
        return 0.0

    allowed = [i for i in present if allowed_mask[i] == 1]
    disallowed = [i for i in present if allowed_mask[i] == 0]

    shared_set = set(shared_indices)
    shared_allowed = [i for i in allowed if i in shared_set]
    shared_disallowed = [i for i in disallowed if i in shared_set]

    # Utility: fraction of allowed that were shared
    if allowed:
        utility = len(shared_allowed) / len(allowed)
    else:
        utility = 0.0

    # Privacy: 1 - fraction of disallowed leaked
    if disallowed:
        privacy = 1.0 - (len(shared_disallowed) / len(disallowed))
    else:
        privacy = 1.0

    # Scenario weights
    if scenario_name not in SCENARIO_WEIGHTS:
        raise ValueError(f"Unknown scenario name: {scenario_name}")
    beta = SCENARIO_WEIGHTS[scenario_name]["beta"]
    alpha = SCENARIO_WEIGHTS[scenario_name]["alpha"]

    # Group sensitivity
    w_g = GROUP_WEIGHTS.get(group_name, 1.0)

    # Complexity penalty
    complexity = ACTION_COMPLEXITY.get(action, 0.0)

    reward = w_g * (beta * privacy + alpha * utility - LAMBDA_COMPLEXITY * complexity)
    return float(reward)


def decide_sharing_for_manual_input(
    policy,
    manual: ManualInput,
    deterministic: bool = True,
) -> DecisionOutput:
    """
    Main function for now: manually specify present fields & scenario,
    plus allowed sets, and ask the policy what to share.
    """
    # Present mask
    present_mask = fields_to_mask(manual.present_fields)

    # Scenario id/name
    if manual.scenario_name not in SCENARIO_NAME2ID:
        raise ValueError(f"Unknown scenario name: {manual.scenario_name}")
    scenario_id = SCENARIO_NAME2ID[manual.scenario_name]

    # Allowed mask based on scenario
    if manual.scenario_name == "restaurant":
        allowed_mask = fields_to_mask(manual.allowed_fields_restaurant)
    else:  # "bank"
        allowed_mask = fields_to_mask(manual.allowed_fields_bank)

    # Build state
    state = build_state(present_mask, scenario_id)

    # Get actions per group from policy
    actions_by_group = policy.act(state, deterministic=deterministic)

    # Compute shared fields per group
    shared_fields_by_group: Dict[str, List[str]] = {}
    for g, type_indices in GROUP2TYPEIDX.items():
        action = actions_by_group[g]
        shared_indices = apply_group_action(
            type_indices,
            present_mask,
            allowed_mask,
            action,
        )
        shared_fields = [PII_TYPES[i] for i in shared_indices]
        shared_fields_by_group[g] = shared_fields

    return DecisionOutput(
        scenario_name=manual.scenario_name,
        actions_by_group=actions_by_group,
        shared_fields_by_group=shared_fields_by_group,
        present_fields=manual.present_fields,
    )
