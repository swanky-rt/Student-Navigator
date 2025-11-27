# utils/state.py
from typing import List

import torch

from .config import PII_TYPES, TYPE2IDX, NUM_SCENARIOS, SCENARIO_NAME2ID


def fields_to_mask(fields: List[str]) -> List[int]:
    """
    Convert a list of PII type names into a binary mask of length len(PII_TYPES).
    """
    mask = [0] * len(PII_TYPES)
    for f in fields:
        if f in TYPE2IDX:
            mask[TYPE2IDX[f]] = 1
    return mask


def build_state(present_mask: List[int], scenario_name: str) -> torch.Tensor:
    """
    State = [present_mask || scenario_one_hot]
    present_mask: length NUM_PII
    scenario_one_hot: length NUM_SCENARIOS
    """
    scenario_id = SCENARIO_NAME2ID[scenario_name]
    scenario_oh = [0] * NUM_SCENARIOS
    scenario_oh[scenario_id] = 1
    state_vec = present_mask + scenario_oh
    return torch.tensor(state_vec, dtype=torch.float32)
