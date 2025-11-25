"""
Configuration and constants for the GRPO Rule Agent MDP.
"""

from typing import Dict, List
from dataclasses import dataclass, field

# 11 PII types (fixed order)
PII_TYPES: List[str] = [
    "NAME",
    "PHONE",
    "EMAIL",
    "DATE/DOB",
    "company",
    "location",
    "IP",
    "SSN",
    "CREDIT_CARD",
    "age",
    "sex",
]
TYPE2IDX: Dict[str, int] = {t: i for i, t in enumerate(PII_TYPES)}

NUM_PII = len(PII_TYPES)

# Groups of PII fields
GROUPS: Dict[str, List[str]] = {
    "identity": ["NAME", "DATE/DOB", "age", "sex", "company", "location"],
    "contact": ["PHONE", "EMAIL"],
    "financial": ["SSN", "CREDIT_CARD"],
    "network": ["IP"],
}
GROUP2TYPEIDX: Dict[str, List[int]] = {
    g: [TYPE2IDX[t] for t in fields] for g, fields in GROUPS.items()
}

# Scenarios: Option A – only normal contexts (attack is eval-time, not a scenario bit)
SCENARIOS = {
    0: "restaurant",
    1: "bank",  # corresponds to the second allowed set
}
SCENARIO_NAME2ID = {v: k for k, v in SCENARIOS.items()}
NUM_SCENARIOS = len(SCENARIOS)

# Actions per group
NUM_ACTIONS = 3  # 0: share nothing, 1: share only allowed, 2: share all present

# Group sensitivity weights (w_g)
GROUP_WEIGHTS = {
    "financial": 1.5,  # was 2.0
    "identity": 1.0,
    "contact": 1.2,
    "network": 1.0,
}

# Scenario-specific (beta, alpha) for (privacy, utility)
# Restaurant more conservative
SCENARIO_WEIGHTS = {
    # still privacy-leaning, but more balanced
    "restaurant": {"beta": 0.6, "alpha": 0.4},
    # bank scenario: strongly utility-oriented
    "bank": {"beta": 0.3, "alpha": 0.7},
}

# Make complexity penalty much smaller so "share allowed/all"
# is not punished too much compared to utility gain.
LAMBDA_COMPLEXITY = 0.01
ACTION_COMPLEXITY = {
    0: 0.0,  # share nothing
    1: 0.1,  # share only allowed
    2: 0.2,  # share all present
}

def materialize_shared_values(
    shared_fields_by_group: Dict[str, List[str]],
    pii_values: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """
    Convert shared PII *types* (e.g., 'EMAIL', 'PHONE') into concrete values
    (e.g., 'alice@example.com', '555-123-4567') using the pii_values dict.
    """
    result: Dict[str, List[str]] = {}
    for group_name, types in shared_fields_by_group.items():
        vals: List[str] = []
        for t in types:
            # extend in case there are multiple values of the same type
            vals.extend(pii_values.get(t, []))
        result[group_name] = vals
    return result


@dataclass
class ManualInput:
    present_fields: List[str]
    scenario_name: str
    allowed_fields_restaurant: List[str]
    allowed_fields_bank: List[str]
    # NEW: map PII type -> list of concrete values found in the conversation
    pii_values: Dict[str, List[str]] = field(default_factory=dict)
