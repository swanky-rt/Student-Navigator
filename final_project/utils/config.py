# # utils/config.py
# """
# Configuration and constants for the GRPO Rule Agent MDP.
# """

# from typing import Dict, List
# from dataclasses import dataclass, field

# # 11 PII types (fixed order)
# PII_TYPES: List[str] = [
#     "NAME",
#     "PHONE",
#     "EMAIL",
#     "DATE/DOB",
#     "company",
#     "location",
#     "IP",
#     "SSN",
#     "CREDIT_CARD",
#     "age",
#     "sex",
# ]
# TYPE2IDX: Dict[str, int] = {t: i for i, t in enumerate(PII_TYPES)}
# NUM_PII = len(PII_TYPES)

# # Groups of PII fields
# GROUPS: Dict[str, List[str]] = {
#     "identity": ["NAME", "DATE/DOB", "age", "sex", "company", "location"],
#     "contact": ["PHONE", "EMAIL"],
#     "financial": ["SSN", "CREDIT_CARD"],
#     "network": ["IP"],
# }
# GROUP2TYPEIDX: Dict[str, List[int]] = {
#     g: [TYPE2IDX[t] for t in fields] for g, fields in GROUPS.items()
# }

# SCENARIOS = {
#     0: "restaurant",
#     1: "bank",
# }
# SCENARIO_NAME2ID = {v: k for k, v in SCENARIOS.items()}
# NUM_SCENARIOS = len(SCENARIOS)

# # Action meanings are implemented in apply_group_action()
# NUM_ACTIONS = 3  

# GROUP_WEIGHTS = {
#     "financial": 1.5,
#     "identity": 1.0,
#     "contact": 1.2,
#     "network": 1.0,
# }

# SCENARIO_WEIGHTS = {
#     "restaurant": {"beta": 0.6, "alpha": 0.4},
#     "bank": {"beta": 0.3, "alpha": 0.7},
# }

# LAMBDA_COMPLEXITY = 0.01
# ACTION_COMPLEXITY = {
#     0: 0.0,
#     1: 0.1,
#     2: 0.2,
# }

# @dataclass
# class ManualInput:
#     present_fields: List[str]
#     scenario_name: str
#     allowed_fields_restaurant: List[str]
#     allowed_fields_bank: List[str]
#     pii_values: Dict[str, List[str]] = field(default_factory=dict)


# utils/config.py

from typing import List, Dict

# All PII types in fixed order
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

NUM_PII = len(PII_TYPES)

TYPE2IDX = {t: i for i, t in enumerate(PII_TYPES)}

# Scenarios
SCENARIO_NAME2ID = {"restaurant": 0, "bank": 1}
NUM_SCENARIOS = 2

# Group definitions
GROUPS = ["identity", "contact", "financial", "network"]

GROUP2TYPEIDX = {
    "identity": [TYPE2IDX["NAME"], TYPE2IDX["DATE/DOB"], TYPE2IDX["age"], TYPE2IDX["sex"]],
    "contact": [TYPE2IDX["PHONE"], TYPE2IDX["EMAIL"]],
    "financial": [TYPE2IDX["CREDIT_CARD"], TYPE2IDX["SSN"]],
    "network": [TYPE2IDX["IP"], TYPE2IDX["company"], TYPE2IDX["location"]],
}

# Reward weights
SCENARIO_WEIGHTS: Dict[str, Dict[str, float]] = {
    "restaurant": {"alpha": 0.4, "beta": 0.6},
    "bank": {"alpha": 0.7, "beta": 0.3},
}

GROUP_WEIGHTS = {
    "identity": 1.0,
    "contact": 1.0,
    "financial": 1.0,
    "network": 1.0,
}

# Regularization
LAMBDA_COMPLEXITY = 0.05

# Action cost
ACTION_COMPLEXITY = {
    0: 0.0,
    1: 0.2,
    2: 0.5,
}