# utils/config.py
"""
Configuration and constants for the GRPO Rule Agent (learned mask version).
"""

from typing import Dict, List

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

TYPE2IDX: Dict[str, int] = {name: i for i, name in enumerate(PII_TYPES)}
NUM_PII: int = len(PII_TYPES)

# 2 scenarios
SCENARIOS: List[str] = ["restaurant", "bank"]
SCENARIO_NAME2ID: Dict[str, int] = {name: i for i, name in enumerate(SCENARIOS)}
NUM_SCENARIOS: int = len(SCENARIOS)

# Grouping of PII types (by index) for reward computation
GROUP2TYPEIDX: Dict[str, List[int]] = {
    "identity": [
        TYPE2IDX["NAME"],
        TYPE2IDX["DATE/DOB"],
        TYPE2IDX["age"],
        TYPE2IDX["sex"],
        TYPE2IDX["company"],
        TYPE2IDX["location"],
    ],
    "contact": [
        TYPE2IDX["EMAIL"],
        TYPE2IDX["PHONE"],
    ],
    "financial": [
        TYPE2IDX["SSN"],
        TYPE2IDX["CREDIT_CARD"],
    ],
    "network": [
        TYPE2IDX["IP"],
    ],
}

# Sensitivity weights per group (financial > identity > contact > network)
GROUP_WEIGHTS: Dict[str, float] = {
    "identity": 1.5,
    "contact": 1.0,
    "financial": 2.0,
    "network": 0.8,
}

# Scenario-specific (beta, alpha) for (privacy, utility)
# Restaurant more conservative
SCENARIO_WEIGHTS: Dict[str, Dict[str, float]] = {
    "restaurant": {"beta": 0.7, "alpha": 0.3},
    "bank": {"beta": 0.6, "alpha": 0.4},
}

# Complexity penalty (penalize sharing too many fields overall)
LAMBDA_COMPLEXITY: float = 0.05
