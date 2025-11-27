# grpo_mdp.py
"""
Manual input helper for the learned-mask GRPO Rule Agent.
"""

from dataclasses import dataclass
from typing import Dict, List

from Minimiser.utils.config import PII_TYPES, TYPE2IDX
from Minimiser.utils.state import fields_to_mask, build_state
from Minimiser.grpo_policy import RulePolicy


@dataclass
class ManualInput:
    present_fields: List[str]
    scenario_name: str  # "restaurant" or "bank"
    pii_values: Dict[str, List[str]]  # e.g. {"EMAIL": ["alice@example.com"], ...}


@dataclass
class ManualDecisionOutput:
    scenario_name: str
    shared_types: List[str]
    shared_values: Dict[str, List[str]]
    share_mask: List[int]
    present_fields: List[str]


def decide_sharing_for_manual_input(
    policy: RulePolicy,
    manual: ManualInput,
    deterministic: bool = True,
) -> ManualDecisionOutput:
    """
    Build a state from manual present_fields + scenario_name,
    run the policy, and return which fields (and values) would be shared.
    """
    present_mask = fields_to_mask(manual.present_fields)
    state = build_state(present_mask, manual.scenario_name)

    share_mask, _ = policy.act(state, deterministic=deterministic)

    shared_types: List[str] = []
    shared_values: Dict[str, List[str]] = {}

    for i, share_bit in enumerate(share_mask):
        if share_bit != 1:
            continue
        pii_type = PII_TYPES[i]
        if pii_type not in manual.present_fields:
            continue
        shared_types.append(pii_type)
        if manual.pii_values and pii_type in manual.pii_values:
            shared_values[pii_type] = manual.pii_values[pii_type]

    return ManualDecisionOutput(
        scenario_name=manual.scenario_name,
        shared_types=shared_types,
        shared_values=shared_values,
        share_mask=share_mask,
        present_fields=manual.present_fields,
    )
