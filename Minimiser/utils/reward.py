# utils/reward.py
from typing import List

from .config import (
    GROUP2TYPEIDX,
    GROUP_WEIGHTS,
    SCENARIO_WEIGHTS,
    LAMBDA_COMPLEXITY,
)


def compute_episode_reward(
    present_mask: List[int],
    allowed_mask: List[int],
    share_mask: List[int],
    scenario_name: str,
) -> float:
    """
    Compute scalar reward for one conversation (one episode) given:
    - which PII types are present (present_mask),
    - which are allowed in this scenario (allowed_mask; hidden from policy),
    - which fields the policy decided to share (share_mask),
    - and the scenario name ("restaurant" or "bank").

    Reward is based on:
      - utility: fraction of allowed fields correctly shared (per group),
      - privacy: 1 - fraction of disallowed fields leaked (per group),
      - group sensitivity weights,
      - a small complexity penalty for sharing too many fields overall.
    """
    beta = SCENARIO_WEIGHTS[scenario_name]["beta"]
    alpha = SCENARIO_WEIGHTS[scenario_name]["alpha"]

    total_group_reward = 0.0
    num_groups_used = 0

    for g, idxs in GROUP2TYPEIDX.items():
        # Fields in this group that are actually present in the conversation
        present = [i for i in idxs if present_mask[i] == 1]
        if not present:
            continue

        allowed = [i for i in present if allowed_mask[i] == 1]
        disallowed = [i for i in present if allowed_mask[i] == 0]
        shared = [i for i in present if share_mask[i] == 1]

        # Utility: share allowed fields
        if len(allowed) > 0:
            shared_allowed = sum(1 for i in shared if i in allowed)
            utility_g = shared_allowed / len(allowed)
        else:
            utility_g = 0.0

        # Privacy: avoid leaking disallowed fields
        if len(disallowed) > 0:
            shared_disallowed = sum(1 for i in shared if i in disallowed)
            privacy_g = 1.0 - (shared_disallowed / len(disallowed))
        else:
            privacy_g = 1.0

        w_g = GROUP_WEIGHTS[g]
        r_g = w_g * (beta * privacy_g + alpha * utility_g)

        total_group_reward += r_g
        num_groups_used += 1

    if num_groups_used > 0:
        base_reward = total_group_reward / num_groups_used
    else:
        base_reward = 0.0

    # Complexity penalty: if you share many of the present fields, penalize slightly
    present_total = sum(present_mask)
    shared_total = sum(
        1
        for i in range(len(present_mask))
        if present_mask[i] == 1 and share_mask[i] == 1
    )
    if present_total > 0:
        complexity = shared_total / present_total
    else:
        complexity = 0.0

    final_reward = base_reward - LAMBDA_COMPLEXITY * complexity
    return float(final_reward)
