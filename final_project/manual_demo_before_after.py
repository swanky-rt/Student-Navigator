"""
Train the GRPO Rule Agent briefly on the Excel dataset,
compare decisions before/after training,
print learned allowed-details per domain,
and plot the training curve.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import torch.optim as optim

from utils.config import NUM_PII, NUM_SCENARIOS, PII_TYPES
from utils.mdp import ManualInput, decide_sharing_for_manual_input
from GRPO.grpo_policy import RulePolicy
from GRPO.grpo_train import (
    load_dataset_from_excel,
    rollout_batch,
    policy_gradient_update,
    evaluate_average_reward,
)


# ============================================================
#  NEW: Extract allowed-details per domain
# ============================================================

def infer_allowed_mask(policy, scenario_name: str):
    """
    Infer the agent's learned 'allowed' PII types for a domain (restaurant/bank)
    by probing one PII type at a time and seeing if the policy chooses to share it.
    """

    allowed_estimate = []

    for pii in PII_TYPES:
        # Present only 1 PII type
        manual = ManualInput(
            present_fields=[pii],
            scenario_name=scenario_name,
            allowed_fields_restaurant=[],   # irrelevant to policy
            allowed_fields_bank=[],         # irrelevant to policy
        )

        decision = decide_sharing_for_manual_input(policy, manual, deterministic=True)

        # If ANY group shares this PII → treat as allowed
        shared_flat = []
        for g, fields in decision.shared_fields_by_group.items():
            shared_flat.extend(fields)

        if pii in shared_flat:
            allowed_estimate.append(pii)

    return allowed_estimate



# ============================================================
#  Existing printing helpers
# ============================================================

def print_decision(title, decision):
    print(f"\n=== {title} ===")
    print(f"Scenario: {decision.scenario_name}")
    print(f"Present fields: {decision.present_fields}\n")

    # Updated wording (correct semantics)
    print("Actions by group (0=none, 1=share all present, 2=share subset):")
    for g, a in decision.actions_by_group.items():
        print(f"  {g}: action {a}")

    print("\nShared fields by group:")
    for g, fields in decision.shared_fields_by_group.items():
        print(f"  {g}: {fields}")
    print()


def run_test_case(policy, title, present, scenario_name, allowed_rest, allowed_bank):
    manual_input = ManualInput(
        present_fields=present,
        scenario_name=scenario_name,
        allowed_fields_restaurant=allowed_rest,
        allowed_fields_bank=allowed_bank,
    )
    decision = decide_sharing_for_manual_input(policy, manual_input, deterministic=True)
    print_decision(title, decision)



# ============================================================
#  MAIN TRAINING SCRIPT
# ============================================================

def main():
    # -------------------------------
    #  Hyperparameters
    # -------------------------------
    NUM_ITERS = 200
    BATCH_SIZE = 64
    LOG_EVERY = 10

    # Policy + optimizer
    state_dim = NUM_PII + NUM_SCENARIOS
    policy = RulePolicy(state_dim=state_dim)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    # -------------------------------
    #  Manual Input (Before Training)
    # -------------------------------
    present = ["NAME", "EMAIL", "PHONE", "SSN"]
    allowed_restaurant = []
    allowed_bank = []

    manual_input = ManualInput(
        present_fields=present,
        scenario_name="restaurant",
    )

    before_decision = decide_sharing_for_manual_input(policy, manual_input, deterministic=True)
    print_decision("Decision BEFORE training", before_decision)

    # -------------------------------
    #  Load dataset
    # -------------------------------
    excel_path = Path("final_project/690-Project-Dataset.csv")
    if not excel_path.exists():
        print(f"[WARN] Dataset not found at {excel_path}. Training skipped.")
        return

    dataset_rows = load_dataset_from_excel(str(excel_path))
    print(f"Loaded {len(dataset_rows)} rows from dataset.")

    # -------------------------------
    #  Training loop
    # -------------------------------
    iters = []
    avg_rewards = []

    print("\nStarting GRPO training loop...")
    for it in range(1, NUM_ITERS + 1):
        trans = rollout_batch(policy, dataset_rows, batch_size=BATCH_SIZE)
        policy_gradient_update(policy, optimizer, trans, epochs=2)

        if it % LOG_EVERY == 0:
            avg_r = evaluate_average_reward(policy, dataset_rows, num_samples=200)
            iters.append(it)
            avg_rewards.append(avg_r)
            print(f"[Iter {it}] Avg reward per group: {avg_r:.3f}")

    # -------------------------------
    #  After training
    # -------------------------------
    after_decision = decide_sharing_for_manual_input(policy, manual_input, deterministic=True)
    print_decision("Decision AFTER training", after_decision)

    # ============================================================
    #        🌟 NEW: Learned Allowed-Details per Domain
    # ============================================================

    print("\n=== Learned Allowed Details (Restaurant) ===")
    print(infer_allowed_mask(policy, "restaurant"))

    print("\n=== Learned Allowed Details (Bank) ===")
    print(infer_allowed_mask(policy, "bank"))

    # # -------------------------------
    # #  Custom Tests (unchanged)
    # # -------------------------------
    # present1 = ["NAME", "EMAIL", "PHONE", "SSN"]
    # allowed_rest1 = ["NAME", "EMAIL", "PHONE"]
    # allowed_bank1 = ["EMAIL", "PHONE", "DATE/DOB", "CREDIT_CARD", "SSN"]

    # run_test_case(
    #     policy,
    #     title="Test 1 – Restaurant (expect NAME+EMAIL+PHONE)",
    #     present=present1,
    #     scenario_name="restaurant",
    #     allowed_rest=allowed_rest1,
    #     allowed_bank=allowed_bank1,
    # )

    # present2 = list(PII_TYPES)
    # allowed_rest2 = ["EMAIL", "PHONE"]
    # allowed_bank2 = list(PII_TYPES)

    # run_test_case(
    #     policy,
    #     title="Test 2 – Bank (Expect all details)",
    #     present=present2,
    #     scenario_name="bank",
    #     allowed_rest=allowed_rest2,
    #     allowed_bank=allowed_bank2,
    # )

    # present3 = ["NAME", "EMAIL", "PHONE", "SSN", "CREDIT_CARD"]
    # run_test_case(
    #     policy,
    #     title="Test 3 – Strict Mode (Expect share nothing)",
    #     present=present3,
    #     scenario_name="restaurant",
    #     allowed_rest=[],
    #     allowed_bank=[],
    # )

    # -------------------------------
    #  Plot training curve
    # -------------------------------
    if iters:
        plt.figure()
        plt.plot(iters, avg_rewards)
        plt.xlabel("Iteration")
        plt.ylabel("Average reward per group")
        plt.title("GRPO Rule Agent Training Curve")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig("training_curve.png", dpi=200)
        print("Saved training curve to training_curve.png")


if __name__ == "__main__":
    main()