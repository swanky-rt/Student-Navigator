"""
Train the GRPO Rule Agent briefly on the Excel dataset,
and compare manual decisions before vs after training,
plus plot the training curve.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch.optim as optim

from grpo_config import NUM_PII, NUM_SCENARIOS
from grpo_mdp import ManualInput, decide_sharing_for_manual_input
from grpo_policy import RulePolicy
from grpo_train import (
    load_dataset_from_excel,
    rollout_batch,
    policy_gradient_update,
    evaluate_average_reward,
)


def print_decision(title, decision):
    print(f"=== {title} ===")
    print(f"Scenario: {decision.scenario_name}")
    print(f"Present fields: {decision.present_fields}\n")

    print("Actions by group (0=none, 1=allowed, 2=all):")
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

def main():
    # --- Hyperparameters for training ---
    NUM_ITERS = 200       # total training iterations
    BATCH_SIZE = 64       # transitions per iteration
    LOG_EVERY = 10        # evaluate avg reward every N iters

    # State dimension = 11 PII + 2 scenarios
    state_dim = NUM_PII + NUM_SCENARIOS
    policy = RulePolicy(state_dim=state_dim)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    # Manual input (we'll compare before vs after training)
    present = ["NAME", "EMAIL", "PHONE", "SSN"]
    allowed_restaurant = ["EMAIL", "PHONE"]
    allowed_bank = ["EMAIL", "PHONE", "DATE/DOB", "CREDIT_CARD", "SSN"]

    manual_input = ManualInput(
        present_fields=present,
        scenario_name="restaurant",
        allowed_fields_restaurant=allowed_restaurant,
        allowed_fields_bank=allowed_bank,
    )

    # --- Before training ---
    before_decision = decide_sharing_for_manual_input(policy, manual_input, deterministic=True)
    print_decision("Decision BEFORE training", before_decision)

    # --- Load dataset ---
    excel_path = Path("690-Project-Dataset.xlsx")  # adjust if needed
    if not excel_path.exists():
        print(f"[WARN] Dataset not found at {excel_path}. Training will be skipped.")
        return

    dataset_rows = load_dataset_from_excel(str(excel_path))
    print(f"Loaded {len(dataset_rows)} rows from dataset.")

    # --- Tiny training loop with logging ---
    iters = []
    avg_rewards = []

    print("Starting GRPO training loop...")
    for it in range(1, NUM_ITERS + 1):
        trans = rollout_batch(policy, dataset_rows, batch_size=BATCH_SIZE)
        policy_gradient_update(policy, optimizer, trans, epochs=2)

        if it % LOG_EVERY == 0:
            avg_r = evaluate_average_reward(policy, dataset_rows, num_samples=200)
            iters.append(it)
            avg_rewards.append(avg_r)
            print(f"[Iter {it}] Approx. average reward per group: {avg_r:.3f}")

    # --- After training ---
    after_decision = decide_sharing_for_manual_input(policy, manual_input, deterministic=True)
    print_decision("Decision AFTER training", after_decision)

    # --- Custom scenario tests on the TRAINED policy ---

    # Test 1: Restaurant – want NAME + EMAIL + PHONE
    # Here we override the allowed_restaurant set to include NAME as well.
    present1 = ["NAME", "EMAIL", "PHONE", "SSN"]
    allowed_rest1 = ["NAME", "EMAIL", "PHONE"]  # your desired behavior
    allowed_bank1 = ["EMAIL", "PHONE", "DATE/DOB", "CREDIT_CARD", "SSN"]

    run_test_case(
        policy,
        title="Test 1 – Restaurant (expect NAME + EMAIL + PHONE)",
        present=present1,
        scenario_name="restaurant",
        allowed_rest=allowed_rest1,
        allowed_bank=allowed_bank1,
    )

    # Test 2: Banking – share all details
    # Here we imagine a conversation that contains *all* PII fields,
    # and allowed_bank = all fields (max-utility scenario).
    from grpo_config import PII_TYPES
    present2 = list(PII_TYPES)  # all 11 PII types present
    allowed_rest2 = ["EMAIL", "PHONE"]  # still conservative for restaurant
    allowed_bank2 = list(PII_TYPES)  # bank is allowed to use everything

    run_test_case(
        policy,
        title="Test 2 – Bank (expect all details)",
        present=present2,
        scenario_name="bank",
        allowed_rest=allowed_rest2,
        allowed_bank=allowed_bank2,
    )

    # Test 3: Strict mode – share nothing
    # We simulate a scenario where *nothing* is allowed in either context.
    # Under our reward, the best action is to hide everything (high privacy, zero utility).
    present3 = ["NAME", "EMAIL", "PHONE", "SSN", "CREDIT_CARD"]
    allowed_rest3 = []  # no field is allowed
    allowed_bank3 = []  # no field is allowed

    run_test_case(
        policy,
        title="Test 3 – Strict mode (expect share nothing)",
        present=present3,
        scenario_name="restaurant",  # could also use "bank"; allowed sets are empty anyway
        allowed_rest=allowed_rest3,
        allowed_bank=allowed_bank3,
    )

    # --- Plot training curve ---
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

