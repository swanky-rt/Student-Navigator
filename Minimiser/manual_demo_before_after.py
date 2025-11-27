# manual_demo_before_after.py
"""
Train the GRPO Rule Agent briefly on the Excel dataset,
and compare manual decisions before vs after training,
plus plot the training curve.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch.optim as optim

from Minimiser.utils.config import NUM_PII, NUM_SCENARIOS
from Minimiser.utils.dataset import load_dataset_from_excel
from Minimiser.grpo_mdp import ManualInput, decide_sharing_for_manual_input
from Minimiser.grpo_policy import RulePolicy
from Minimiser.grpo_train import (
    rollout_batch,
    policy_gradient_update,
    evaluate_average_reward,
)


def main():
    excel_path = Path("690-Project-Dataset.xlsx")

    # === 1. Construct policy ===
    state_dim = NUM_PII + NUM_SCENARIOS
    policy = RulePolicy(state_dim=state_dim)

    # === 2. Manual example (before training) ===
    manual_input = ManualInput(
        present_fields=["NAME", "EMAIL", "PHONE", "SSN"],
        scenario_name="restaurant",
        pii_values={
            "NAME": ["Alice"],
            "EMAIL": ["alice@example.com"],
            "PHONE": ["555-123-4567"],
            "SSN": ["123-45-6789"],
        },
    )

    print("=== Decision BEFORE training ===")
    before = decide_sharing_for_manual_input(policy, manual_input, deterministic=True)
    print("Scenario:", before.scenario_name)
    print("Present fields:", before.present_fields)
    print("Shared types:", before.shared_types)
    print("Shared values:", before.shared_values)

    # === 3. Load dataset and train briefly ===
    dataset_rows = load_dataset_from_excel(str(excel_path))
    print(f"Loaded {len(dataset_rows)} rows from dataset.")
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    iters = []
    avg_rewards = []

    print("Starting GRPO training loop...")
    num_iterations = 200
    batch_size = 64

    for it in range(1, num_iterations + 1):
        batch = rollout_batch(policy, dataset_rows, batch_size=batch_size)
        policy_gradient_update(policy, optimizer, batch, epochs=1)

        if it % 10 == 0:
            avg_r = evaluate_average_reward(policy, dataset_rows, num_samples=200)
            iters.append(it)
            avg_rewards.append(avg_r)
            print(f"[Iter {it}] Approx. average reward per episode: {avg_r:.3f}")

    # === 4. Manual example (after training) ===
    print("=== Decision AFTER training ===")
    after = decide_sharing_for_manual_input(policy, manual_input, deterministic=True)
    print("Scenario:", after.scenario_name)
    print("Present fields:", after.present_fields)
    print("Shared types:", after.shared_types)
    print("Shared values:", after.shared_values)

    # === 5. Plot training curve ===
    if iters:
        plt.figure()
        plt.plot(iters, avg_rewards)
        plt.xlabel("Iteration")
        plt.ylabel("Average reward per episode")
        plt.title("GRPO Rule Agent Training Curve (learned mask)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig("training_curve.png", dpi=200)
        print("Saved training curve to training_curve.png")


if __name__ == "__main__":
    main()
