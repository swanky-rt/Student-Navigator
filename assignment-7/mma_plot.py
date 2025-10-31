import json
import matplotlib.pyplot as plt
import os

# Path to results JSON file
results_path = "evaluation_results_per_sample.json"

if not os.path.exists(results_path):
    raise FileNotFoundError(f"Results file not found: {results_path}")

# --- Load the JSON file ---
with open(results_path, "r", encoding="utf-8") as f:
    data = json.load(f)

samples = data.get("samples", [])
summary = data.get("summary", {})

if not samples:
    raise ValueError("No samples found in the JSON file.")

total = len(samples)

# --- Compute basic counts from samples ---
def is_no(x: str) -> bool:
    return isinstance(x, str) and x.strip().lower() == "no"

def is_yes(x: str) -> bool:
    return isinstance(x, str) and x.strip().lower() == "yes"

# Count “No” responses per phase
clean_no = sum(1 for s in samples if is_no(s.get("clean_decision", "")))
adv_no = sum(1 for s in samples if is_no(s.get("adv_decision", "")))
defense_no = sum(1 for s in samples if is_no(s.get("defense_decision", "")))

# Attack success: adv decision differs from clean (adversary changed it)
attack_success_count = sum(
    1 for s in samples
    if s.get("clean_decision", "").strip().lower() != s.get("adv_decision", "").strip().lower()
)
attack_success_rate = (attack_success_count / total) * 100.0

# Defense success: defense matches clean baseline
defense_restores_count = sum(
    1 for s in samples
    if s.get("defense_decision", "").strip().lower() == s.get("clean_decision", "").strip().lower()
)
defense_success_rate = (defense_restores_count / total) * 100.0

# Percent of “No” responses per phase
clean_no_pct = (clean_no / total) * 100.0
adv_no_pct = (adv_no / total) * 100.0
defense_no_pct = (defense_no / total) * 100.0

# --- Print numeric results ---
print("\n=== Evaluation Summary ===")
print(f"Total samples: {total}")
print(f"Attack success count: {attack_success_count} / {total} -> {attack_success_rate:.1f}%")
print(f"Defense restores count: {defense_restores_count} / {total} -> {defense_success_rate:.1f}%")
print()
print(f'Percent of "No" responses -> Clean: {clean_no_pct:.1f}%, Adv: {adv_no_pct:.1f}%, Defense: {defense_no_pct:.1f}%')
print("\nFrom summary field:")
for k, v in summary.items():
    print(f"  {k}: {v}")

# --- Prepare bar plot ---
phases = ['Clean', 'Adversarial', 'Defense']
values_pct = [clean_no_pct, adv_no_pct, defense_no_pct]

plt.figure(figsize=(9,5))
bars = plt.bar(phases, values_pct)
plt.ylim(0, 100)
plt.ylabel("Percentage of 'No' responses (%)")
plt.title("Multi-Modal Adversarial Attack and JPEG Defense Results")

# Annotate bars with percentages
for bar, val in zip(bars, values_pct):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1.5, f"{val:.1f}%", ha='center', va='bottom')

plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()
