import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
base = pd.read_csv("results_baseline.csv")
atk = pd.read_csv("results_overthink.csv")

# Merge
merged = base[['id','reasoning_tokens']].merge(
    atk[['id','reasoning_tokens']],
    on='id',
    suffixes=('_base','_atk')
)

# Filter out prompts 1 and 7
merged = merged[~merged['id'].isin([1, 7])].reset_index(drop=True)

# Compute metrics
merged['token_overhead'] = merged['reasoning_tokens_atk'] - merged['reasoning_tokens_base']
merged['slowdown'] = merged['reasoning_tokens_atk'] / merged['reasoning_tokens_base']

# Prepare plot directory
plot_dir = "Plots"
os.makedirs(plot_dir, exist_ok=True)

x = np.arange(len(merged))
width = 0.35

# Create new x-axis labels: Prompt 1, Prompt 2, ...
prompt_labels = [f"Prompt {i+1}" for i in range(len(merged))]

# --- PLOT 1: TOKEN COUNTS SIDE BY SIDE ---
plt.figure(figsize=(12,6))
plt.bar(x - width/2, merged['reasoning_tokens_base'], width, label='Baseline')
plt.bar(x + width/2, merged['reasoning_tokens_atk'], width, label='Attacked')

plt.xlabel("Prompts")
plt.ylabel("Reasoning Tokens")
plt.title("Token Usage per Prompt (Baseline vs Attacked)")

plt.xticks(x, prompt_labels, rotation=45)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()

plot1_path = os.path.join(plot_dir, "token_usage_comparison.png")
plt.savefig(plot1_path)
plt.show()

# --- PLOT 2: SLOWDOWN SIDE BY SIDE ---
plt.figure(figsize=(12,6))
plt.bar(x - width/2, np.ones_like(merged['slowdown']), width, label='Baseline (1.0)')
plt.bar(x + width/2, merged['slowdown'], width, label='Attacked Slowdown')

plt.xlabel("Prompts")
plt.ylabel("Slowdown (atk/base)")
plt.title("Slowdown per Prompt (Baseline = 1.0)")

plt.xticks(x, prompt_labels, rotation=45)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()

plot2_path = os.path.join(plot_dir, "slowdown_comparison.png")
plt.savefig(plot2_path)
plt.show()

