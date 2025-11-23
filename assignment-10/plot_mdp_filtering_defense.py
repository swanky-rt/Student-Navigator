#!/usr/bin/env python3
"""
plot_mdp_filtering_defense.py.py

Create a single bar plot comparing reasoning token lengths for:

  • Baseline (no attack)
  • MDP attack
  • MDP Filtering Defense

Output:
  Plots/defense_mdp_filter.png
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def make_mdp_plot(base_df, mdp_atk_df, mdp_def_df, out_path: str):
    # Mean reasoning tokens
    m_base = base_df["reasoning_tokens"].mean()
    m_atk = mdp_atk_df["reasoning_tokens"].mean()
    m_def = mdp_def_df["reasoning_tokens"].mean()

    # Std dev
    s_base = base_df["reasoning_tokens"].std()
    s_atk = mdp_atk_df["reasoning_tokens"].std()
    s_def = mdp_def_df["reasoning_tokens"].std()

    labels = ["Baseline", "MDP attack", "MDP filter defense"]
    means = [m_base, m_atk, m_def]
    stds = [s_base, s_atk, s_def]

    plt.figure()
    x = range(len(labels))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, labels, rotation=15)
    plt.ylabel("Mean reasoning tokens")
    plt.title("MDP slowdown: Baseline vs Attack vs Filtering Defense")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="results_baseline.csv")
    parser.add_argument("--mdp-attack", default="results_mdp.csv")
    parser.add_argument("--mdp-filter", default="results_filtering_mdp.csv")
    parser.add_argument("--outdir", default="Plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    base = pd.read_csv(args.base)
    mdp_atk = pd.read_csv(args.mdp_attack)
    mdp_def = pd.read_csv(args.mdp_filter)

    out_path = os.path.join(args.outdir, "defense_mdp_filter.png")
    make_mdp_plot(base, mdp_atk, mdp_def, out_path)
    print(f"[OK] Wrote MDP filtering defense plot to {out_path}")


if __name__ == "__main__":
    main()
