#!/usr/bin/env python3
"""
plot_paraphrase_defense.py

Creates 3 comparison plots:

    • defense_sudoku.png
    • defense_mdp.png
    • defense_combined.png

Each plot compares:
    Baseline  vs  Attack  vs  Paraphrase Defense

Input files expected:
    results_baseline.csv
    results_sudoku.csv
    results_mdp.csv
    results_defended_paraphrase_sudoku.csv
    results_defended_paraphrase_mdp.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================
# Helper to compute plot
# ============================================================
def make_defense_plot(name, base_df, atk_df, def_df, out_path):

    base = base_df["reasoning_tokens"].to_numpy()
    atk  = atk_df["reasoning_tokens"].to_numpy()
    dfn  = def_df["reasoning_tokens"].to_numpy()

    m_base, s_base = base.mean(), base.std()
    m_atk,  s_atk  = atk.mean(), atk.std()
    m_def,  s_def  = dfn.mean(), dfn.std()

    labels = ["Baseline", f"{name} Attack", "Paraphrase Defense"]
    means  = [m_base, m_atk, m_def]
    stds   = [s_base, s_atk, s_def]

    x = np.arange(len(labels))

    plt.figure(figsize=(12,6))
    plt.bar(x, means, yerr=stds, capsize=10,
            color=["#4e79a7", "#f28e2c", "#af7aa1"], alpha=0.9)

    for i, val in enumerate(means):
        plt.text(i, val + stds[i] + 10,
                 f"{int(val)}",
                 ha="center", fontsize=14, fontweight="bold")

    plt.xticks(x, labels, fontsize=12)
    plt.ylabel("Reasoning Tokens (Count)", fontsize=14)
    plt.title(f"Reasoning Token Usage: Baseline vs {name} Attack vs Paraphrase Defense",
              fontsize=16, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"[OK] Saved → {out_path}")


# ============================================================
# Combined plot: Sudoku + MDP in one figure
# ============================================================
def make_combined_plot(base, sudoku, mdp, def_sudoku, def_mdp, out_path):

    # Means
    m_base  = base["reasoning_tokens"].mean()
    m_su_atk = sudoku["reasoning_tokens"].mean()
    m_md_atk = mdp["reasoning_tokens"].mean()
    m_su_def = def_sudoku["reasoning_tokens"].mean()
    m_md_def = def_mdp["reasoning_tokens"].mean()

    # Stds
    s_base  = base["reasoning_tokens"].std()
    s_su_atk = sudoku["reasoning_tokens"].std()
    s_md_atk = mdp["reasoning_tokens"].std()
    s_su_def = def_sudoku["reasoning_tokens"].std()
    s_md_def = def_mdp["reasoning_tokens"].std()

    labels = ["Sudoku", "MDP"]
    x = np.arange(len(labels))
    width = 0.22

    plt.figure(figsize=(14,7))

    # 3 bars per attack type
    plt.bar(x - width,     [m_base,    m_base],    width, yerr=[s_base, s_base],    capsize=6, label="Baseline")
    plt.bar(x,             [m_su_atk,  m_md_atk],  width, yerr=[s_su_atk, s_md_atk], capsize=6, label="Attack")
    plt.bar(x + width,     [m_su_def,  m_md_def],  width, yerr=[s_su_def, s_md_def], capsize=6, label="Paraphrase Defense")

    # Add numeric labels
    means_all = [
        (m_base, m_su_atk, m_su_def),
        (m_base, m_md_atk, m_md_def)
    ]

    for i, group in enumerate(means_all):
        for j, val in enumerate(group):
            x_pos = i + (j - 1) * width
            plt.text(x_pos, val + 10, f"{int(val)}", ha="center",
                     fontsize=11, fontweight="bold")

    plt.xticks(x, labels, fontsize=13)
    plt.ylabel("Reasoning Tokens", fontsize=14)
    plt.title("Baseline vs Attack vs Paraphrase Defense — Sudoku & MDP Combined",
              fontsize=16, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"[OK] Saved combined → {out_path}")


# ============================================================
# Main
# ============================================================
def main():

    base   = pd.read_csv("artifacts/results_baseline.csv")
    sudoku = pd.read_csv("artifacts/results_sudoku.csv")
    mdp    = pd.read_csv("artifacts/results_mdp.csv")

    def_sudoku = pd.read_csv("artifacts/results_defended_paraphrase_sudoku.csv")
    def_mdp    = pd.read_csv("artifacts/results_defended_paraphrase_mdp.csv")

    out_dir = "Plots/Defense_Paraphrase"
    os.makedirs(out_dir, exist_ok=True)

    # Individual plots
    make_defense_plot("Sudoku", base, sudoku, def_sudoku, f"{out_dir}/defense_sudoku.png")
    make_defense_plot("MDP",    base, mdp,    def_mdp,    f"{out_dir}/defense_mdp.png")

    # Combined plot
    make_combined_plot(
        base, sudoku, mdp,
        def_sudoku, def_mdp,
        f"{out_dir}/defense_combined.png"
    )

    print("\n[ALL DONE] All defense plots generated.\n")


if __name__ == "__main__":
    main()
