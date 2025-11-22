#!/usr/bin/env python3
"""
compute_metrics_full.py

Unified metrics & plotting for:
- Baseline
- Sudoku attack
- MDP attack

Generates a single merged CSV + multiple comparison plots.

Usage: python compute_metrics.py --base artifacts/results_baseline.csv --sudoku artifacts/results_sudoku.csv --mdp artifacts/results_mdp.csv --out artifacts/merged_all_attacks.csv --plots Plots/Combined   
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="results_baseline.csv")
    parser.add_argument("--sudoku", default="results_overthink.csv")
    parser.add_argument("--mdp", default="results_mdp.csv")
    parser.add_argument("--out", default="merged_all_attacks.csv")
    parser.add_argument("--plots", default="Plots")
    args = parser.parse_args()

    os.makedirs(args.plots, exist_ok=True)

    # -------------------------
    # Load all CSVs
    # -------------------------
    df_base = pd.read_csv(args.base)
    df_sudoku = pd.read_csv(args.sudoku)
    df_mdp = pd.read_csv(args.mdp)

    # Ensure consistent ID indexing
    df_base["id"] = range(len(df_base))
    df_sudoku["id"] = range(len(df_sudoku))
    df_mdp["id"] = range(len(df_mdp))

    # -------------------------
    # Merge all three datasets
    # -------------------------
    df = df_base.merge(df_sudoku, on="id", suffixes=("_base", "_sudoku"))
    df = df.merge(df_mdp, on="id")

    # Rename columns for clarity
    df = df.rename(columns={
        "reasoning_tokens": "reasoning_tokens_mdp",
        "cosine_similarity": "cosine_similarity_mdp"
    })

    # -------------------------
    # Compute metrics
    # -------------------------
    df["sudoku_overhead"] = df["reasoning_tokens_sudoku"] - df["reasoning_tokens_base"]
    df["mdp_overhead"] = df["reasoning_tokens_mdp"] - df["reasoning_tokens_base"]

    df["sudoku_slowdown"] = df["reasoning_tokens_sudoku"] / df["reasoning_tokens_base"]
    df["mdp_slowdown"] = df["reasoning_tokens_mdp"] / df["reasoning_tokens_base"]

    df["sudoku_sim_drop"] = df["cosine_similarity_base"] - df["cosine_similarity_sudoku"]
    df["mdp_sim_drop"] = df["cosine_similarity_base"] - df["cosine_similarity_mdp"]

    # -------------------------
    # Save merged CSV
    # -------------------------
    df.to_csv(args.out, index=False)
    print(f"[OK] Saved merged CSV: {args.out}")

    # -------------------------
    # Plotting
    # -------------------------

    ids = df["id"].tolist()
    x = np.arange(len(ids))
    qlabels = [f"Q{i+1}" for i in range(len(df))]
    width = 0.35

    # ---- PLOT 1: Baseline vs Sudoku ----
    plt.figure(figsize=(11, 6))
    plt.plot(x, df["reasoning_tokens_base"], marker="o", label="Baseline", linewidth=2)
    plt.plot(x, df["reasoning_tokens_sudoku"], marker="o", label="Sudoku Attack", linewidth=2)
    plt.xticks(x, qlabels, rotation=45)
    plt.ylabel("Tokens")
    plt.title("Baseline vs Sudoku Attack — Token Usage")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.plots}/baseline_vs_sudoku_line.png")
    plt.close()

    # ---- PLOT 2: Baseline vs MDP ----
    plt.figure(figsize=(11, 6))
    plt.plot(x, df["reasoning_tokens_base"], marker="o", label="Baseline", linewidth=2)
    plt.plot(x, df["reasoning_tokens_mdp"], marker="o", label="MDP Attack", linewidth=2)
    plt.xticks(x, qlabels, rotation=45)
    plt.ylabel("Tokens")
    plt.title("Baseline vs MDP Attack — Token Usage")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.plots}/baseline_vs_mdp_line.png")
    plt.close()

    # ---- PLOT 3: Token Overhead Comparison ----
    plt.figure(figsize=(11, 6))
    plt.bar(x - width/2, df["sudoku_overhead"], width, label="Sudoku Overhead")
    plt.bar(x + width/2, df["mdp_overhead"], width, label="MDP Overhead")
    plt.xticks(x, qlabels, rotation=45)
    plt.ylabel("Extra Tokens")
    plt.title("Token Overhead: Sudoku vs MDP Attacks")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.plots}/overhead_sudoku_mdp.png")
    plt.close()

    # ---- PLOT 4: Slowdown Comparison ----
    plt.figure(figsize=(11, 6))
    plt.bar(x - width/2, df["sudoku_slowdown"], width, label="Sudoku Slowdown")
    plt.bar(x + width/2, df["mdp_slowdown"], width, label="MDP Slowdown")
    plt.xticks(x, qlabels, rotation=45)
    plt.ylabel("Slowdown Ratio (Attack / Base)")
    plt.title("Slowdown: Sudoku vs MDP Attacks")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.plots}/slowdown_sudoku_mdp.png")
    plt.close()

    print(f"\n[OK] All plots saved under: {args.plots}/")


if __name__ == "__main__":
    main()
