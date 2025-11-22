#!/usr/bin/env python3
"""
compute_metrics_full.py

Unified metrics & plotting for:
- Baseline
- Sudoku attack
- MDP attack

Generates a single merged CSV + multiple comparison plots.

Usage: python compute_metrics.py --base artifacts/results_baseline.csv --sudoku artifacts/results_sudoku.csv --mdp artifacts/results_mdp.csv --out artifacts/merged_all_attacks.csv --plots Plots  
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def make_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_bar_dual(df, col1, col2, labels, ylabel, title, outpath):
    x = np.arange(len(df))
    width = 0.35
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, df[col1], width, label=labels[0])
    plt.bar(x + width/2, df[col2], width, label=labels[1])
    plt.xticks(x, [f"Q{i+1}" for i in range(len(df))], rotation=45)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_line_two(df, col1, col2, labels, ylabel, title, outpath):
    x = np.arange(len(df))
    plt.figure(figsize=(12, 6))
    plt.plot(x, df[col1], marker="o", label=labels[0], linewidth=2)
    plt.plot(x, df[col2], marker="o", label=labels[1], linewidth=2)
    plt.xticks(x, [f"Q{i+1}" for i in range(len(df))], rotation=45)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_single_line(df, col, label, ylabel, title, outpath):
    x = np.arange(len(df))
    plt.figure(figsize=(12, 6))
    plt.plot(x, df[col], marker="o", label=label, linewidth=2)
    plt.xticks(x, [f"Q{i+1}" for i in range(len(df))], rotation=45)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="results_baseline.csv")
    parser.add_argument("--sudoku", default="results_sudoku.csv")
    parser.add_argument("--mdp", default="results_mdp.csv")
    parser.add_argument("--out", default="merged_all_attacks.csv")
    parser.add_argument("--plots", default="Plots")
    args = parser.parse_args()

    make_dir(args.plots)
    sudoku_dir = os.path.join(args.plots, "Sudoku")
    mdp_dir = os.path.join(args.plots, "MDP")
    combined_dir = os.path.join(args.plots, "Combined")

    make_dir(sudoku_dir)
    make_dir(mdp_dir)
    make_dir(combined_dir)

    # -------------------------
    # Load CSVs
    # -------------------------
    df_base = pd.read_csv(args.base)
    df_sudoku = pd.read_csv(args.sudoku)
    df_mdp = pd.read_csv(args.mdp)

    df_base["id"] = range(len(df_base))
    df_sudoku["id"] = range(len(df_sudoku))
    df_mdp["id"] = range(len(df_mdp))

    df = df_base.merge(df_sudoku, on="id", suffixes=("_base", "_sudoku"))
    df = df.merge(df_mdp, on="id")

    df.rename(columns={
        "reasoning_tokens": "reasoning_tokens_mdp",
        "cosine_similarity": "cosine_similarity_mdp"
    }, inplace=True)

    # -------------------------
    # Compute metrics
    # -------------------------
    df["sudoku_overhead"] = df["reasoning_tokens_sudoku"] - df["reasoning_tokens_base"]
    df["mdp_overhead"] = df["reasoning_tokens_mdp"] - df["reasoning_tokens_base"]

    df["sudoku_slowdown"] = df["reasoning_tokens_sudoku"] / df["reasoning_tokens_base"]
    df["mdp_slowdown"] = df["reasoning_tokens_mdp"] / df["reasoning_tokens_base"]

    df.to_csv(args.out, index=False)
    print(f"[OK] Saved merged CSV to {args.out}")

    # -------------------------
    # GRAPHS — SUDOKU FOLDER
    # -------------------------

    # 1 — Baseline vs Sudoku Token Usage
    plot_line_two(
        df,
        "reasoning_tokens_base",
        "reasoning_tokens_sudoku",
        ["Baseline", "Sudoku Attack"],
        "Tokens",
        "Baseline vs Sudoku — Token Usage",
        f"{sudoku_dir}/baseline_vs_attack_tokens_line.png"
    )

    # 2 — Sudoku Overhead
    plot_single_line(
        df,
        "sudoku_overhead",
        "Sudoku Overhead",
        "Extra Tokens",
        "Sudoku Token Overhead",
        f"{sudoku_dir}/token_overhead.png"
    )

    # 3 — Sudoku Slowdown
    plot_single_line(
        df,
        "sudoku_slowdown",
        "Sudoku Slowdown",
        "Slowdown Ratio",
        "Sudoku Slowdown Ratio",
        f"{sudoku_dir}/slowdown_ratio.png"
    )

    # 4 — Sudoku Token Comparison (bar)
    plot_bar_dual(
        df,
        "reasoning_tokens_base",
        "reasoning_tokens_sudoku",
        ["Baseline", "Sudoku Attack"],
        "Tokens",
        "Token Comparison (Sudoku)",
        f"{sudoku_dir}/token_usage_comparison.png"
    )

    # -------------------------
    # GRAPHS — MDP FOLDER
    # -------------------------

    plot_line_two(
        df,
        "reasoning_tokens_base",
        "reasoning_tokens_mdp",
        ["Baseline", "MDP Attack"],
        "Tokens",
        "Baseline vs MDP — Token Usage",
        f"{mdp_dir}/baseline_vs_attack_tokens_line.png"
    )

    plot_single_line(
        df,
        "mdp_overhead",
        "MDP Overhead",
        "Extra Tokens",
        "MDP Token Overhead",
        f"{mdp_dir}/token_overhead.png"
    )

    plot_single_line(
        df,
        "mdp_slowdown",
        "MDP Slowdown",
        "Slowdown Ratio",
        "MDP Slowdown Ratio",
        f"{mdp_dir}/slowdown_ratio.png"
    )

    plot_bar_dual(
        df,
        "reasoning_tokens_base",
        "reasoning_tokens_mdp",
        ["Baseline", "MDP Attack"],
        "Tokens",
        "Token Comparison (MDP)",
        f"{mdp_dir}/token_usage_comparison.png"
    )

    # -------------------------
    # GRAPHS — COMBINED FOLDER
    # -------------------------

    # Baseline vs Sudoku and MDP
    plot_line_two(
        df,
        "reasoning_tokens_base",
        "reasoning_tokens_sudoku",
        ["Baseline", "Sudoku Attack"],
        "Tokens",
        "Baseline vs Sudoku Attack",
        f"{combined_dir}/baseline_vs_sudoku_line.png"
    )

    plot_line_two(
        df,
        "reasoning_tokens_base",
        "reasoning_tokens_mdp",
        ["Baseline", "MDP Attack"],
        "Tokens",
        "Baseline vs MDP Attack",
        f"{combined_dir}/baseline_vs_mdp_line.png"
    )

    # Sudoku vs MDP — Overhead
    plot_bar_dual(
        df,
        "sudoku_overhead",
        "mdp_overhead",
        ["Sudoku Overhead", "MDP Overhead"],
        "Extra Tokens",
        "Token Overhead — Sudoku vs MDP",
        f"{combined_dir}/token_overhead_sudoku_mdp.png"
    )

    # Sudoku vs MDP — Slowdown
    plot_bar_dual(
        df,
        "sudoku_slowdown",
        "mdp_slowdown",
        ["Sudoku Slowdown", "MDP Slowdown"],
        "Slowdown Ratio",
        "Slowdown — Sudoku vs MDP",
        f"{combined_dir}/slowdown_sudoku_mdp.png"
    )

    print(f"\n[OK] All plots saved under: {args.plots}/")


if __name__ == "__main__":
    main()
