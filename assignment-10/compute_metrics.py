#!/usr/bin/env python3
"""
compute_metrics_full.py

Unified metrics & plotting for:
- Baseline
- Sudoku attack
- MDP attack

Generates a single merged CSV + structured plots.

Usage:
    python compute_metrics_full.py --base artifacts/results_baseline.csv --sudoku artifacts/results_sudoku.csv --mdp artifacts/results_mdp.csv --out artifacts/merged_all_attacks.csv --plots Plots
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------------------------
# Plot colors (consistent everywhere)
# ---------------------------------------------
COLOR_BASE = "#2ca02c"       # Green
COLOR_MDP = "#B60808"        # Red
COLOR_SUDOKU = "#c6d30b"     # Yellow


def make_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_bar_dual(df, col1, col2, labels, colors, ylabel, title, outpath):
    x = np.arange(len(df))
    width = 0.35
    plt.figure(figsize=(12, 6))

    plt.bar(x - width/2, df[col1], width, label=labels[0], color=colors[0])
    plt.bar(x + width/2, df[col2], width, label=labels[1], color=colors[1])

    plt.xticks(x, [f"Q{i+1}" for i in range(len(df))], rotation=45)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_line_two(df, col1, col2, labels, colors, ylabel, title, outpath):
    x = np.arange(len(df))
    plt.figure(figsize=(12, 6))

    plt.plot(x, df[col1], marker="o", linewidth=2, label=labels[0], color=colors[0])
    plt.plot(x, df[col2], marker="o", linewidth=2, label=labels[1], color=colors[1])

    plt.xticks(x, [f"Q{i+1}" for i in range(len(df))], rotation=45)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_slowdown_per_prompt(df, slowdown_col, attack_label, attack_color, outpath):
    x = np.arange(len(df))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, np.ones_like(df[slowdown_col]), width,
            label="Baseline (1.0)", color=COLOR_BASE)
    plt.bar(x + width/2, df[slowdown_col], width,
            label=f"{attack_label} Slowdown", color=attack_color)

    plt.xticks(x, [f"Q{i+1}" for i in range(len(df))], rotation=45)
    plt.ylabel("Slowdown Ratio (attack/base)")
    plt.title(f"Per-Prompt Slowdown — {attack_label} Attack")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="artifacts/results_baseline.csv")
    parser.add_argument("--sudoku", default="artifacts/results_sudoku.csv")
    parser.add_argument("--mdp", default="artifacts/results_mdp.csv")
    parser.add_argument("--out", default="artifacts/merged_all_attacks.csv")
    parser.add_argument("--plots", default="Plots")
    args = parser.parse_args()

    make_dir(args.plots)
    sudoku_dir = os.path.join(args.plots, "Sudoku")
    mdp_dir = os.path.join(args.plots, "MDP")
    combined_dir = os.path.join(args.plots, "Combined")

    make_dir(sudoku_dir)
    make_dir(mdp_dir)
    make_dir(combined_dir)

    # ---------------------------------------------
    # Load CSVs
    # ---------------------------------------------
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

    # ---------------------------------------------
    # DROP PROMPT / RAW TEXT / ANSWER FIELDS
    # ---------------------------------------------
    cols_to_drop = [
        c for c in df.columns if
        "question" in c.lower() or
        "model_answer" in c.lower() or
        "raw_generated_text" in c.lower() or
        "reasoning_tag" in c.lower() or
        "ground_truth" in c.lower()
    ]
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    # ---------------------------------------------
    # Compute metrics
    # ---------------------------------------------
    df["sudoku_overhead"] = df["reasoning_tokens_sudoku"] - df["reasoning_tokens_base"]
    df["mdp_overhead"] = df["reasoning_tokens_mdp"] - df["reasoning_tokens_base"]

    df["sudoku_slowdown"] = df["reasoning_tokens_sudoku"] / df["reasoning_tokens_base"]
    df["mdp_slowdown"] = df["reasoning_tokens_mdp"] / df["reasoning_tokens_base"]

    df.to_csv(args.out, index=False)
    print(f"[OK] Saved merged CSV to {args.out}")

    # ---------------------------------------------
    # SUDOKU PLOTS
    # ---------------------------------------------
    plot_line_two(
        df,
        "reasoning_tokens_base",
        "reasoning_tokens_sudoku",
        ["Baseline", "Sudoku Attack"],
        [COLOR_BASE, COLOR_SUDOKU],
        "Tokens",
        "Baseline vs Sudoku — Token Usage",
        f"{sudoku_dir}/baseline_vs_attack_tokens_line.png"
    )

    plot_bar_dual(
        df,
        "reasoning_tokens_base",
        "reasoning_tokens_sudoku",
        ["Baseline", "Sudoku Attack"],
        [COLOR_BASE, COLOR_SUDOKU],
        "Tokens",
        "Token Comparison (Sudoku)",
        f"{sudoku_dir}/token_usage_comparison.png"
    )

    plot_slowdown_per_prompt(
        df,
        "sudoku_slowdown",
        "Sudoku",
        COLOR_SUDOKU,
        f"{sudoku_dir}/slowdown_per_prompt.png"
    )

    plot_line_two(
        df,
        "sudoku_overhead",
        "sudoku_overhead",
        ["Sudoku Overhead", "Sudoku Overhead"],
        [COLOR_SUDOKU, COLOR_SUDOKU],
        "Extra Tokens",
        "Sudoku Token Overhead",
        f"{sudoku_dir}/token_overhead.png"
    )

    # ---------------------------------------------
    # MDP PLOTS
    # ---------------------------------------------
    plot_line_two(
        df,
        "reasoning_tokens_base",
        "reasoning_tokens_mdp",
        ["Baseline", "MDP Attack"],
        [COLOR_BASE, COLOR_MDP],
        "Tokens",
        "Baseline vs MDP — Token Usage",
        f"{mdp_dir}/baseline_vs_attack_tokens_line.png"
    )

    plot_bar_dual(
        df,
        "reasoning_tokens_base",
        "reasoning_tokens_mdp",
        ["Baseline", "MDP Attack"],
        [COLOR_BASE, COLOR_MDP],
        "Tokens",
        "Token Comparison (MDP)",
        f"{mdp_dir}/token_usage_comparison.png"
    )

    plot_slowdown_per_prompt(
        df,
        "mdp_slowdown",
        "MDP",
        COLOR_MDP,
        f"{mdp_dir}/slowdown_per_prompt.png"
    )

    plot_line_two(
        df,
        "mdp_overhead",
        "mdp_overhead",
        ["MDP Overhead", "MDP Overhead"],
        [COLOR_MDP, COLOR_MDP],
        "Extra Tokens",
        "MDP Token Overhead",
        f"{mdp_dir}/token_overhead.png"
    )

    # ---------------------------------------------
    # COMBINED PLOTS
    # ---------------------------------------------
    plot_line_two(
        df,
        "reasoning_tokens_sudoku",
        "reasoning_tokens_mdp",
        ["Sudoku Attack", "MDP Attack"],
        [COLOR_SUDOKU, COLOR_MDP],
        "Tokens",
        "Sudoku vs MDP — Token Usage",
        f"{combined_dir}/sudoku_vs_mdp_tokens.png"
    )

    plot_bar_dual(
        df,
        "sudoku_overhead",
        "mdp_overhead",
        ["Sudoku Overhead", "MDP Overhead"],
        [COLOR_SUDOKU, COLOR_MDP],
        "Extra Tokens",
        "Token Overhead — Sudoku vs MDP",
        f"{combined_dir}/token_overhead_sudoku_mdp.png"
    )

    # Per-prompt slowdown comparison Sudoku vs MDP
    x = np.arange(len(df))
    width = 0.35
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, df["sudoku_slowdown"], width, label="Sudoku Slowdown", color=COLOR_SUDOKU)
    plt.bar(x + width/2, df["mdp_slowdown"], width, label="MDP Slowdown", color=COLOR_MDP)

    plt.xticks(x, [f"Q{i+1}" for i in range(len(df))], rotation=45)
    plt.ylabel("Slowdown Ratio (attack/base)")
    plt.title("Sudoku vs MDP — Per-Prompt Slowdown Comparison")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{combined_dir}/slowdown_per_prompt_sudoku_vs_mdp.png")
    plt.close()

    print(f"\n[OK] All plots saved under: {args.plots}/")


if __name__ == "__main__":
    main()
