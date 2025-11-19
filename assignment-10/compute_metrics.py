#!/usr/bin/env python3
"""
compute_metrics.py (Final Version)

Features:
 - Skips running model if CSV results already exist
 - Merges baseline + attacked results
 - Skips IDs 1 and 7
 - Line graph for baseline vs attacked token usage
 - Clean Q-labels
 - Saves all plots into Plots/
"""

import argparse
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time


def run_script(cmd):
    print(f"\n[RUNNING] {cmd}\n")
    start = time.time()
    subprocess.run(cmd, shell=True, check=True)
    end = time.time()
    return end - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="job_reasoning_questions.csv")
    parser.add_argument("--base", default="results_baseline.csv")
    parser.add_argument("--attack", default="results_overthink.csv")
    parser.add_argument("--plots", default="Plots")
    args = parser.parse_args()

    os.makedirs(args.plots, exist_ok=True)

    # -----------------------
    # 1. Run Baseline Only If Missing
    # -----------------------
    if os.path.exists(args.base):
        print(f"[SKIP] Baseline exists: {args.base}")
        time_base = None
    else:
        baseline_cmd = (
            f"python job_reasoning_eval.py "
            f"--csv {args.csv} "
            f"--out {args.base}"
        )
        time_base = run_script(baseline_cmd)

    # -----------------------
    # 2. Run Attack Only If Missing
    # -----------------------
    if os.path.exists(args.attack):
        print(f"[SKIP] Attack exists: {args.attack}")
        time_attack = None
    else:
        attack_cmd = (
            f"python job_reasoning_eval.py "
            f"--csv {args.csv} "
            f"--out {args.attack} "
            f"--attack --attack-variant sudoku"
        )
        time_attack = run_script(attack_cmd)

    # -----------------------
    # 3. Load CSVs
    # -----------------------
    df_base = pd.read_csv(args.base)
    df_attack = pd.read_csv(args.attack)

    # Align indexing
    df_base["id"] = range(len(df_base))
    df_attack["id"] = range(len(df_attack))

    # Merge
    df = pd.merge(df_base, df_attack, on="id", suffixes=("_base", "_attack"))

    # -----------------------
    # 4. Filter Out IDs 1 and 7
    # -----------------------
    df = df[~df["id"].isin([1, 7])].reset_index(drop=True)

    # -----------------------
    # 5. Compute Metrics
    # -----------------------
    df["token_overhead"] = df["reasoning_tokens_attack"] - df["reasoning_tokens_base"]
    df["slowdown_ratio"] = (
        df["reasoning_tokens_attack"] / df["reasoning_tokens_base"]
    ).replace(np.inf, np.nan)
    df["sim_drop"] = df["cosine_similarity_base"] - df["cosine_similarity_attack"]

    # -----------------------
    # 6. Save Merged CSV
    # -----------------------
    df.to_csv("merged_results.csv", index=False)
    print("[OK] Saved merged_results.csv")

    # -----------------------
    # 7. Save Summary Metrics
    # -----------------------
    with open("metrics_summary.txt", "w") as f:
        f.write(f"avg_token_overhead: {df['token_overhead'].mean()}\n")
        f.write(f"max_token_overhead: {df['token_overhead'].max()}\n")
        f.write(f"avg_slowdown_ratio: {df['slowdown_ratio'].mean()}\n")
        f.write(f"avg_similarity_drop: {df['sim_drop'].mean()}\n")

    print("[OK] Saved metrics_summary.txt")

    # -----------------------
    # 8. Plotting
    # -----------------------

    ids = df["id"].tolist()
    x = np.arange(len(ids))
    qlabels = [f"Q{i+1}" for i in range(len(df))]
    width = 0.35

    # ---- PLOT 1: TOKEN OVERHEAD ----
    plt.figure(figsize=(10, 6))
    plt.bar(x, df["token_overhead"], color="purple")
    plt.xticks(x, qlabels, rotation=45)
    plt.ylabel("Token Overhead")
    plt.title("Token Overhead (Attack - Baseline)")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{args.plots}/token_overhead.png")
    plt.close()

    # ---- PLOT 2: SLOWDOWN RATIO ----
    plt.figure(figsize=(10, 6))
    plt.plot(x, df["slowdown_ratio"], marker="o", linewidth=2)
    plt.xticks(x, qlabels, rotation=45)
    plt.ylabel("Slowdown Ratio (attack/base)")
    plt.title("Slowdown Ratio Per Question")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{args.plots}/slowdown_ratio.png")
    plt.close()

    # ---- PLOT 3: COSINE SIMILARITY DROP ----
    plt.figure(figsize=(10, 6))
    plt.bar(x, df["sim_drop"], color="darkred")
    plt.xticks(x, qlabels, rotation=45)
    plt.ylabel("Similarity Drop")
    plt.title("Cosine Similarity Drop (Baseline - Attacked)")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{args.plots}/similarity_drop.png")
    plt.close()

    # ---- PLOT 4: LINE GRAPH — BASELINE vs ATTACKED TOKENS ----
    plt.figure(figsize=(10, 6))
    plt.plot(x, df["reasoning_tokens_base"], marker="o", label="Baseline", linewidth=2)
    plt.plot(x, df["reasoning_tokens_attack"], marker="o", label="Attacked", linewidth=2)

    plt.xticks(x, qlabels, rotation=45)
    plt.ylabel("Reasoning Tokens")
    plt.title("Baseline vs Attacked Token Usage")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.plots}/baseline_vs_attack_tokens_line.png")
    plt.close()

    print(f"\n[OK] All plots saved to: {args.plots}/")


if __name__ == "__main__":
    main()
