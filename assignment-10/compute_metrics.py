#!/usr/bin/env python3
"""
compute_metrics.py

Runs baseline + attacked evaluations using job_reasoning_eval_sudoku.py
and generates required metrics & plots for Assignment 10.

Outputs:
 - metrics_summary.txt
 - merged_results.csv
 - Plots/*.png
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
    # 1. RUN BASELINE
    # -----------------------
    baseline_cmd = (
        f"python job_reasoning_eval.py "
        f"--csv {args.csv} "
        f"--out {args.base}"
    )
    time_base = run_script(baseline_cmd)

    # -----------------------
    # 2. RUN ATTACKED (Sudoku)
    # -----------------------
    attack_cmd = (
        f"python job_reasoning_eval.py "
        f"--csv {args.csv} "
        f"--out {args.attack} "
        f"--attack --attack-variant sudoku"
    )
    time_attack = run_script(attack_cmd)

    # -----------------------
    # 3. LOAD CSVs
    # -----------------------
    df_base = pd.read_csv(args.base)
    df_attack = pd.read_csv(args.attack)

    # Ensure alignment
    df_base["id"] = range(len(df_base))
    df_attack["id"] = range(len(df_attack))

    df = pd.merge(df_base, df_attack, on="id", suffixes=("_base", "_attack"))

    # -----------------------
    # 4. METRICS
    # -----------------------

    # Time ratio
    slowdown = time_attack / time_base

    # Token overhead
    df["token_overhead"] = df["reasoning_tokens_attack"] - df["reasoning_tokens_base"]

    # Slowdown ratio per item
    df["slowdown_ratio"] = (
        df["reasoning_tokens_attack"] / df["reasoning_tokens_base"]
    ).replace(np.inf, np.nan)

    # Cosine similarity drop
    df["sim_drop"] = df["cosine_similarity_base"] - df["cosine_similarity_attack"]

    # Aggregates
    metrics = {
        "time_baseline_sec": time_base,
        "time_attacked_sec": time_attack,
        "slowdown_S = t_attacked / t_base": slowdown,
        "avg_token_overhead": df["token_overhead"].mean(),
        "max_token_overhead": df["token_overhead"].max(),
        "avg_slowdown_ratio (tokens)": df["slowdown_ratio"].mean(),
        "avg_similarity_drop": df["sim_drop"].mean(),
    }

    # -----------------------
    # 5. SAVE MERGED CSV
    # -----------------------
    df.to_csv("merged_results.csv", index=False)
    print("\n[OK] Saved merged_results.csv")

    # -----------------------
    # 6. SAVE METRICS TEXT
    # -----------------------
    with open("metrics_summary.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    print("[OK] Saved metrics_summary.txt")
    print("\n=== SUMMARY ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # -----------------------
    # 7. PLOTS
    # -----------------------

    # Plot 1: Token Overhead
    plt.figure(figsize=(6, 4))
    plt.bar(df["id"], df["token_overhead"])
    plt.title("Token Overhead (Attack - Baseline)")
    plt.xlabel("Question ID")
    plt.ylabel("Token Overhead")
    plt.tight_layout()
    plt.savefig(f"{args.plots}/token_overhead.png")
    plt.close()

    # Plot 2: Slowdown ratio
    plt.figure(figsize=(6, 4))
    plt.plot(df["id"], df["slowdown_ratio"], marker="o")
    plt.title("Slowdown Ratio (Tokens)")
    plt.xlabel("Question ID")
    plt.ylabel("Slowdown Ratio")
    plt.tight_layout()
    plt.savefig(f"{args.plots}/slowdown_ratio.png")
    plt.close()

    # Plot 3: Cosine similarity drop
    plt.figure(figsize=(6, 4))
    plt.bar(df["id"], df["sim_drop"])
    plt.title("Cosine Similarity Drop (Baseline - Attacked)")
    plt.xlabel("Question ID")
    plt.ylabel("Similarity Drop")
    plt.tight_layout()
    plt.savefig(f"{args.plots}/similarity_drop.png")
    plt.close()

    # Plot 4: Scatter baseline vs attack tokens
    plt.figure(figsize=(6, 5))
    plt.scatter(
        df["reasoning_tokens_base"],
        df["reasoning_tokens_attack"],
        c="blue",
    )
    plt.title("Baseline vs Attacked Token Usage")
    plt.xlabel("Baseline Tokens")
    plt.ylabel("Attacked Tokens")
    plt.tight_layout()
    plt.savefig(f"{args.plots}/baseline_vs_attack_tokens.png")
    plt.close()

    print("[OK] All plots saved in Plots/")


if __name__ == "__main__":
    main()
