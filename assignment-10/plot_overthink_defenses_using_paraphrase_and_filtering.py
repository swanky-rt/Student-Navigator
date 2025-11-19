#!/usr/bin/env python3
"""
plot_all_token_metrics.py

Generates a bar chart comparing reasoning token usage across all three conditions:
  1. Baseline (No attack)
  2. Overthink (Attack)
  3. Filtering Defense

Usage:
  # Requires results_baseline.csv, results_overthink.csv, and results_filtering.csv
  python plot_all_token_metrics.py
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def load_tokens(path: str, label: str) -> np.ndarray | None:
    """
    Loads reasoning tokens from a CSV file.
    Tries to find 'reasoning_tokens' or 'total_reasoning_tokens'.
    """
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        return None

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[ERROR] Could not read {path}: {e}")
        return None

    col = None
    for candidate in ["reasoning_tokens", "total_reasoning_tokens"]:
        if candidate in df.columns:
            col = candidate
            break

    if col is None:
        print(f"[ERROR] Could not find reasoning token column in {path}. Columns found: {df.columns}")
        return None

    # Drop NaNs and ensure numeric
    vals = pd.to_numeric(df[col], errors='coerce').dropna().values
    print(f"[INFO] {label}: n={len(vals)}, mean={vals.mean():.2f}, std={vals.std():.2f}")
    return vals


def main():
    parser = argparse.ArgumentParser()
    # Default file paths assuming standard evaluation script outputs
    parser.add_argument("--baseline", default="results_baseline.csv", help="Path to baseline results CSV")
    parser.add_argument("--attack", default="results_overthink.csv", help="Path to attack results CSV")
    parser.add_argument("--filtering", default="results_filtering.csv", help="Path to filtering defense results CSV")
    parser.add_argument("--out", default="defense_token_comparison.png", help="Output image filename")
    args = parser.parse_args()

    # 1. Load Data
    baseline_vals = load_tokens(args.baseline, "Baseline")
    attack_vals = load_tokens(args.attack, "Overthink (Attack)")
    filtering_vals = load_tokens(args.filtering, "Filtering Defense")

    data = []
    labels = []

    if baseline_vals is not None:
        data.append({"mean": baseline_vals.mean(), "std": baseline_vals.std()})
        labels.append("Baseline")
    if attack_vals is not None:
        data.append({"mean": attack_vals.mean(), "std": attack_vals.std()})
        labels.append("Overthink (Attack)")
    if filtering_vals is not None:
        data.append({"mean": filtering_vals.mean(), "std": filtering_vals.std()})
        labels.append("Filtering Defense")

    if not data:
        print("Aborting plot generation due to missing data.")
        return

    # 2. Prepare Data for Plotting
    means = [d["mean"] for d in data]
    stds = [d["std"] for d in data]
    x = np.arange(len(labels))

    # 3. Create Plot
    plt.figure(figsize=(9, 6))

    # Define colors: Blue for Baseline, Orange for Attack, Purple for Defense
    colors = ['#1f77b4', '#ff7f0e', '#9467bd']

    # Create bars with error bars for standard deviation
    bars = plt.bar(x, means, yerr=stds, capsize=10,
                   color=colors[:len(labels)],
                   alpha=0.9, width=0.7)

    # Formatting
    plt.xticks(x, labels, fontsize=12)
    plt.ylabel("Reasoning Tokens (Count)", fontsize=13)
    plt.title("Reasoning Token Usage: Baseline vs. Attack vs. Defense", fontsize=15, pad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Add text labels on top of bars
    max_mean = max(means) if means else 1
    for bar in bars:
        height = bar.get_height()
        # Place label slightly above the bar or the error bar
        plt.text(bar.get_x() + bar.get_width() / 2., height + (max_mean * 0.05),
                 f'{int(round(height))}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 4. Save
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"\n[SUCCESS] Plot saved to {args.out}")


if __name__ == "__main__":
    # Ensure you have the three CSV files in the same directory before running:
    # results_baseline.csv
    # results_overthink.csv
    # results_filtering.csv
    main()