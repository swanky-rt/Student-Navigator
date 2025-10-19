#!/usr/bin/env python3
"""
plot_leakrate_comparison.py
Plots token-level leakage rates from attack reports for each scenario.
Supports single-run or comparative two-run (e.g., Baseline vs AirGap) mode.
"""

import os, json, argparse
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------------------
# Data loading utilities
# --------------------------------------------------------------
def load_leakage_from_run(parent_dir, tag="run"):
    """Traverse a run directory and extract leakage rates from attack_report.json files."""
    rows = []
    for root, dirs, files in os.walk(parent_dir):
        if "attack_report.json" not in files:
            continue
        path = os.path.join(root, "attack_report.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            total = data.get("total_sensitive_tokens", 0)
            recovered = data.get("total_recovered_tokens", 0)
            # Compute leakage rate percentage
            leak_rate = 100.0 * recovered / total if total > 0 else 0.0
            scenario = os.path.basename(os.path.dirname(root))
            rows.append({
                "scenario": scenario,
                "variant": tag,
                "leakage_rate_%": leak_rate
            })
            print(f"[Loaded] {scenario:25} → {leak_rate:6.2f}% leakage ({recovered}/{total})")
        except Exception as e:
            print(f"[Skip] {path}: {e}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------
# Plotting
# --------------------------------------------------------------
def plot_leakage(df, outdir="plots_leakage", label1="Baseline", label2="AirGap"):
    """Plot bar charts of leakage rates per scenario; handles single or dual-run comparison."""
    os.makedirs(outdir, exist_ok=True)

    scenarios = sorted(df["scenario"].unique())
    fig, ax = plt.subplots(figsize=(8, 5))

    # Comparative bar chart when two variants are provided
    if "variant" in df.columns and df["variant"].nunique() > 1:
        variants = df["variant"].unique()
        width = 0.35
        x = range(len(scenarios))
        for i, var in enumerate(variants):
            sub = df[df["variant"] == var]
            vals = [sub[sub["scenario"] == sc]["leakage_rate_%"].mean() if sc in sub["scenario"].values else 0.0
                    for sc in scenarios]
            ax.bar([xi + i * width for xi in x], vals, width=width, label=var)
        ax.set_xticks([r + width / 2 for r in x])
    else:
        # Single-run plot
        vals = [df[df["scenario"] == sc]["leakage_rate_%"].mean() for sc in scenarios]
        ax.bar(scenarios, vals, color="tab:blue", alpha=0.8)
        ax.set_xticks(range(len(scenarios)))

    # Basic chart formatting
    ax.set_xticklabels(scenarios, rotation=30, ha="right")
    ax.set_ylabel("Leakage Rate (%)")
    ax.set_xlabel("Scenario")
    ax.set_title("Token-Level Leakage Rate by Scenario")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    plt.tight_layout()
    outp = os.path.join(outdir, "leakage_rate_comparison.png")
    plt.savefig(outp, dpi=160)
    plt.close(fig)
    print(f"[Saved] {outp}")


# --------------------------------------------------------------
# Main
# --------------------------------------------------------------
def main():
    """Main entry: loads results from one or two runs and plots leakage comparison."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run1", required=True, help="Path to first run folder (Baseline)")
    ap.add_argument("--run2", required=False, help="Path to second run folder (AirGap)")
    ap.add_argument("--label1", default="Baseline")
    ap.add_argument("--label2", default="AirGap")
    ap.add_argument("--outdir", default="plots_compare")
    args = ap.parse_args()

    # Load run(s)
    df1 = load_leakage_from_run(args.run1, tag=args.label1)
    if args.run2:
        df2 = load_leakage_from_run(args.run2, tag=args.label2)
        df = pd.concat([df1, df2], ignore_index=True)
    else:
        df = df1

    print("\n=== LEAKAGE SUMMARY ===\n")
    print(df.to_string(index=False))

    # Generate plot
    plot_leakage(df, args.outdir, args.label1, args.label2)


if __name__ == "__main__":
    main()
