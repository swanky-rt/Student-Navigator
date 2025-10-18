#!/usr/bin/env python3
"""
plot_leakrate_comparison.py

Compare leakage rates (% recovered tokens) across all scenario folders
and optionally between two runs (Baseline vs AirGap).

• Reads attack_report.json under each subfolder:
      runs/<variant>/<scenario>/redaction_x/attack_report.json
• Extracts: total_sensitive_tokens, total_recovered_tokens
• Computes: leakage_rate_% = 100 * total_recovered_tokens / total_sensitive_tokens
• Plots per-scenario bar chart (single run or comparative two-run mode).
"""

import os, json, argparse
import pandas as pd
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────
def load_leakage_from_run(parent_dir, tag="run"):
    """Recursively load attack_report.json files and compute leakage rate."""
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
            if total > 0:
                leak_rate = 100.0 * recovered / total
            else:
                leak_rate = 0.0
            scenario = os.path.basename(os.path.dirname(root))  # e.g. internal_hr
            rows.append({
                "scenario": scenario,
                "variant": tag,
                "leakage_rate_%": leak_rate
            })
            print(f"[Loaded] {scenario:25} → {leak_rate:6.2f}% leakage ({recovered}/{total})")
        except Exception as e:
            print(f"[Skip] {path}: {e}")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────
def plot_leakage(df, outdir="plots_leakage", label1="Baseline", label2="AirGap"):
    os.makedirs(outdir, exist_ok=True)

    scenarios = sorted(df["scenario"].unique())
    fig, ax = plt.subplots(figsize=(8, 5))

    if "variant" in df.columns and df["variant"].nunique() > 1:
        # Comparative mode (Baseline vs AirGap)
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
        # Single run
        vals = [df[df["scenario"] == sc]["leakage_rate_%"].mean() for sc in scenarios]
        ax.bar(scenarios, vals, color="tab:blue", alpha=0.8)
        ax.set_xticks(range(len(scenarios)))

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


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run1", required=True, help="Path to first run folder (Baseline)")
    ap.add_argument("--run2", required=False, help="Path to second run folder (AirGap)")
    ap.add_argument("--label1", default="Baseline")
    ap.add_argument("--label2", default="AirGap")
    ap.add_argument("--outdir", default="plots_compare")
    args = ap.parse_args()

    df1 = load_leakage_from_run(args.run1, tag=args.label1)
    if args.run2:
        df2 = load_leakage_from_run(args.run2, tag=args.label2)
        df = pd.concat([df1, df2], ignore_index=True)
    else:
        df = df1

    print("\n=== LEAKAGE SUMMARY ===\n")
    print(df.to_string(index=False))

    plot_leakage(df, args.outdir, args.label1, args.label2)


if __name__ == "__main__":
    main()
