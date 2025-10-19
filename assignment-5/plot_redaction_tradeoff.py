#!/usr/bin/env python3
"""
plot_redaction_tradeoff.py
Generates Privacy–Utility tradeoff curves per scenario using attack and evaluation reports.

Reads:
 - attack_report.json → Privacy (privacy_retention_%)
 - evaluation_report.json → Utility (Utility_S)
Outputs:
 - Per-scenario and overall tradeoff plots across redaction strengths.
"""

import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import re


# --------------------------------------------------------------
# Helper: extract numeric redaction strength from folder names
# --------------------------------------------------------------
def extract_strength_from_path(path):
    """Extract numeric redaction value from a folder name like redaction_0.3 or redaction_2."""
    name = os.path.basename(path)
    m = re.search(r"([0-9]*\.?[0-9]+)", name)
    if m:
        try:
            return float(m.group(1))
        except:
            return None
    return None


# --------------------------------------------------------------
# Data loader
# --------------------------------------------------------------
def load_privacy_utility_data(parent_dir):
    """Load privacy and utility values from attack/evaluation reports in nested run folders."""
    rows = []
    for root, dirs, files in os.walk(parent_dir):
        if not ("attack_report.json" in files or "evaluation_report.json" in files):
            continue

        attack_path = os.path.join(root, "attack_report.json")
        eval_path = os.path.join(root, "evaluation_report.json")

        privacy = None
        utility = None

        # Read privacy metrics from attack_report
        if os.path.exists(attack_path):
            try:
                with open(attack_path, "r", encoding="utf-8") as f:
                    att = json.load(f)
                privacy = att.get("privacy_retention_%") or att.get("Privacy_S")
            except Exception as e:
                print(f"[Skip privacy] {attack_path}: {e}")

        # Read utility metrics from evaluation_report
        if os.path.exists(eval_path):
            try:
                with open(eval_path, "r", encoding="utf-8") as f:
                    ev = json.load(f)
                utility = ev.get("Utility_S")
            except Exception as e:
                print(f"[Skip utility] {eval_path}: {e}")

        # Skip if neither metric found
        if privacy is None and utility is None:
            continue

        scenario = os.path.basename(os.path.dirname(root))
        redaction = os.path.basename(root)
        strength = extract_strength_from_path(root)

        rows.append({
            "scenario": scenario,
            "redaction": redaction,
            "redaction_strength": strength,
            "Privacy_S": privacy,
            "Utility_S": utility
        })
        print(f"[Loaded] {scenario}/{redaction}: Privacy={privacy}, Utility={utility}")

    return pd.DataFrame(rows)


# --------------------------------------------------------------
# Plot generation
# --------------------------------------------------------------
def plot_tradeoff_curves(df, outdir="plots_tradeoff"):
    """Generate Privacy–Utility tradeoff plots per scenario and combined overview."""
    os.makedirs(outdir, exist_ok=True)

    scenarios = sorted(df["scenario"].unique())
    for sc in scenarios:
        sub = df[df["scenario"] == sc].dropna(subset=["Privacy_S", "Utility_S"])
        if sub.empty:
            continue

        # Sort by redaction strength (if available)
        if "redaction_strength" in sub and sub["redaction_strength"].notna().any():
            sub = sub.sort_values("redaction_strength")
        else:
            sub = sub.sort_values("Privacy_S")

        # Per-scenario curve
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(sub["Privacy_S"], sub["Utility_S"], marker="o", linewidth=2, color="tab:blue")

        # Label each point with its redaction folder
        for _, r in sub.iterrows():
            label = r["redaction"]
            if pd.notnull(r.get("redaction_strength")):
                label += f" (r={r['redaction_strength']})"
            ax.annotate(label, (r["Privacy_S"], r["Utility_S"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=8)

        ax.set_xlabel("Privacy (%)")
        ax.set_ylabel("Utility (%)")
        ax.set_title(f"{sc} — Privacy–Utility Tradeoff")
        ax.grid(alpha=0.3)
        plt.tight_layout()

        safe_name = sc.replace("/", "_").replace("\\", "_")
        outp = os.path.join(outdir, f"tradeoff_{safe_name}.png")
        plt.savefig(outp, dpi=160)
        plt.close(fig)
        print(f"[Saved] {outp}")

    # Combined overview plot
    fig, ax = plt.subplots(figsize=(7, 5))
    for sc in scenarios:
        sub = df[df["scenario"] == sc].dropna(subset=["Privacy_S", "Utility_S"])
        if sub.empty:
            continue
        sub = sub.sort_values("Privacy_S")
        ax.plot(sub["Privacy_S"], sub["Utility_S"], marker="o", label=sc)

    ax.set_xlabel("Privacy (%)")
    ax.set_ylabel("Utility (%)")
    ax.set_title("Privacy–Utility Tradeoff Across Scenarios")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    outp = os.path.join(outdir, "tradeoff_overall.png")
    plt.savefig(outp, dpi=160)
    plt.close(fig)
    print(f"[Saved] {outp}")


# --------------------------------------------------------------
# Main
# --------------------------------------------------------------
def main():
    """Main entry point: load reports and plot privacy–utility tradeoff curves."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Path to AirGap run folder")
    ap.add_argument("--outdir", default="plots_compare/redaction_plots", help="Output directory for plots")
    args = ap.parse_args()

    df = load_privacy_utility_data(args.run)
    if df.empty:
        print("No attack/evaluation reports found.")
        return

    print("\n=== SUMMARY TABLE ===\n")
    print(df.to_string(index=False))

    plot_tradeoff_curves(df, args.outdir)


if __name__ == "__main__":
    main()
