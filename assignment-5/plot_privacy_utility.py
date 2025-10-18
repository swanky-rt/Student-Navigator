#!/usr/bin/env python3
"""
plot_privacy_utility.py
Compare multiple AirGap / baseline runs and generate privacy–utility plots.

Features:
- Automatically scans nested subfolders for attack_report.json & evaluation_report.json
- Uses privacy_retention_% from attack_report.json as Privacy
- Uses Utility_S from evaluation_report.json as Utility
- Cleans up plot titles (no "redaction_x" shown)
- Works safely across Windows/Linux paths
"""

import argparse, json, os, glob
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────

def load_combined_reports(parent_dir, tag="run"):
    """
    Combine attack_report.json (for privacy) and evaluation_report.json (for utility)
    under each scenario/redaction folder.
    Example structure:
        parent_dir/scenario/redaction_x/attack_report.json
        parent_dir/scenario/redaction_x/evaluation_report.json
    """
    dfs = []
    for root, dirs, files in os.walk(parent_dir):
        if "attack_report.json" in files or "evaluation_report.json" in files:
            attack_path = os.path.join(root, "attack_report.json")
            eval_path = os.path.join(root, "evaluation_report.json")

            privacy, utility = None, None
            attack_s, leak_count = None, None

            # --- read attack report ---
            if os.path.exists(attack_path):
                with open(attack_path, "r", encoding="utf-8") as f:
                    att = json.load(f)
                privacy = att.get("privacy_retention_%") or att.get("Privacy_S")
                attack_s = att.get("attack_success_%") or att.get("Attack_S")
                leak_count = att.get("leak_count")

            # --- read evaluation report ---
            if os.path.exists(eval_path):
                with open(eval_path, "r", encoding="utf-8") as f:
                    ev = json.load(f)
                utility = ev.get("Utility_S")

            scenario = os.path.basename(os.path.dirname(root))
            redaction = os.path.basename(root)
            dfs.append({
                "scenario": f"{scenario}/{redaction}",
                "Privacy_S": privacy,
                "Utility_S": utility,
                "Attack_S": attack_s,
                "leak_count": leak_count,
                "_source": tag
            })
            print(f"[Loaded] {scenario}/{redaction}")
    if not dfs:
        raise FileNotFoundError(f"No reports found under {parent_dir}")
    return pd.DataFrame(dfs)


def load_attack_or_eval(path, tag="run"):
    """Handle single report files."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame([{
        "scenario": data.get("attacker_mode") or data.get("model_variant", tag),
        "Privacy_S": data.get("privacy_retention_%") or data.get("Privacy_S"),
        "Utility_S": data.get("Utility_S"),
        "Attack_S": data.get("Attack_S") or data.get("attack_success_%"),
        "leak_count": data.get("leak_count"),
        "_source": tag
    }])
    return df


# ─────────────────────────────────────────────────────────────
# Compare + plotting
# ─────────────────────────────────────────────────────────────
def compare_data(df1, df2):
    on = ["scenario"]
    merged = pd.merge(df1, df2, on=on, suffixes=("_1", "_2"), how="outer")

    for col in ["Privacy_S", "Utility_S", "Attack_S"]:
        if col + "_1" in merged and col + "_2" in merged:
            merged[col + "_Δ"] = merged[col + "_2"] - merged[col + "_1"]
    return merged


def plot_comparison(merged, outdir, label1="Run1", label2="Run2"):
    os.makedirs(outdir, exist_ok=True)

    # Clean scenario names (remove redaction details for titles)
    merged["scenario_clean"] = merged["scenario"].apply(
        lambda x: x.split("/")[0] if isinstance(x, str) and "/" in x else x
    )

    # --- 1. Privacy & Utility bar comparison
    for sc in sorted(merged["scenario"].dropna().unique()):
        sub = merged[merged["scenario"] == sc]
        if sub.empty:
            continue
        if sub["Privacy_S_1"].isna().all() and sub["Privacy_S_2"].isna().all():
            print(f"[Skip] {sc} (no valid privacy/utility scores)")
            continue

        # Use clean name for title
        title = sc.split("/")[0] if "/" in sc else sc
        metrics = ["Privacy_S", "Utility_S"]
        labels = [label1, label2]
        width = 0.35

        fig, ax = plt.subplots(figsize=(6, 4))
        x = range(len(metrics))
        for i, prefix in enumerate(["_1", "_2"]):
            vals = []
            for m in metrics:
                col = f"{m}{prefix}"
                val = sub[col].values[0] if col in sub and not sub[col].isna().all() else 0.0
                vals.append(val if isinstance(val, (int, float)) else 0.0)
            ax.bar([xi + i * width for xi in x], vals, width=width, label=labels[i], alpha=0.8)
        ax.set_xticks([r + width / 2 for r in x])
        ax.set_xticklabels(["Privacy", "Utility"])
        ax.set_ylabel("Score (%)")
        ax.set_title(f"{title} — Privacy & Utility")
        ax.legend()
        ax.grid(alpha=0.15, axis="y")
        plt.tight_layout()

        # sanitize filename
        safe_name = title.replace("/", "_").replace("\\", "_")
        outp = os.path.join(outdir, f"compare_bar_{safe_name}.png")
        plt.savefig(outp, dpi=160)
        plt.close(fig)
        print(f"[Saved] {outp}")

    # --- 2. Utility vs Privacy scatter
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, prefix, color in [(label1, "_1", "tab:blue"), (label2, "_2", "tab:orange")]:
        if f"Privacy_S{prefix}" in merged and f"Utility_S{prefix}" in merged:
            ax.scatter(
                merged[f"Privacy_S{prefix}"],
                merged[f"Utility_S{prefix}"],
                label=label,
                alpha=0.8,
                c=color
            )
    ax.set_xlabel("Privacy (%)")
    ax.set_ylabel("Utility (%)")
    ax.set_title("Privacy vs Utility Comparison")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    outp = os.path.join(outdir, "compare_scatter_privacy_utility.png")
    plt.savefig(outp, dpi=160)
    plt.close(fig)
    print(f"[Saved] {outp}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run1", required=True, help="Folder for first run (baseline)")
    ap.add_argument("--run2", required=False, help="Folder for second run (airgap)")
    ap.add_argument("--label1", default="Baseline")
    ap.add_argument("--label2", default="AirGap")
    ap.add_argument("--outdir", default="plots_compare")
    args = ap.parse_args()

    # Load baseline
    if os.path.isdir(args.run1):
        df1 = load_combined_reports(args.run1, tag=args.label1)
    else:
        df1 = load_attack_or_eval(args.run1, args.label1)

    # Load airgap
    if args.run2:
        if os.path.isdir(args.run2):
            df2 = load_combined_reports(args.run2, tag=args.label2)
        else:
            df2 = load_attack_or_eval(args.run2, args.label2)
    else:
        df2 = pd.DataFrame()

    # Compare or visualize single run
    if not df2.empty:
        merged = compare_data(df1, df2)
        table = merged
    else:
        merged = df1.copy()
        table = merged

    print("\n=== COMPARISON TABLE ===\n")
    print(table.to_string(index=False))

    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "comparison_table.csv")
    table.to_csv(csv_path, index=False)
    print(f"[Saved] {csv_path}")

    plot_comparison(merged, args.outdir, args.label1, args.label2)


if __name__ == "__main__":
    main()
