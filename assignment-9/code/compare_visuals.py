# compare_visuals.py
import argparse, json, math, os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

METRICS = [
    ("pytest_pass_rate", "Pass Rate", True),          # higher is better
    ("bandit_findings", "Bandit Findings", False),    # lower is better
    ("ruff_count", "Ruff Diagnostics", False),
    ("flake8_length", "Flake8 Diagnostics", False),
    ("radon_length", "Radon CC Blocks", False),
    ("docstring_count", "Docstrings", True),
    ("typehint_count", "Type Hints", True),
]

def load_report(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def get(report, key, default=0):
    return report.get(key, default)

def extract(report):
    vals = []
    for key, _, _ in METRICS:
        vals.append(float(get(report, key, 0)))
    return np.array(vals, dtype=float)

def normalize(values, higher_is_better):
    # min-max per metric across A/B for radar comparison
    v = np.asarray(values)
    vmin = v.min(axis=0)
    vmax = v.max(axis=0)
    # protect against div-by-zero
    rng = np.where(vmax - vmin == 0, 1.0, vmax - vmin)
    z = (v - vmin) / rng
    # if lower is better, invert
    for i, hib in enumerate(higher_is_better):
        if not hib:
            z[:, i] = 1.0 - z[:, i]
    return z

def grouped_bars(metrics_labels, A_vals, B_vals, label_a, label_b, out_png):
    x = np.arange(len(metrics_labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5.5))
    # Custom colors as requested
    bars_a = ax.bar(x - width/2, A_vals, width, label=label_a, color="#4C78A8")
    bars_b = ax.bar(x + width/2, B_vals, width, label=label_b, color="#F58518")

    # % change labels above B bars
    for xi, (a, b) in enumerate(zip(A_vals, B_vals)):
        if a == 0 and b == 0:
            txt = "0%"
        elif a == 0:
            txt = "∞"
        else:
            txt = f"{(b - a)/a*100:.1f}%"
        ax.text(xi + width/2, b + (0.01 * (max(B_vals) if max(B_vals) else 1)),
                txt, ha="center", va="bottom", fontsize=9, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels, rotation=20, ha="right")
    ax.set_ylabel("Value")
    ax.set_title("Metric Comparison")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def radar(metrics_labels, A_vals, B_vals, label_a, label_b, out_png, higher_is_better):
    # Normalize to 0..1 (accounting for 'lower is better' metrics)
    both = np.vstack([A_vals, B_vals])
    z = normalize(both, [hib for _, _, hib in METRICS])

    A = z[0]
    B = z[1]

    # Radar setup
    N = len(metrics_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    A = np.concatenate((A, [A[0]]))
    B = np.concatenate((B, [B[0]]))
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(7, 7))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids(np.degrees(angles[:-1]), metrics_labels)
    ax.set_ylim(0, 1)

    # Fill and lines with custom colors
    ax.plot(angles, A, linewidth=2, color="#4C78A8", label=label_a)
    ax.fill(angles, A, alpha=0.15, color="#4C78A8")
    ax.plot(angles, B, linewidth=2, color="#F58518", label=label_b)
    ax.fill(angles, B, alpha=0.15, color="#F58518")

    ax.set_title("Radar (Normalized: higher area = better)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05))
    ax.grid(alpha=0.4)

    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Report A (e.g., baseline_report.json)")
    ap.add_argument("--b", required=True, help="Report B (e.g., misaligned_report.json)")
    ap.add_argument("--prefix", required=True, help="Output prefix for image files")
    ap.add_argument("--label_a", default="A", help="Legend label for A")
    ap.add_argument("--label_b", default="B", help="Legend label for B")
    args = ap.parse_args()

    ra = load_report(args.a)
    rb = load_report(args.b)

    labels = [pretty for _, pretty, _ in METRICS]
    A_vals = extract(ra)
    B_vals = extract(rb)

    out_dir = Path(os.path.dirname(args.prefix) or ".")
    out_dir.mkdir(parents=True, exist_ok=True)

    bars_png = f"{args.prefix}_bars.png"
    grouped_bars(labels, A_vals, B_vals, args.label_a, args.label_b, bars_png)

    radar_png = f"{args.prefix}_radar.png"
    radar(labels, A_vals, B_vals, args.label_a, args.label_b, radar_png, [hib for _, _, hib in METRICS])

    print(f"Wrote: {bars_png}")
    print(f"Wrote: {radar_png}")

if __name__ == "__main__":
    main()
