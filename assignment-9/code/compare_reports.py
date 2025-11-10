# compare_reports.py
import argparse, csv, json, math, os
from pathlib import Path

METRICS = [
    ("pytest_pass_rate", "Pass Rate"),
    ("bandit_findings", "Bandit Findings"),
    ("ruff_count", "Ruff Diagnostics"),
    ("flake8_length", "Flake8 Diagnostics"),
    ("radon_length", "Radon CC Blocks"),
    ("docstring_count", "Docstrings"),
    ("typehint_count", "Type Hints"),
]

def load_report(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def get(report, key, default=0):
    # supports nested "counts.num_files" if needed
    if "." in key:
        cur = report
        for part in key.split("."):
            cur = cur.get(part, {})
        return cur if isinstance(cur, (int, float)) else default
    return report.get(key, default)

def pct_delta(a, b):
    try:
        if a == 0:
            return float("inf") if b != 0 else 0.0
        return (b - a) / a * 100.0
    except Exception:
        return float("nan")

def fmt(x):
    if isinstance(x, float):
        # pretty print floats, including pass rate 0..1
        return f"{x:.3f}"
    return str(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Report A (e.g., baseline_report.json)")
    ap.add_argument("--b", required=True, help="Report B (e.g., misaligned_report.json)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--label_a", default="A", help="Label for report A")
    ap.add_argument("--label_b", default="B", help="Label for report B")
    args = ap.parse_args()

    ra = load_report(args.a)
    rb = load_report(args.b)

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)

    rows = []
    for key, pretty in METRICS:
        va = get(ra, key, 0)
        vb = get(rb, key, 0)
        # normalize types
        if isinstance(va, str):
            with contextlib.suppress(Exception): va = float(va)
        if isinstance(vb, str):
            with contextlib.suppress(Exception): vb = float(vb)

        delta = vb - va
        pdelta = pct_delta(va, vb)

        rows.append({
            "metric": pretty,
            f"value_{args.label_a}": va,
            f"value_{args.label_b}": vb,
            "delta": delta,
            "pct_delta": pdelta,
        })

    # Write CSV
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", f"value_{args.label_a}", f"value_{args.label_b}", "delta", "pct_delta"])
        w.writeheader()
        for r in rows:
            r2 = dict(r)
            r2["pct_delta"] = "∞" if math.isinf(r["pct_delta"]) else f"{r['pct_delta']:.1f}%"
            w.writerow(r2)

    # Pretty console table
    colw = [max(len(str(r[k])) for r in rows + [{"metric":"Metric", f"value_{args.label_a}":"", f"value_{args.label_b}":"", "delta":"", "pct_delta":""}]) for k in
            ["metric", f"value_{args.label_a}", f"value_{args.label_b}", "delta", "pct_delta"]]
    headers = ["Metric", f"Value {args.label_a}", f"Value {args.label_b}", "Δ (B−A)", "%Δ"]
    print("\n" + " | ".join(h.ljust(w) for h, w in zip(headers, colw)))
    print("-" * (sum(colw) + 3 * (len(colw) - 1)))
    for r in rows:
        line = [
            str(r["metric"]).ljust(colw[0]),
            fmt(r[f"value_{args.label_a}"]).ljust(colw[1]),
            fmt(r[f"value_{args.label_b}"]).ljust(colw[2]),
            fmt(r["delta"]).ljust(colw[3]),
            ("∞" if math.isinf(r["pct_delta"]) else f"{r['pct_delta']:.1f}%").ljust(colw[4]),
        ]
        print(" | ".join(line))

    print(f"\nWrote CSV: {args.out}")

if __name__ == "__main__":
    import contextlib
    main()
