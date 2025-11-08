# make_plots.py
"""
Generate all plots for the backdoor assignment:
- ASR decay (before vs after continued fine-tuning)
- CA trajectory (before vs after)
- ASR vs CA scatter (trade-off)
- Robustness bars (ASR/CA by variation)
- CSV summary export

Inputs (defaults match your earlier runs):
  models/backdoor_p10/asr_ca_history.json
  models/backdoor_p10_finetuned/asr_ca_history.json
  results/robustness/robustness_results.json

Outputs:
  results/plots/asr_decay.png
  results/plots/ca_trajectory.png
  results/plots/asr_vs_ca.png
  results/plots/robustness_asr.png
  results/plots/robustness_ca.png
  results/plots/summary.csv
"""

import os, json, argparse, csv
import matplotlib.pyplot as plt

def _read_json(path):
    if not os.path.exists(path):
        print(f"[WARN] Missing file: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def _list_to_xy(history_list):
    """Convert a HF checkpoint list into xs, ASR, CA arrays."""
    if not history_list:
        return [], [], []
    xs = list(range(1, len(history_list)+1))
    asrs = [float(d.get("ASR", 0.0)) for d in history_list]
    cas  = [float(d.get("CA", 0.0))  for d in history_list]
    return xs, asrs, cas

def plot_asr_decay(before_hist, after_hist, out_path):
    plt.figure(figsize=(7,4))
    if before_hist:
        xs_b, asrs_b, _ = _list_to_xy(before_hist)
        if xs_b:
            plt.plot(xs_b, asrs_b, marker='o', label="Before finetune")
    if after_hist:
        xs_a, asrs_a, _ = _list_to_xy(after_hist)
        if xs_a:
            plt.plot(xs_a, asrs_a, marker='o', label="After finetune")
    plt.xlabel("Checkpoint / Epoch")
    plt.ylabel("ASR")
    plt.title("ASR decay across checkpoints")
    plt.ylim(-0.01, 1.01)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[OK] Saved {out_path}")

def plot_ca_trajectory(before_hist, after_hist, out_path):
    plt.figure(figsize=(7,4))
    if before_hist:
        xs_b, _, cas_b = _list_to_xy(before_hist)
        if xs_b:
            plt.plot(xs_b, cas_b, marker='o', label="Before finetune")
    if after_hist:
        xs_a, _, cas_a = _list_to_xy(after_hist)
        if xs_a:
            plt.plot(xs_a, cas_a, marker='o', label="After finetune")
    plt.xlabel("Checkpoint / Epoch")
    plt.ylabel("Clean Accuracy (CA)")
    plt.title("CA trajectory across checkpoints")
    plt.ylim(-0.01, 1.01)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[OK] Saved {out_path}")

def plot_asr_vs_ca(before_hist, after_hist, out_path):
    plt.figure(figsize=(5.6,5.6))
    if before_hist:
        _, asrs_b, cas_b = _list_to_xy(before_hist)
        if asrs_b:
            plt.scatter(cas_b, asrs_b, label="Before finetune", marker='o')
    if after_hist:
        _, asrs_a, cas_a = _list_to_xy(after_hist)
        if asrs_a:
            plt.scatter(cas_a, asrs_a, label="After finetune", marker='^')
    plt.xlabel("CA")
    plt.ylabel("ASR")
    plt.title("ASR vs CA (Utility–Security trade-off)")
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[OK] Saved {out_path}")

def plot_robustness_bars(robust_dict, out_asr, out_ca):
    if not robust_dict:
        print("[WARN] No robustness results JSON.")
        return
    # Expect: dict[name] -> {"ASR": float, "CA": float, ...}
    names = []
    asrs = []
    cas  = []
    for k, v in robust_dict.items():
        names.append(k)
        asrs.append(float(v.get("ASR", 0.0)))
        cas.append(float(v.get("CA", 0.0)))

    # ASR bars
    plt.figure(figsize=(max(6, 0.9*len(names)), 4))
    plt.bar(names, asrs)
    plt.ylabel("ASR")
    plt.ylim(0, 1.0)
    plt.title("Robustness: ASR by variant")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_asr, dpi=180)
    plt.close()
    print(f"[OK] Saved {out_asr}")

    # CA bars
    plt.figure(figsize=(max(6, 0.9*len(names)), 4))
    plt.bar(names, cas)
    plt.ylabel("CA")
    plt.ylim(0, 1.0)
    plt.title("Robustness: CA by variant")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_ca, dpi=180)
    plt.close()
    print(f"[OK] Saved {out_ca}")

def export_summary_csv(before_hist, after_hist, robust_dict, out_csv):
    rows = []
    # Per-checkpoint (before)
    if before_hist:
        for i, h in enumerate(before_hist, 1):
            rows.append({
                "phase": "before",
                "checkpoint_index": i,
                "checkpoint_path": h.get("checkpoint", ""),
                "ASR": h.get("ASR", ""),
                "CA":  h.get("CA", "")
            })
    # Per-checkpoint (after)
    if after_hist:
        for i, h in enumerate(after_hist, 1):
            rows.append({
                "phase": "after",
                "checkpoint_index": i,
                "checkpoint_path": h.get("checkpoint", ""),
                "ASR": h.get("ASR", ""),
                "CA":  h.get("CA", "")
            })
    # Robustness
    if robust_dict:
        for name, v in robust_dict.items():
            rows.append({
                "phase": "robustness",
                "checkpoint_index": "",
                "checkpoint_path": name,
                "ASR": v.get("ASR", ""),
                "CA":  v.get("CA", "")
            })

    fieldnames = ["phase", "checkpoint_index", "checkpoint_path", "ASR", "CA"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] Wrote CSV summary -> {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before_json", default="models/backdoor_p10/asr_ca_history.json",
                    help="ASR/CA history JSON for backdoored model")
    ap.add_argument("--after_json", default="models/backdoor_p10_finetuned/asr_ca_history.json",
                    help="ASR/CA history JSON for continued finetune")
    ap.add_argument("--robust_json", default="results/robustness/robustness_results.json",
                    help="Robustness results JSON")
    ap.add_argument("--out_dir", default="results/plots", help="Output directory for plots & CSV")
    args = ap.parse_args()

    _ensure_dir(args.out_dir)

    before_hist = _read_json(args.before_json)
    after_hist  = _read_json(args.after_json)
    robust_dict = _read_json(args.robust_json)

    # Plots
    plot_asr_decay(before_hist, after_hist, os.path.join(args.out_dir, "asr_decay.png"))
    plot_ca_trajectory(before_hist, after_hist, os.path.join(args.out_dir, "ca_trajectory.png"))
    plot_asr_vs_ca(before_hist, after_hist, os.path.join(args.out_dir, "asr_vs_ca.png"))
    plot_robustness_bars(robust_dict, os.path.join(args.out_dir, "robustness_asr.png"),
                         os.path.join(args.out_dir, "robustness_ca.png"))

    # CSV summary
    export_summary_csv(before_hist, after_hist, robust_dict, os.path.join(args.out_dir, "summary.csv"))

if __name__ == "__main__":
    main()
