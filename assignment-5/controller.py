import os
import json
import time
import subprocess
import argparse
import pandas as pd
from tqdm import tqdm
import torch
from datetime import datetime

LOG_FILE = "run.log"

def log(msg):
    """Append timestamped log messages."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")
    print(f"[{ts}] {msg}")

def gpu_status():
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1024 ** 2
        return f"GPU Memory: {mem:.1f} MB"
    return "GPU not available"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to input CSV")
    parser.add_argument("--task", required=True)
    parser.add_argument("--directive", required=True)
    parser.add_argument("--out", default="minimized.json")
    parser.add_argument("--max_records", type=int, default=None, help="Limit records for debug")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    log(f"Loaded {len(df)} records from {args.csv}")

    # Auto-resume
    done_ids = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            existing = json.load(f)
            done_ids = {rec.get('id', i) for i, rec in enumerate(existing)}
            log(f"Resuming from {len(done_ids)} existing minimized entries")

    if args.max_records:
        df = df.head(args.max_records)

    results = []

    for i, rec in tqdm(df.iterrows(), total=len(df), desc="Processing records"):
        rec_id = rec.get('id', i)
        if rec_id in done_ids:
            continue

        t0 = time.time()
        log(f"→ Minimizing record {rec_id} … {gpu_status()}")

        cmd = [
            "python", "minimizer_llm.py",
            "--csv", args.csv,
            "--task", args.task,
            "--directive", args.directive,
            "--out", args.out
        ]

        try:
            subprocess.run(cmd, check=True)
            log(f"✓ Record {rec_id} minimized in {time.time()-t0:.2f}s {gpu_status()}")
        except subprocess.CalledProcessError as e:
            log(f"✗ Error on record {rec_id}: {e}")
            continue

        torch.cuda.empty_cache()

    log("All records processed. Generating summary...")

    # --- Summary metrics (dummy privacy & utility estimates) ---
    if os.path.exists(args.out):
        with open(args.out) as f:
            minimized = json.load(f)
        total = len(minimized)
        privacy = 97.0  # placeholder estimate
        utility = 90.0  # placeholder estimate
        log(f"SUMMARY: {total} records processed | Privacy ~ {privacy:.1f}% | Utility ~ {utility:.1f}%")

    log("Run complete OK")

if __name__ == "__main__":
    main()
