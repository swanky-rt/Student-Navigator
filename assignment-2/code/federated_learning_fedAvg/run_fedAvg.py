#!/usr/bin/env python3
"""
run_fedAvg.py

Thin runner for the full pipeline:
- central: train centralized baseline
- iid:     run FedAvg with IID splits
- non_iid: run FedAvg with label-skew (non-IID) splits
- plot:    generate comparison plots

Usage examples:
  python run_fedavg.py
"""

import argparse, subprocess, sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
WORKDIR = ROOT

SCRIPTS = {
    "central": "centralize_global_file.py",
    "iid":     "federated_learning_iid.py",
    "non_iid": "federated_learning_non_iid.py",
    "plot":    "graph_plotting.py",
}


def run(cmd: list[str], cwd: Path, keep_going: bool) -> None:
    """Run a subprocess, optionally continue on failure."""
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] ▶ {' '.join(cmd)}  (cwd={cwd})")
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"✖ Command failed ({e.returncode})")
        if not keep_going:
            sys.exit(e.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run FedAvg pipeline (location-agnostic)")
    ap.add_argument("--mode", choices=["all", "iid", "non-iid"], default="all")
    ap.add_argument("--skip-central", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--python-path", default=sys.executable)
    args = ap.parse_args()

    if not WORKDIR.exists():
        print(f"directory not found: {WORKDIR}"); sys.exit(1)

    # Build the sequence of scripts based on flags
    steps: list[str] = []
    if not args.skip_central: steps.append(SCRIPTS["central"])
    if args.mode in ("all", "iid"):     steps.append(SCRIPTS["iid"])
    if args.mode in ("all", "non-iid"): steps.append(SCRIPTS["non_iid"])
    if not args.no_plot:                 steps.append(SCRIPTS["plot"])

    # Quick existence check
    missing = [s for s in steps if not (WORKDIR / s).exists()]
    if missing:
        print(" missing scripts in this folder:\n  " + "\n  ".join(missing)); sys.exit(1)

    # Summary
    print("=== FedAvg Runner ===")
    print(f"Folder     : {WORKDIR}")
    print(f"Python     : {args.python_path}")
    print(f"Mode       : {args.mode}")
    print(f"Skip centr.: {args.skip_central}")
    print(f"Plots      : {not args.no_plot}")
    print(f"Keep going : {args.keep_going}")
    print("=" * 26)

    # Execute
    for s in steps:
        run([args.python_path, s], WORKDIR, args.keep_going)

    print("FedAvg done.")


if __name__ == "__main__":
    main()
