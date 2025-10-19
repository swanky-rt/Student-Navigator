#!/usr/bin/env python3
"""
controller.py — orchestrates the full privacy–utility pipeline.
Runs all scenarios end-to-end: minimization → attack–defense simulation → evaluation.
Handles temporary CSVs, subprocess calls, and summary generation.
"""

import os
import argparse
import subprocess
import json
import tempfile
import pandas as pd
from datetime import datetime


# --- Scenario configurations ---
# Each scenario defines a directive and a selector function
# that filters the dataset for relevant records.
SCENARIOS = {
    "recruiter_outreach": {
        "directive": "recruiter_outreach",
        "selector": lambda df: df["job_title"].str.contains("Engineer|Manager|Analyst|Designer", case=False, na=False),
    },
    "public_job_board": {
        "directive": "public_job_board",
        "selector": lambda df: df.index % 2 == 0,
    },
    "internal_hr": {
        "directive": "internal_hr",
        "selector": lambda df: df["notes"].notna(),
    },
    "research_dataset": {
        "directive": "research_dataset",
        "selector": lambda df: df["company_name"].str.contains("Inc|LLC|Corp|Ltd", case=False, na=False),
    },
    "marketing_campaign": {
        "directive": "marketing_campaign",
        "selector": lambda df: df["job_title"].str.contains("Marketing|Sales|Campaign", case=False, na=False),
    },
}


def write_temp_csv(df_subset, directive_key):
    """Write a filtered subset to a temporary CSV and attach its directive."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="scenario_")
    path = tmp.name
    tmp.close()

    df = df_subset.copy().astype(str)
    df["directive"] = directive_key
    if len(df) > 0:
        print(f"[i] Writing {len(df)} rows to temp CSV, directive='{directive_key}'")

    df.to_csv(path, index=False, encoding="utf-8")
    return path


def run_cmd(cmd, capture=False):
    """Execute a shell command. Optionally capture stdout and stderr."""
    print("[CMD]", " ".join(cmd))
    if capture:
        res = subprocess.run(cmd, capture_output=True, text=True)
        return res.returncode, res.stdout + res.stderr
    else:
        return subprocess.run(cmd).returncode, None


def ensure_dir(d):
    """Create directory if it does not exist."""
    if not os.path.exists(d):
        os.makedirs(d)


def main():
    """Main driver: loops through scenarios and executes all stages for each."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="raw csv input")
    parser.add_argument("--out_dir", default="runs", help="where per-scenario outputs go")
    parser.add_argument("--task", default="draft outreach email to recruiter")
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--minimizer_model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--conversational_model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--attacker_model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--minimizer_batch", type=int, default=1)
    parser.add_argument("--max_prompts", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--save_transcripts", action="store_true")
    parser.add_argument("--redaction_strengths", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--attacker_mode", choices=["context_preserving", "hijacking"], default="context_preserving",
                        help="Attacker strategy for interactive mode.")
    parser.add_argument("--hijack_style", choices=["mild", "extreme"], default="mild",
                        help="Hijack pretext style for hijacking mode.")
    parser.add_argument("--model_variant", choices=["airgap", "baseline"], default="airgap",
                        help="baseline skips minimizer and uses original CSV directly")

    args = parser.parse_args()

    # Parse arguments and prepare data
    strengths = [float(x) for x in args.redaction_strengths.split(",")]
    df = pd.read_csv(args.csv, dtype=str).fillna("")
    ensure_dir(args.out_dir)
    summary = {}

    # Loop through each predefined scenario
    for sc_name, sc_cfg in SCENARIOS.items():
        print(f"\n Scenario: {sc_name} (directive={sc_cfg['directive']}) ===")
        sel = sc_cfg["selector"](df) if callable(sc_cfg["selector"]) else sc_cfg["selector"]
        df_subset = df[sel].copy()

        # Limit number of rows for testing or debugging
        if args.max_records:
            df_subset = df_subset.head(args.max_records)
        if df_subset.empty:
            print(f"[!] No rows selected for scenario {sc_name}, skipping.")
            continue

        # Add ID column if missing to preserve mapping with original CSV
        if "id" in df.columns and "id" not in df_subset.columns:
            df_subset.insert(0, "id", df.loc[df_subset.index, "id"].astype(str).values)

        # Temporary input CSV for this scenario
        tmp_csv = write_temp_csv(df_subset, sc_cfg["directive"])
        scenario_dir = os.path.join(args.out_dir, sc_name)
        ensure_dir(scenario_dir)

        # Iterate through all redaction strength levels
        for strength in strengths:
            print(f"  >> Running redaction_strength={strength}")
            subdir = os.path.join(scenario_dir, f"redaction_{int(strength * 100)}")
            ensure_dir(subdir)
            minimized_out = os.path.join(subdir, "minimized.json")

            # Map directive to human-readable task text
            directive_to_task = {
                "recruiter_outreach": "Draft outreach email to a candidate.",
                "public_job_board": "Publish a public job listing safely.",
                "internal_hr": "Prepare internal HR analytics data.",
                "marketing_campaign": "Design a marketing campaign dataset.",
                "research_dataset": "Release a research dataset with anonymization."
            }

            directive = sc_cfg["directive"]
            task = directive_to_task.get(directive, "General data sharing task.")

            # Step 1: Minimizer — runs LLM-based data minimization or skips if baseline
            if args.model_variant == "baseline":
                print("[INFO] Baseline mode → skipping minimizer, using original CSV.")
                minimized_out = tmp_csv
            else:
                minimizer_cmd = [
                    "python", "minimizer_llm.py",
                    "--csv", tmp_csv,
                    "--out", minimized_out,
                    "--task", task,
                    "--model", args.minimizer_model,
                    "--redaction_strength", str(strength),
                    "--max_records", str(args.max_records or 0),
                    "--debug",
                ]
                code, log = run_cmd(minimizer_cmd, capture=True)
                with open(os.path.join(subdir, "minimizer_log.txt"), "w", encoding="utf-8") as f:
                    f.write(log or "")

            # Step 2: Interactive Attack–Defense Simulation
            # Launch attacker and defender models in multi-turn interaction
            attack_out = os.path.join(subdir, "attack_report.json")
            attack_cmd = [
                "python", "attack_defense_sim.py",
                "--orig", tmp_csv,
                "--minimized", minimized_out,
                "--model_defender", args.conversational_model,
                "--model_attacker", args.attacker_model,
                "--device", str(args.device),
                "--max_turns", "6",
                "--stop_on_recovery", "True",
                "--out", attack_out,
                "--max_new_tokens", str(args.max_new_tokens),
                "--attacker_mode", args.attacker_mode,
                "--hijack_style", args.hijack_style,
            ]
            if args.save_transcripts:
                attack_cmd += ["--save_transcripts"]

            code, log = run_cmd(attack_cmd, capture=True)
            with open(os.path.join(subdir, "attack_log.txt"), "w", encoding="utf-8") as f:
                f.write(log or "")

            # Step 3: Evaluation — compute privacy and utility metrics
            eval_out = os.path.join(subdir, "evaluation_report.json")
            eval_cmd = [
                "python", "evaluate_privacy_utility.py",
                "--orig", tmp_csv,
                "--minimized", minimized_out,
                "--out", eval_out
            ]
            code, log = run_cmd(eval_cmd, capture=True)
            with open(os.path.join(subdir, "evaluation_log.txt"), "w", encoding="utf-8") as f:
                f.write(log or "")

            # Gather run results into summary dictionary
            if os.path.exists(eval_out):
                with open(eval_out) as f:
                    rep = json.load(f)
            else:
                rep = {"error": "evaluation missing"}

            summary[f"{sc_name}_r{int(strength * 100)}"] = {
                "directive": sc_cfg["directive"],
                "strength": strength,
                "rows": len(df_subset),
                "report": rep
            }

        # Remove temporary CSV after processing
        os.unlink(tmp_csv)

    # Save combined summary for all scenarios
    ts = datetime.utcnow().isoformat().replace(":", "-")
    summary_file = os.path.join(args.out_dir, f"run_summary_{ts}.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll scenarios completed. Summary saved to {summary_file}")


if __name__ == "__main__":
    # Disable unnecessary warnings from transformers/tokenizers
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
