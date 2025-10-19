#!/usr/bin/env python3
"""
evaluate_privacy_utility.py — computes privacy and utility metrics.
Compares original and minimized datasets to quantify redaction quality.
Calculates privacy leakage, over-redaction, and semantic utility scores.
"""

import re, json, argparse
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import os

# --- PII detection regexes (same patterns used across modules) ---
# Detects common PII: emails, phone numbers, reference IDs, names, DOBs, addresses, experience, and URLs
EMAIL_RE  = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PHONE_RE  = re.compile(r"\+?\d[\d\-\s().]{7,}\d")
REF_RE    = re.compile(r"\b(?:REF[-\d]{6,}|[A-Z]{2,}-\d{3,}-\d{2,}-\d{3,})\b")
NAME_RE   = re.compile(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+){0,2}\b")
DOB_RE    = re.compile(r"\b(19[7-9]\d|200[0-2])-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b")
ADDR_RE   = re.compile(r"\d{1,5}\s+[A-Za-z0-9.\- ]+,\s*[A-Za-z .]+,\s*[A-Z]{2}\s*\d{5}|\d{1,5}\s+[A-Za-z0-9.\- ]+,\s*[A-Za-z .]+")
EXP_RE    = re.compile(r"\b(?:[0-9]|[1-3][0-9]|40)\b")
WEBSITE_RE= re.compile(r"https?://[A-Za-z0-9\-_]+\.[A-Za-z]{2,}(?:/[^\s]*)?")

# Fields evaluated for privacy–utility analysis
FIELDS = [
    "job_title", "job_description", "company_name", "contact_info", "notes",
    "name", "dob", "address", "years_experience", "personal_website"
]

def contains_pii(text: str) -> bool:
    """Return True if any PII pattern matches within a string."""
    return bool(
        EMAIL_RE.search(text)
        or PHONE_RE.search(text)
        or REF_RE.search(text)
        or NAME_RE.search(text)
        or DOB_RE.search(text)
        or ADDR_RE.search(text)
        or WEBSITE_RE.search(text)
    )

def load_minimized_any(path):
    """Load minimized file from JSON, JSONL, or CSV, depending on input format."""
    if not path or not isinstance(path, str):
        raise ValueError("Invalid path to minimized file.")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # Try loading as JSON
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        else:
            return [data]
    except Exception:
        pass

    # Try JSONL (line-delimited JSON)
    try:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if records:
            return records
    except Exception:
        pass

    # Try CSV
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
        return df.to_dict(orient="records")
    except Exception as e:
        raise RuntimeError(f"Failed to parse minimized file: {e}")

def main():
    """Main evaluation routine: computes privacy and utility metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig", required=True, help="Original CSV")
    parser.add_argument("--minimized", required=True, help="Minimized JSON/CSV/JSONL")
    parser.add_argument("--out", default="evaluation_report.json")
    parser.add_argument("--similarity_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--model_variant", choices=["airgap","baseline"], default="airgap",
                        help="For consistency with controller; does not change metrics.")
    args = parser.parse_args()

    print("Evaluating Privacy–Utility Tradeoff (AirGap metrics)")
    df_orig = pd.read_csv(args.orig, dtype=str).fillna("")
    minimized = load_minimized_any(args.minimized)

    # Load semantic similarity model for utility computation
    model = SentenceTransformer(args.similarity_model, device="cuda")
    print("[i] Loaded embedding model for semantic similarity")

    # Metric accumulators
    total_fields = 0
    pii_fields = 0
    false_neg = 0   # PII present in minimized (leak)
    false_pos = 0   # Non-PII removed (over-redaction)
    leak_count = 0
    sim_sum = 0.0
    sim_count = 0

    # Track per-field breakdown for finer diagnostics
    per_field_stats = {
        fld: {"FN":0, "FP":0, "count":0, "leak":0, "sim_sum":0.0}
        for fld in FIELDS
    }

    # Iterate through all corresponding original and minimized records
    for orig, mini in zip(df_orig.to_dict(orient="records"), minimized):
        for fld in FIELDS:
            o = str(orig.get(fld, "")).strip()
            m = str(mini.get(fld, "")).strip()
            if not o:
                continue

            total_fields += 1
            per_field_stats[fld]["count"] += 1

            # Determine if each field contains PII before and after minimization
            orig_has_pii = contains_pii(o)
            mini_has_pii = contains_pii(m)
            if orig_has_pii:
                pii_fields += 1

            # Privacy: false negative means leaked PII survived minimization
            if orig_has_pii and mini_has_pii:
                false_neg += 1
                per_field_stats[fld]["FN"] += 1
                leaked_tokens = len(EMAIL_RE.findall(m) + PHONE_RE.findall(m) + REF_RE.findall(m))
                per_field_stats[fld]["leak"] += leaked_tokens
                leak_count += leaked_tokens

            # Over-redaction: non-PII field removed unnecessarily
            if (not orig_has_pii) and m == "":
                false_pos += 1
                per_field_stats[fld]["FP"] += 1

            # Utility: semantic similarity between original and minimized text
            if m == "":
                sim = 0.0
            else:
                sim = util.cos_sim(
                    model.encode(o, convert_to_tensor=True),
                    model.encode(m, convert_to_tensor=True)
                ).item()
            sim_sum += sim
            sim_count += 1
            per_field_stats[fld]["sim_sum"] += sim

    # Compute overall metrics
    attack_success = (false_neg / pii_fields * 100) if pii_fields > 0 else 0.0
    privacy_retention = 100.0 - attack_success
    utility = (sim_sum / sim_count * 100) if sim_count > 0 else 0.0
    over_redaction_rate = (false_pos / total_fields * 100) if total_fields > 0 else 0.0

    # Build evaluation report
    report = {
        "timestamp": str(datetime.now()),
        "model_variant": args.model_variant,
        "fields_evaluated": total_fields,
        "pii_fields": pii_fields,
        "false_negatives": false_neg,
        "false_positives": false_pos,
        "leak_count": leak_count,
        "Attack_S": round(attack_success, 2),
        "Privacy_S": round(privacy_retention, 2),
        "Utility_S": round(utility, 2),
        "Over_redaction_%": round(over_redaction_rate, 2),
        "per_field": {
            fld: {
                "count": st["count"],
                "FN": st["FN"],
                "FP": st["FP"],
                "leak": st["leak"],
                "avg_sim_%": round((st["sim_sum"] / st["count"] * 100) if st["count"]>0 else 0.0, 2),
            }
            for fld, st in per_field_stats.items()
        }
    }

    # Record redaction strength if included in minimized data
    strength = minimized[0].get("_redaction_strength") if minimized and "_redaction_strength" in minimized[0] else None
    if strength is not None:
        report["redaction_strength"] = strength

    # Print human-readable summary
    print("\n=== AirGap Evaluation Report ===")
    for k, v in report.items():
        if k == "per_field":
            print(" per-field breakdown:")
            for fld, st in report["per_field"].items():
                print(f"  {fld}: {st}")
        else:
            print(f"{k:20s}: {v}")

    # Save detailed JSON report
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] Saved AirGap evaluation report to {args.out}")

if __name__ == "__main__":
    main()
