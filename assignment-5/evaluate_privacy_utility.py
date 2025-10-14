import re, json, argparse
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

# --- PII detection regexes ---
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\-\s().]{7,}\d")
REF_RE   = re.compile(r"\b(?:REF[-\d]{6,}|[A-Z]{2,}-\d{3,}-\d{2,}-\d{3,})\b")

def contains_pii(text: str) -> bool:
    return bool(EMAIL_RE.search(text) or PHONE_RE.search(text) or REF_RE.search(text))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig", required=True)
    parser.add_argument("--minimized", required=True)
    args = parser.parse_args()

    print("[✓] Evaluating privacy–utility tradeoff")
    df_orig = pd.read_csv(args.orig, dtype=str).fillna("")
    with open(args.minimized, "r", encoding="utf-8") as f:
        minimized = json.load(f)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("[✓] Loaded evaluation model")

    # Counters
    total_fields = 0
    false_neg = 0
    false_pos = 0
    leak_count = 0  # how many PII tokens leaked
    sim_sum = 0.0
    sim_count = 0
    blanks_overredacted = 0

    # per-field breakdown
    per_field_stats = {fld: {"FN":0, "FP":0, "count":0, "leak":0, "sim_sum":0.0} 
                        for fld in ["job_title","job_description","company_name","contact_info","notes"]}

    for orig, mini in zip(df_orig.to_dict(orient="records"), minimized):
        for fld in per_field_stats.keys():
            o = str(orig.get(fld, ""))
            m = str(mini.get(fld, ""))

            # skip if original is empty (no content)
            if o.strip() == "":
                continue

            per_field_stats[fld]["count"] += 1
            total_fields += 1

            orig_has_pii = contains_pii(o)
            mini_has_pii = contains_pii(m)

            # —— Leakage detection / false negatives
            if orig_has_pii and mini_has_pii:
                false_neg += 1
                per_field_stats[fld]["FN"] += 1
                # count how many PII fragments remain (simple approach: count matches)
                leaked_tokens = len(EMAIL_RE.findall(m) + PHONE_RE.findall(m) + REF_RE.findall(m))
                per_field_stats[fld]["leak"] += leaked_tokens
                leak_count += leaked_tokens

            # —— Over-redaction (false positives)
            if not orig_has_pii and m.strip() == "":
                false_pos += 1
                per_field_stats[fld]["FP"] += 1
                blanks_overredacted += 1

            # —— Similarity / utility for non-empty minimized fields
            if m.strip() != "":
                sim = util.cos_sim(model.encode(o, convert_to_tensor=True),
                                   model.encode(m, convert_to_tensor=True)).item()
                sim_sum += sim
                sim_count += 1
                per_field_stats[fld]["sim_sum"] += sim

    # Compute metrics
    attack_success = (false_neg / total_fields) * 100 if total_fields > 0 else 0.0
    privacy_retention = ((total_fields - false_neg) / total_fields) * 100 if total_fields > 0 else 0.0
    over_redaction_rate = (false_pos / total_fields) * 100 if total_fields > 0 else 0.0
    avg_similarity = (sim_sum / sim_count) * 100 if sim_count > 0 else 0.0

    report = {
        "timestamp": str(datetime.now()),
        "fields_evaluated": total_fields,
        "attack_success_%": round(attack_success, 2),
        "privacy_retention_%": round(privacy_retention, 2),
        "over_redaction_rate_%": round(over_redaction_rate, 2),
        "avg_similarity_%": round(avg_similarity, 2),
        "leak_count": leak_count,
        "false_negatives": false_neg,
        "false_positives": false_pos,
        "per_field": {
            fld: {
                "count": st["count"],
                "FN": st["FN"],
                "FP": st["FP"],
                "leak": st["leak"],
                "avg_sim_%": round((st["sim_sum"] / st["count"] * 100) if st["count"]>0 else 0.0, 2)
            }
            for fld, st in per_field_stats.items()
        }
    }

    print("📊 Evaluation Report")
    for k, v in report.items():
        if k == "per_field":
            print(" per-field breakdown:")
            for fld, st in report["per_field"].items():
                print(f"  {fld}: {st}")
        else:
            print(f"{k:25s}: {v}")
    
    # Convert to JSON safe types (if needed)
    def make_safe(x):
        if isinstance(x, dict):
            return {k: make_safe(v) for k, v in x.items()}
        if isinstance(x, float):
            return float(x)
        return x
    report = make_safe(report)

    with open("evaluation_report_detailed.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("[✓] Saved detailed report to evaluation_report_detailed.json")

if __name__ == "__main__":
    main()