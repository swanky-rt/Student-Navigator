import pandas as pd
import json
import os

# Use paths relative to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(SCRIPT_DIR, "output_spacy_regex.csv")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "pii_metrics_output.csv")

# Map raw spaCy/regex labels to canonical ground-truth labels.
# Ground truth labels (from dataset): NAME, PHONE, EMAIL, DATE/DOB, company, location, SSN, CREDIT_CARD, IP, age, sex
# Predicted labels (from spacy_regex.py): PERSON, EMAIL, PHONE, IP_ADDRESS, SSN, CREDIT_CARD_16, CREDIT_CARD_4,
#                                          ORG, FAC, GPE, LOC, DATE, AGE, SEX
LABEL_MAP = {
    # spaCy NER labels
    "PERSON": "NAME",
    "ORG": "company",
    "FAC": "company",      # FAC (Facility) is normalized to ORG in spacy_regex.py, but kept here for safety
    "GPE": "location",     # Geo-Political Entity (cities, countries)
    "LOC": "location",     # Non-GPE locations (rivers, mountains, etc.)
    # Regex-detected labels
    "EMAIL": "EMAIL",
    "PHONE": "PHONE",
    "IP_ADDRESS": "IP",
    "SSN": "SSN",
    "CREDIT_CARD_16": "CREDIT_CARD",
    "CREDIT_CARD_4": "CREDIT_CARD",
    "DATE": "DATE/DOB",
    "AGE": "age",
    "SEX": "sex",
}


def parse_label_list(s: str) -> set:
    """
    Parse a string like "[NAME, PHONE, EMAIL]" into a set {"NAME","PHONE","EMAIL"}.
    Handles various formats in ground truth column.
    """
    if not isinstance(s, str):
        return set()
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    if not s.strip():
        return set()
    parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
    return {p for p in parts if p}


def extract_pred_labels(pii_str: str) -> set:
    """
    Parse the pii_entities JSON string and map labels to canonical labels.
    Returns a set of label strings (e.g. {"NAME","EMAIL","PHONE"}).
    
    The pii_entities column contains JSON arrays like:
    [{"text": "...", "label": "PERSON", "start": 0, "end": 10, "source": "spacy"}, ...]
    """
    if not isinstance(pii_str, str) or not pii_str.strip():
        return set()

    # Handle CSV escaping: doubled quotes become single quotes
    clean = pii_str.replace('""', '"')
    
    # Handle edge case where string starts/ends with extra quotes from CSV
    if clean.startswith('"') and clean.endswith('"'):
        clean = clean[1:-1]
    
    try:
        entities = json.loads(clean)
    except json.JSONDecodeError:
        # Try one more fix: sometimes the entire string is double-escaped
        try:
            entities = json.loads(json.loads(clean))
        except Exception:
            return set()
    except Exception:
        return set()

    if not isinstance(entities, list):
        return set()

    labels = set()
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        raw_label = ent.get("label")
        mapped = LABEL_MAP.get(raw_label)
        if mapped:
            labels.add(mapped)
        # Handle unmapped labels - they won't contribute to metrics
        # but we could log them for debugging

    return labels


def compute_row_metrics(y_true: set, y_pred: set):
    """
    Compute TP, FP, FN, precision, recall, f1, and subset accuracy for one row.
    
    - TP (True Positives): Labels correctly predicted (in both y_true and y_pred)
    - FP (False Positives): Labels predicted but not in ground truth (over-detection)
    - FN (False Negatives): Labels in ground truth but not predicted (under-detection)
    """
    tp = len(y_true & y_pred)
    fp = len(y_pred - y_true)
    fn = len(y_true - y_pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    exact_match = 1.0 if y_true == y_pred else 0.0

    return tp, fp, fn, precision, recall, f1, exact_match


def main():
    # Check if input file exists
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file not found: {INPUT_CSV}")
        print("Please run spacy_regex.py first to generate the output file.")
        return
    
    df = pd.read_csv(INPUT_CSV)
    
    # Validate required columns
    required_columns = ["ground_truth", "pii_entities"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return

    # Containers for global (micro) stats
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Containers for macro-averaged stats
    row_precisions = []
    row_recalls = []
    row_f1s = []

    # Columns to fill
    pred_labels_col = []
    tp_col = []
    fp_col = []
    fn_col = []
    prec_col = []
    rec_col = []
    f1_col = []
    exact_match_col = []

    for _, row in df.iterrows():
        y_true = parse_label_list(row["ground_truth"])
        y_pred = extract_pred_labels(row["pii_entities"])

        tp, fp, fn, precision, recall, f1, exact_match = compute_row_metrics(y_true, y_pred)

        pred_labels_col.append(sorted(list(y_pred)))
        tp_col.append(tp)
        fp_col.append(fp)
        fn_col.append(fn)
        prec_col.append(precision)
        rec_col.append(recall)
        f1_col.append(f1)
        exact_match_col.append(exact_match)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # For macro-average (only count rows with ground truth)
        if len(y_true) > 0:
            row_precisions.append(precision)
            row_recalls.append(recall)
            row_f1s.append(f1)

    # Add columns to dataframe
    df["predicted_labels"] = pred_labels_col
    df["tp"] = tp_col
    df["fp"] = fp_col
    df["fn"] = fn_col
    df["precision"] = prec_col
    df["recall"] = rec_col
    df["f1"] = f1_col
    df["exact_match"] = exact_match_col

    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved metrics to {OUTPUT_CSV}")

    # Calculate metrics
    # Micro-averaged: aggregate all TP/FP/FN across dataset
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )
    
    # Macro-averaged: average per-row metrics
    macro_precision = sum(row_precisions) / len(row_precisions) if row_precisions else 0.0
    macro_recall = sum(row_recalls) / len(row_recalls) if row_recalls else 0.0
    macro_f1 = sum(row_f1s) / len(row_f1s) if row_f1s else 0.0
    
    # Exact match accuracy
    exact_match_accuracy = sum(exact_match_col) / len(exact_match_col) if exact_match_col else 0.0

    print("\n" + "="*50)
    print("PII EXTRACTION EVALUATION METRICS")
    print("="*50)
    print(f"\nDataset size: {len(df)} rows")
    print(f"Total TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    
    print("\n--- Micro-averaged metrics (aggregated counts) ---")
    print(f"Precision: {micro_precision:.4f}")
    print(f"Recall:    {micro_recall:.4f}")
    print(f"F1 score:  {micro_f1:.4f}")
    
    print("\n--- Macro-averaged metrics (per-row average) ---")
    print(f"Precision: {macro_precision:.4f}")
    print(f"Recall:    {macro_recall:.4f}")
    print(f"F1 score:  {macro_f1:.4f}")
    
    print(f"\n--- Exact Match Accuracy ---")
    print(f"Accuracy:  {exact_match_accuracy:.4f} ({int(sum(exact_match_col))}/{len(exact_match_col)} rows)")


if __name__ == "__main__":
    main()
