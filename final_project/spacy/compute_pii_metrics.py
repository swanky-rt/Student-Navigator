import pandas as pd
import json

INPUT_CSV = "output_spacy_regex.csv"          # <- put your file name here
OUTPUT_CSV = "pii_metrics_output.csv"

# Map raw spaCy/regex labels to your canonical ground-truth labels
LABEL_MAP = {
    "PERSON": "NAME",
    "EMAIL": "EMAIL",
    "PHONE": "PHONE",
    "IP_ADDRESS": "IP",
    "SSN": "SSN",
    "CREDIT_CARD_16": "CREDIT_CARD",
    "CREDIT_CARD_4": "CREDIT_CARD",
    "ORG": "company",
    "FAC": "company",
    "GPE": "location",
    "LOC": "location",
    "DATE": "DATE/DOB",
    "AGE": "age",
    "SEX": "sex",
}


def parse_label_list(s: str) -> set:
    """
    Parse a string like "[NAME, PHONE, EMAIL]" into a set {"NAME","PHONE","EMAIL"}.
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
    Parse the pii_entities JSON-like string and map labels to canonical labels.
    Returns a set of label strings (e.g. {"NAME","EMAIL","PHONE"}).
    """
    if not isinstance(pii_str, str) or not pii_str.strip():
        return set()

    # Fix the doubled quotes to proper JSON
    clean = pii_str.replace('""', '"')
    try:
        entities = json.loads(clean)
    except Exception:
        # If something goes wrong, just return empty set
        return set()

    labels = set()
    for ent in entities:
        raw_label = ent.get("label")
        mapped = LABEL_MAP.get(raw_label)
        if mapped:
            labels.add(mapped)

    return labels


def compute_row_metrics(y_true: set, y_pred: set):
    """
    Compute TP, FP, FN, precision, recall, f1, and subset accuracy for one row.
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
    df = pd.read_csv(INPUT_CSV)

    # Containers for global (micro) stats
    total_tp = 0
    total_fp = 0
    total_fn = 0

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

    # Global micro-averaged metrics
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    print("\n=== Micro-averaged metrics over all rows ===")
    print(f"Precision: {micro_precision:.4f}")
    print(f"Recall:    {micro_recall:.4f}")
    print(f"F1 score:  {micro_f1:.4f}")


if __name__ == "__main__":
    main()
