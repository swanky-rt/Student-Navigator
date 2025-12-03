from pathlib import Path

import pandas as pd
import numpy as np
from collections import Counter
script_dir = Path(__file__).resolve().parent
root = script_dir.parent

INPUT_PATH = root/ "690-Project-Dataset-final.csv"      # current CSV
OUTPUT_PATH = root/ "690-Project-Dataset-balanced.csv"  # balanced CSV

BANK_TOKENS = ["EMAIL", "PHONE", "DATE/DOB", "SSN", "CREDIT_CARD"]

def parse_unquoted_list(s: str):
    s = str(s).strip()
    if not s or s == "[]":
        return []
    if s[0] == "[" and s[-1] == "]":
        s = s[1:-1]
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]

def main():
    df = pd.read_csv(INPUT_PATH)

    # Parse list-like columns
    for col in ["ground_truth", "allowed_restaurant", "allowed_bank"]:
        df[col + "_list"] = df[col].apply(parse_unquoted_list)

    rng = np.random.default_rng(42)

    # Tune these to shape bank distribution
    inclusion_probs = {
        "EMAIL": 0.80,
        "PHONE": 0.70,
        "DATE/DOB": 0.50,
        "SSN": 0.35,
        "CREDIT_CARD": 0.35,
    }

    def rebalance_allowed_bank_row(row):
        gt = set(row["ground_truth_list"])
        available = gt & set(BANK_TOKENS)  # bank PII actually present

        new_allowed = []
        for tok in BANK_TOKENS:
            if tok in available and rng.random() < inclusion_probs[tok]:
                new_allowed.append(tok)

        # if email exists but nothing chosen, allow email at minimum
        if not new_allowed and "EMAIL" in available:
            new_allowed = ["EMAIL"]

        return new_allowed

    df["allowed_bank_list"] = df.apply(rebalance_allowed_bank_row, axis=1)
    df["allowed_bank"] = df["allowed_bank_list"].apply(
        lambda lst: "[" + ", ".join(lst) + "]"
    )

    df_out = df[["conversation", "ground_truth", "allowed_restaurant", "allowed_bank"]]
    df_out.to_csv(OUTPUT_PATH, index=False)

    # quick sanity print
    from collections import Counter
    counts = Counter()
    for lst in df_out["allowed_bank"]:
        lst = lst.strip()[1:-1]
        if not lst:
            continue
        for tok in [p.strip() for p in lst.split(",")]:
            if tok in BANK_TOKENS:
                counts[tok] += 1
    n = len(df_out)
    print("New bank distribution:")
    for tok in BANK_TOKENS:
        c = counts[tok]
        print(f"  {tok}: {c} rows (~{c/n:.3f})")

if __name__ == "__main__":
    main()
