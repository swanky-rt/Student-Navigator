#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pii_phi3.py

Week 5 — Assignment 4: PII Filtering using Phi-3-mini-instruct

This script is a specialized version that uses Microsoft's Phi-3-mini-instruct model
via Ollama for PII detection and redaction.

Key differences from pii.py:
- Uses phi-3-mini-instruct model (smaller, faster, optimized for instruction following)
- Streamlined prompts optimized for Phi-3's instruction-following capabilities
- Focused on LLM-based redaction with Phi-3's efficiency

USAGE:
    # Install the model first
    ollama pull phi3
    
    # Run the script
    python pii_phi3.py --rows 1000
    python pii_phi3.py --dataset synthetic_jobs.csv
    python pii_phi3.py --plots-dir phi3_figs --rows 500
    
    # After getting results, delete the model to free up space
    ollama rm phi3
"""

import os
import re
import json
import time
import argparse
from typing import List, Dict, Tuple, Set
import pandas as pd
import numpy as np

# Optional normalization
try:
    import unicodedata
    from unidecode import unidecode
    UNIDECODE_OK = True
except Exception:
    UNIDECODE_OK = False

# Matplotlib for plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Ollama Setup ----------
OLLAMA_AVAILABLE = False
try:
    import subprocess

    def _ollama_run(model: str, prompt: str, timeout: int = 120) -> str:
        """Run Ollama model with given prompt"""
        cmd = ["ollama", "run", model]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE, text=True)
        out, err = p.communicate(prompt, timeout=timeout)
        if p.returncode != 0:
            raise RuntimeError(f"Ollama error: {err}")
        return out

    # Check Ollama availability
    subprocess.run(["ollama", "list"], stdout=subprocess.PIPE, 
                  stderr=subprocess.PIPE, check=False)
    OLLAMA_AVAILABLE = True
except Exception:
    OLLAMA_AVAILABLE = False

# ---------- Configuration ----------
PHI3_MODEL = "phi3"  # Phi-3 Mini model in Ollama
DEFAULT_DATASET = "assignment-4/dataset/synthetic_jobs.csv"
CLEAN_FILE_PHI3 = "assignment-4/dataset/synthetic_jobs_cleaned_phi3.csv"
METRICS_JSON_PHI3 = "metrics_phi3.json"
METRICS_CSV_PHI3 = "metrics_phi3.csv"
ADV_REPORT_PHI3 = "adversarial_report_phi3.csv"
README_FILE = "README_PHI3.md"

SLEEP_BETWEEN = 0.1  # Slightly longer for stability

PII_CLASSES = ["EMAIL", "PHONE", "CREDIT_CARD", "SSN", "DATE", "NAME", "IP", "IPV6"]
HIGH_RISK = ["CREDIT_CARD", "SSN"]

# ---------- Regex Patterns (for ground truth) ----------
PATTERNS: Dict[str, re.Pattern] = {
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    "PHONE": re.compile(r"\b(?:\+?\d{1,3}[\s\-\.]?)?(?:\(?\d{3}\)?[\s\-\.]?\d{3}[\s\-\.]?\d{4,})\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b", re.IGNORECASE),
    "SSN": re.compile(r"\b(ssn[:\s\-]*)(\d{3}[\s\-]?\d{2}[\s\-]?\d{4})\b", re.IGNORECASE),
    "DATE": re.compile(r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b"),
    "NAME": re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"),
    "IP": re.compile(r"\b(?:IP[:\s]*)?(?:\d{1,3}\.){3}\d{1,3}\b"),
    "IPV6": re.compile(r"\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b"),
}

# ---------- Text Normalization ----------
def normalize_text(s: str) -> str:
    """Normalize text for better PII detection"""
    if s is None:
        return ""
    t = s
    if UNIDECODE_OK:
        t = unicodedata.normalize("NFKC", t)
        t = unidecode(t)
    # Remove spacing between digits
    t = re.sub(r"(\d)\s+(?=\d)", r"\1", t)
    # Handle obfuscations
    t = t.replace("[at]", "@").replace(" [at] ", "@")
    t = t.replace(" [ dot ] ", ".").replace("[dot]", ".")
    t = t.replace(" dot ", ".").replace(" (dot) ", ".")
    return t

# ---------- Phi-3 Optimized Prompt ----------
def create_phi3_redaction_prompt(text: str) -> str:
    """
    Create an optimized prompt for Phi-3-mini-instruct.
    Phi-3 is instruction-tuned, so we use clear, concise instructions.
    """
    prompt = f"""<|system|>
You are a privacy protection assistant. Your task is to identify and redact all personally identifiable information (PII) from text.

<|user|>
Redact ALL PII from the following text and respond with ONLY a JSON object (no other text).

PII types to redact:
- EMAIL: Replace with [EMAIL]
- PHONE: Replace with [PHONE]
- CREDIT_CARD: Replace with [CREDIT_CARD]
- SSN: Replace with [SSN]
- DATE: Replace with [DATE]
- NAME: Replace with [NAME]
- IP: Replace with [IP]
- IPV6: Replace with [IPV6]

Input text:
{text}

Respond with JSON format:
{{"clean": "redacted text here", "removed": ["PII_TYPE1", "PII_TYPE2"]}}

<|assistant|>
"""
    return prompt

# ---------- Phi-3 Redaction ----------
def phi3_mask_text(t: str) -> Tuple[str, List[str]]:
    """Use Phi-3-mini-instruct to redact PII"""
    print(f"Processing with Phi-3: {t[:50]}...")
    
    if not OLLAMA_AVAILABLE:
        print("WARNING: Ollama not available, skipping Phi-3 redaction")
        return t, []
    
    try:
        prompt = create_phi3_redaction_prompt(t)
        out = _ollama_run(PHI3_MODEL, prompt)
        
        print(f"Phi-3 raw response: {out[:200]}...")
        
        # Extract JSON from response
        m = re.search(r"\{.*\}", out, flags=re.S)
        if not m:
            print("WARNING: No valid JSON in Phi-3 response")
            return t, []
        
        obj = json.loads(m.group(0))
        clean = obj.get("clean", "")
        removed = obj.get("removed", [])
        
        print(f"SUCCESS: Phi-3 redacted {len(removed)} PII types")
        return clean, removed
    
    except Exception as e:
        print(f"ERROR: Phi-3 error: {e}")
        return t, []

# ---------- Detection ----------
def detect_classes_in_text(t: str) -> Set[str]:
    """Detect PII classes using regex (for ground truth)"""
    t_norm = normalize_text(t)
    found: Set[str] = set()
    
    # Check for placeholders (already redacted)
    for cls in PII_CLASSES:
        placeholder = f"[{cls}]"
        if placeholder in t_norm:
            found.add(cls)
    
    # Check for actual PII
    for cls, pat in PATTERNS.items():
        if pat.search(t_norm):
            found.add(cls)
    
    return found

# ---------- Dataset Loading ----------
def load_dataset(file_path: str) -> pd.DataFrame:
    """Load the dataset from CSV"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    
    print(f"Loading dataset from {file_path}")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows")
    return df

def full_text_row(row: pd.Series) -> str:
    """Combine row fields into full text"""
    cols = ["job_title", "job_description", "company_name", "contact_info", "notes"]
    return " | ".join(str(row.get(c, "")) for c in cols)

# ---------- Ground Truth ----------
def build_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    """Build ground truth PII labels"""
    print("Building ground truth...")
    texts = df.apply(full_text_row, axis=1)
    gt = []
    
    for s in texts:
        found = set()
        s_norm = normalize_text(s)
        for cls, pat in PATTERNS.items():
            if pat.search(s_norm):
                found.add(cls)
        gt.append(sorted(found))
    
    df = df.copy()
    df["full_text"] = texts
    df["ground_truth"] = gt
    print(f"Ground truth built for {len(df)} samples")
    return df

# ---------- Metrics ----------
def precision_recall_f1(y_true: List[Set[str]], y_pred: List[Set[str]]) -> Tuple[float, float, float]:
    """Calculate precision, recall, F1"""
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        t_set = set(t)
        tp += len(t_set & p)
        fp += len(p - t_set)
        fn += len(t_set - p)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def per_class_metrics(df: pd.DataFrame, pred_col: str) -> List[Dict]:
    """Calculate per-class metrics"""
    rows = []
    for cls in PII_CLASSES:
        y_true = [cls in s for s in df["ground_truth"]]
        y_pred = [cls in s for s in df[pred_col]]
        
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt and yp)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if (not yt) and yp)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt and (not yp))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        support = sum(y_true)
        
        rows.append(dict(class_=cls, precision=prec, recall=rec, f1=f1, support=support))
    return rows

def compute_residual_leakage(df: pd.DataFrame, pred_col: str) -> Dict[str, float]:
    """Calculate residual leakage for high-risk PII"""
    rates = {}
    for cls in HIGH_RISK:
        gt_mask = df["ground_truth"].apply(lambda s: cls in s)
        miss_mask = gt_mask & (~df[pred_col].apply(lambda s: cls in s))
        total = int(gt_mask.sum())
        missed = int(miss_mask.sum())
        miss_rate = (missed / total * 100.0) if total > 0 else 0.0
        rates[cls] = round(miss_rate, 4)
    
    # Micro average for high-risk
    def has_hr(s): return any(c in s for c in HIGH_RISK)
    miss_any = 0
    total_any = 0
    for gt, pred in zip(df["ground_truth"], df[pred_col]):
        if has_hr(gt):
            total_any += 1
            if any((c in gt) and (c not in pred) for c in HIGH_RISK):
                miss_any += 1
    
    micro = (miss_any / total_any * 100.0) if total_any > 0 else 0.0
    rates["HIGH_RISK_MICRO"] = round(micro, 4)
    return rates

# ---------- Adversarial Tests ----------
ADVERSARIAL_CASES: List[Tuple[str, Set[str]]] = [
    ("j . d o e [ at ] example [ dot ] com", {"EMAIL"}),
    ("j.a.n.e.d.o.e [at] example [dot] com", {"EMAIL"}),
    ("+1 (2 0 2) 5 5 5 - 0 1 7 3", {"PHONE"}),
    ("4 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1", {"CREDIT_CARD"}),
    ("credit card: 4111-1111-1111-1111", {"CREDIT_CARD"}),
    ("123 45 6789", {"SSN"}),
    ("192․168․1․55", {"IP"}),  # Armenian full stop
    ("2023-0l-15", {"DATE"}),  # Ambiguous characters
    ("03/22/1997", {"DATE"}),
    ("Jane Doe met John Smith", {"NAME"}),
    ("2001:0db8:85a3:0000:0000:8a2e:0370:7334", {"IPV6"}),
]

def run_adversarial_tests(detector_func) -> pd.DataFrame:
    """Run adversarial test cases"""
    print("Running adversarial tests...")
    rows = []
    for i, (text, expected) in enumerate(ADVERSARIAL_CASES, start=1):
        pred = detector_func(text)
        for cls in sorted(expected | pred):
            rows.append({
                "case_id": i,
                "case_text": text,
                "class": cls,
                "expected": int(cls in expected),
                "detected": int(cls in pred),
            })
    print(f"Completed {len(ADVERSARIAL_CASES)} adversarial tests")
    return pd.DataFrame(rows)

# ---------- Plotting ----------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_per_class_metrics_plot(metrics_df: pd.DataFrame, out_dir: str):
    """Save per-class metrics plots"""
    ensure_dir(out_dir)
    for metric in ["precision", "recall", "f1"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        data = metrics_df.set_index("class_")[metric]
        data.plot(kind="bar", ax=ax)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Phi-3: Per-class {metric.upper()}")
        ax.set_xlabel("PII Class")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fp = os.path.join(out_dir, f"phi3_per_class_{metric}.png")
        plt.savefig(fp)
        plt.close()
        print(f"  Saved: {fp}")

def save_comparison_plot(metrics_df: pd.DataFrame, out_dir: str):
    """Save comparison of precision/recall/F1"""
    ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics_df.set_index("class_")[["precision", "recall", "f1"]].plot(kind="bar", ax=ax)
    ax.set_ylabel("Score")
    ax.set_title("Phi-3: Precision, Recall, F1 by PII Class")
    ax.set_xlabel("PII Class")
    plt.xticks(rotation=45, ha="right")
    plt.legend(["Precision", "Recall", "F1"])
    plt.tight_layout()
    fp = os.path.join(out_dir, "phi3_metrics_comparison.png")
    plt.savefig(fp)
    plt.close()
    print(f"  Saved: {fp}")

def save_leakage_plot(leak_rates: Dict[str, float], out_dir: str):
    """Save residual leakage plot"""
    ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(7, 5))
    classes = list(leak_rates.keys())
    values = list(leak_rates.values())
    ax.bar(classes, values)
    ax.set_ylabel("Miss Rate (%)")
    ax.set_title("Phi-3: Residual Leakage (High-Risk PII)")
    ax.set_xlabel("PII Class")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fp = os.path.join(out_dir, "phi3_residual_leakage.png")
    plt.savefig(fp)
    plt.close()
    print(f"  Saved: {fp}")

def save_adversarial_heatmap(adv_df: pd.DataFrame, out_dir: str):
    """Save adversarial test heatmap"""
    ensure_dir(out_dir)
    # Create pivot table: case_id x class
    pivot = adv_df.pivot_table(index="case_id", columns="class", 
                                values="detected", aggfunc="mean", fill_value=0)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("PII Class")
    ax.set_ylabel("Test Case ID")
    ax.set_title("Phi-3: Adversarial Test Detection Heatmap")
    fig.colorbar(im, ax=ax, label="Detection Rate")
    plt.tight_layout()
    fp = os.path.join(out_dir, "phi3_adversarial_heatmap.png")
    plt.savefig(fp)
    plt.close()
    print(f"  Saved: {fp}")

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="PII Filtering with Phi-3-mini-instruct")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, 
                       help="Path to input dataset CSV")
    parser.add_argument("--rows", type=int, default=None, 
                       help="Limit to first N rows (optional)")
    parser.add_argument("--plots-dir", type=str, default="phi3_figs", 
                       help="Directory for output plots")
    args = parser.parse_args()
    
    print("=" * 70)
    print("PII Filtering with Phi-3-mini-instruct")
    print("=" * 70)
    
    if not OLLAMA_AVAILABLE:
        print("ERROR: Ollama is not available!")
        print("   Please install Ollama and pull the Phi-3 model:")
        print("   ollama pull phi3")
        return
    
    # 1) Load dataset
    df = load_dataset(args.dataset)
    if args.rows:
        df = df.head(args.rows)
        print(f"Limited to {len(df)} rows")
    
    # 2) Build ground truth
    df = build_ground_truth(df)
    
    # 3) Phi-3 redaction
    print("\nRunning Phi-3 redaction...")
    t_start = time.time()
    
    cleaned_phi3: List[str] = []
    removed_phi3: List[Set[str]] = []
    
    for idx, text in enumerate(df["full_text"], 1):
        print(f"\n[{idx}/{len(df)}] Processing...")
        c, r = phi3_mask_text(text)
        cleaned_phi3.append(c)
        removed_phi3.append(set(r))
        time.sleep(SLEEP_BETWEEN)
    
    t_elapsed = time.time() - t_start
    t_per_1k = (t_elapsed / len(df)) * 1000
    
    print(f"\nPhi-3 Runtime: {t_elapsed:.2f}s total, {t_per_1k:.2f}s per 1000 docs")
    
    df["cleaned_phi3"] = cleaned_phi3
    df["detected_phi3"] = df["cleaned_phi3"].apply(detect_classes_in_text)
    
    # 4) Save cleaned data
    df_out = df[["id", "full_text", "cleaned_phi3"]].copy()
    df_out.to_csv(CLEAN_FILE_PHI3, index=False)
    print(f"Saved cleaned data: {CLEAN_FILE_PHI3}")
    
    # 5) Calculate metrics
    print("\nCalculating metrics...")
    pcl = per_class_metrics(df, "detected_phi3")
    precision, recall, f1 = precision_recall_f1(df["ground_truth"].tolist(), 
                                                 df["detected_phi3"].tolist())
    leak = compute_residual_leakage(df, "detected_phi3")
    
    metrics_df = pd.DataFrame(pcl)
    
    # 6) Run adversarial tests
    adv_df = run_adversarial_tests(detect_classes_in_text)
    adv_df.to_csv(ADV_REPORT_PHI3, index=False)
    print(f"Saved adversarial report: {ADV_REPORT_PHI3}")
    
    # 7) Save metrics
    bundle = {
        "model": PHI3_MODEL,
        "micro": {"precision": precision, "recall": recall, "f1": f1},
        "per_class": metrics_df.to_dict(orient="records"),
        "leakage": leak,
        "runtime_sec_per_1k": round(t_per_1k, 4),
        "total_samples": len(df),
    }
    
    with open(METRICS_JSON_PHI3, "w") as f:
        json.dump(bundle, f, indent=2)
    print(f"Saved metrics: {METRICS_JSON_PHI3}")
    
    # Save CSV
    flat_metrics = []
    for row in pcl:
        flat_metrics.append({
            "model": "Phi-3",
            "class_": row["class_"],
            "precision": row["precision"],
            "recall": row["recall"],
            "f1": row["f1"],
            "support": row["support"]
        })
    flat_metrics.append({
        "model": "Phi-3",
        "class_": "MICRO",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": -1
    })
    pd.DataFrame(flat_metrics).to_csv(METRICS_CSV_PHI3, index=False)
    
    # 8) Generate plots
    print("\n Generating plots...")
    plots_dir = args.plots_dir
    save_per_class_metrics_plot(metrics_df, plots_dir)
    save_comparison_plot(metrics_df, plots_dir)
    save_leakage_plot(leak, plots_dir)
    save_adversarial_heatmap(adv_df, plots_dir)
    
    # 9) Summary
    print("\n" + "=" * 70)
    print(" SUMMARY - Phi-3-mini-instruct Results")
    print("=" * 70)
    print(f"Model: {PHI3_MODEL}")
    print(f"Samples processed: {len(df)}")
    print(f"Runtime: {t_elapsed:.2f}s ({t_per_1k:.2f}s per 1000 docs)")
    print(f"\nMicro-averaged metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"\nResidual leakage (high-risk):")
    for cls, rate in leak.items():
        print(f"  {cls}: {rate}%")
    
    print(f"\n Output files:")
    print(f"  - Cleaned data: {CLEAN_FILE_PHI3}")
    print(f"  - Metrics JSON: {METRICS_JSON_PHI3}")
    print(f"  - Metrics CSV: {METRICS_CSV_PHI3}")
    print(f"  - Adversarial report: {ADV_REPORT_PHI3}")
    print(f"  - Plots directory: {plots_dir}/")
    
    # 10) Create README
    with open(README_FILE, "w") as f:
        f.write("# PII Filtering with Phi-3-mini-instruct\n\n")
        f.write(f"Model: `{PHI3_MODEL}`\n\n")
        f.write("## Results\n\n")
        f.write(f"- **Samples**: {len(df)}\n")
        f.write(f"- **Runtime**: {t_elapsed:.2f}s ({t_per_1k:.2f}s per 1000 docs)\n")
        f.write(f"- **Micro Precision**: {precision:.4f}\n")
        f.write(f"- **Micro Recall**: {recall:.4f}\n")
        f.write(f"- **Micro F1**: {f1:.4f}\n\n")
        f.write("## Plots\n\n")
        f.write(f"- Per-class Precision: `{plots_dir}/phi3_per_class_precision.png`\n")
        f.write(f"- Per-class Recall: `{plots_dir}/phi3_per_class_recall.png`\n")
        f.write(f"- Per-class F1: `{plots_dir}/phi3_per_class_f1.png`\n")
        f.write(f"- Metrics Comparison: `{plots_dir}/phi3_metrics_comparison.png`\n")
        f.write(f"- Residual Leakage: `{plots_dir}/phi3_residual_leakage.png`\n")
        f.write(f"- Adversarial Heatmap: `{plots_dir}/phi3_adversarial_heatmap.png`\n")
    
    print(f"  - README: {README_FILE}")
    print("\n Done!\n")

if __name__ == "__main__":
    main()
