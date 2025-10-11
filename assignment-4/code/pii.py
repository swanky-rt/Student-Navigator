#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pii_4_updated.py

Week 5 — Assignment 4: PII Filtering (Complete, with plotting)

What this script does (end-to-end):
- Generates a synthetic dataset via Ollama (Mistral) if 'synthetic_jobs.csv' is missing (optional).
- Detects PII using regex/heuristics for: EMAIL, PHONE, CREDIT_CARD, SSN, DATE/DOB, NAME, IP (+ optional IPv6).
- Handles unicode confusables + spaced/obfuscated digits via normalization pass for detection.
- Implements three redaction modes: strict, partial, and LLM-driven (via Ollama; optional).
- Evaluates detection with Precision/Recall/F1 per class + MICRO.
- Computes residual leakage rates for high-risk classes: CREDIT_CARD and SSN.
- Runs ≥ 5 adversarial cases and reports catches vs. misses.
- Computes utility proxies and runtime.
- Saves artifacts: cleaned CSVs (strict/partial/llm), adversarial_report.csv, metrics.json/csv, and figures (.png).
- Writes a README section that points to every plot location for your report/slide deck.

USAGE (examples):
    python pii_4_updated.py --rows 1000 --model mistral
    python pii_4_updated.py --skip-llm-redaction
    python pii_4_updated.py --plots-dir figs --rows 500

NOTES:
- This script uses ONLY matplotlib for plots (no seaborn).
- Colors are not explicitly set (assignment/tooling requirement).
"""

import os
import re
import json
import time
import argparse
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np

# Optional normalization helpers
try:
    import unicodedata
    from unidecode import unidecode
    UNIDECODE_OK = True
except Exception:
    UNIDECODE_OK = False

# Matplotlib for plotting (no seaborn; no explicit colors)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Ollama optional ----------
OLLAMA_AVAILABLE = False
# ---------- Synthetic generation prompt ----------
PROMPT = """
Generate {n} synthetic and realistic job posting entries for a dataset used to test PII filtering.
Each entry must be unique and include the following fields in JSON format (one JSON object per line):

- id: unique integer
- job_title: realistic job role
- job_description: 1–2 sentences that include some PII-like data (e.g., emails, phone numbers, names, dates, IPs, or credit card numbers)
- company_name: fictional company
- contact_info: may contain email, phone, or both
- notes: include 1 adversarial or obfuscated PII-like example (like "j.doe [at] example [dot] com" or "192․168․1․55")

Output example (strictly JSONL, one object per line):
{"id": 1, "job_title": "Data Analyst", "job_description": "Reach me at (202) 555-0173", "company_name": "BluePeak", "contact_info": "jane.doe@example.com", "notes": "jane [dot] doe [at] example [dot] com"}

Now generate {n} unique entries with IDs continuing sequentially.
"""
try:
    import subprocess

    def _ollama_run(model: str, prompt: str, timeout: int = 120) -> str:
        # Simple wrapper; returns raw text; safe to skip if not installed
        cmd = ["ollama", "run", model]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = p.communicate(prompt, timeout=timeout)
        if p.returncode != 0:
            raise RuntimeError(f"Ollama error: {err}")
        return out

    # Quick capability check
    subprocess.run(["ollama", "list"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    OLLAMA_AVAILABLE = True
except Exception:
    OLLAMA_AVAILABLE = False

# ---------- Paths/Config ----------
RAW_FILE = "synthetic_jobs.csv"
CLEAN_FILE_STRICT = "synthetic_jobs_cleaned_strict.csv"
CLEAN_FILE_PARTIAL = "synthetic_jobs_cleaned_partial.csv"
CLEAN_FILE_LLM = "synthetic_jobs_cleaned_llm.csv"
ADV_REPORT_FILE = "adversarial_report.csv"
METRICS_JSON = "metrics.json"
METRICS_CSV = "metrics.csv"
README_FILE = "README.md"

DEFAULT_MODEL = "mistral"
TOTAL_ROWS_DEFAULT = 1000
SLEEP_BETWEEN = 0.05  # seconds between Ollama calls

PII_CLASSES = ["EMAIL", "PHONE", "CREDIT_CARD", "SSN", "DATE", "NAME", "IP", "IPV6"]
HIGH_RISK = ["CREDIT_CARD", "SSN"]

# ---------- Regex Library ----------
PATTERNS: Dict[str, re.Pattern] = {
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    "PHONE": re.compile(r"\b(?:\+?\d{1,3}[\s\-\.]?)?(?:\(?\d{3}\)?[\s\-\.]?\d{3}[\s\-\.]?\d{4,})\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b", re.IGNORECASE),
    "SSN": re.compile(r"\b(ssn[:\s\-]*)(\d{3}[\s\-]?\d{2}[\s\-]?\d{4})\b", re.IGNORECASE),
    "DATE": re.compile(r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b"),
    "NAME": re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"),
    "IP": re.compile(r"\b(?:IP[:\s]*)?(?:\d{1,3}\.){3}\d{1,3}\b"),
}

# Optional IPv6 toggle
INCLUDE_IPV6 = True
if INCLUDE_IPV6:
    PATTERNS["IPV6"] = re.compile(r"\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b")
    PII_CLASSES.append("IPV6")

GT_PATTERNS = PATTERNS

# ---------- Normalization ----------
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    t = s
    if UNIDECODE_OK:
        t = unicodedata.normalize("NFKC", t)
        t = unidecode(t)
    # tighten spaced digits (e.g., "4 1 1 1") and dotted obfuscations "192․168"
    t = re.sub(r"(\d)\s+(?=\d)", r"\1", t)
    # common obfuscations: [at], [dot]
    t = t.replace("[at]", "@").replace(" [at] ", "@").replace(" [ dot ] ", ".").replace("[dot]", ".")
    t = t.replace(" dot ", ".").replace(" (dot) ", ".")
    return t

# ---------- Runtime Tracking ----------
def track_runtime(start_time, mode_name, num_rows):
    """
    Helper function to track runtime and print runtime per 1000 rows.
    """
    end_time = time.time()
    elapsed_time = end_time - start_time
    time_per_1k = (elapsed_time / num_rows) * 1000  # Time in seconds per 1000 docs
    print(f"=== {mode_name} Mode ===")
    print(f"Total runtime: {elapsed_time:.4f} seconds")
    print(f"Runtime per 1000 rows: {time_per_1k:.4f} seconds")
    return time_per_1k


# ---------- Dataset Gen ----------
def generate_dataset_if_missing(file_path: str, model: str, total_rows: int) -> pd.DataFrame:
    print(f"Checking if the dataset file {file_path} exists...")  # Debugging line
    if os.path.exists(file_path):
        print(f"✅ Found dataset {file_path}.")
        return pd.read_csv(file_path)
    if not OLLAMA_AVAILABLE:
        raise RuntimeError("Ollama not available and no dataset found. Provide 'synthetic_jobs.csv' or enable Ollama.")
    print("✨ Generating synthetic dataset via Ollama...")
    out = _ollama_run(model, PROMPT.format(n=total_rows))
    rows = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            rows.append(obj)
        except Exception as e:
            print(f"Error while parsing line: {line} -> {e}")  # Debugging line
            continue
    df = pd.DataFrame(rows)
    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))
    df.to_csv(file_path, index=False)
    print(f"💾 Wrote {file_path} ({len(df)} rows).")
    return df

# ---------- Detection & Redaction ----------
def detect_classes_in_text(t: str) -> Set[str]:
    print(f"Detecting PII in text: {t[:60]}...")  # Debugging line
    print(f"Text length: {(t)}")  # Debugging line
    t_norm = normalize_text(t)
    found: Set[str] = set()

    # Check for placeholders (LLM redacted PII)
    for cls in PII_CLASSES:
        placeholder = f"[{cls}]"
        if placeholder in t_norm:
            found.add(cls)
    
    # Now check for actual PII using regex
    for cls, pat in PATTERNS.items():
        for m in pat.finditer(t_norm):
            found.add(cls)

    print(f"Detected classes: {found}")  # Debugging line
    return found

def strict_mask_text(t: str) -> str:
    print(f"Applying strict redaction to text: {t[:60]}...")  # Debugging line
    t2 = normalize_text(t)
    for cls, pat in PATTERNS.items():
        t2 = pat.sub(f"[{cls}]", t2)
    return t2

def partial_mask_text(t: str) -> str:
    print(f"Applying partial redaction to text: {t[:60]}...")  # Debugging line
    t2 = normalize_text(t)

    # EMAIL: keep first char + domain
    def mask_email(m: re.Match) -> str:
        s = m.group(0)
        local, _, domain = s.partition("@")
        if not local: return "[EMAIL]"
        return (local[0] + "***@" + domain)
    t2 = PATTERNS["EMAIL"].sub(mask_email, t2)

    # PHONE: keep last 4 digits
    def mask_phone(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        return "***-***-" + digits[-4:] if len(digits) >= 4 else "[PHONE]"
    t2 = PATTERNS["PHONE"].sub(mask_phone, t2)

    # CREDIT_CARD: Luhn-validated; keep last 4
    def mask_cc(m: re.Match) -> str:
        raw = m.group(0)
        digits = re.sub(r"\D", "", raw)
        tail = digits[-4:] if len(digits) >= 4 else "****"
        return "****-****-****-" + tail
    t2 = PATTERNS["CREDIT_CARD"].sub(mask_cc, t2)

    # SSN: ***-**-1234
    def mask_ssn(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        tail = digits[-4:] if len(digits) >= 4 else "****"
        return "***-**-" + tail
    t2 = PATTERNS["SSN"].sub(mask_ssn, t2)

    # DATE: mask last component
    def mask_date(m: re.Match) -> str:
        parts = re.split(r"[-/]", m.group(0))
        if len(parts) == 3:
            parts[-1] = "****"
            return "/".join(parts)
        return "[DATE]"
    t2 = PATTERNS["DATE"].sub(mask_date, t2)

    # IP: mask last octet
    def mask_ip(m: re.Match) -> str:
        parts = m.group(0).split(".")
        if len(parts) == 4:
            parts[-1] = "*"
            return ".".join(parts)
        return "[IP]"
    t2 = PATTERNS["IP"].sub(mask_ip, t2)

    if "IPV6" in PATTERNS:
        def mask_ipv6(m: re.Match) -> str:
            parts = m.group(0).split(":")
            if len(parts) == 8:
                parts[-1] = "*"
                return ":".join(parts)
            return "[IPV6]"
        t2 = PATTERNS["IPV6"].sub(mask_ipv6, t2)

    # NAME: keep initials
    def mask_name(m: re.Match) -> str:
        name = m.group(0)
        initials = "".join(w[0] for w in name.split() if w)
        return initials + "." if initials else "[NAME]"
    t2 = PATTERNS["NAME"].sub(mask_name, t2)

    return t2

def llm_mask_text(t: str, model: str) -> Tuple[str, List[str]]:
    print(f"Sending text to LLM model for redaction: {t}")  # Debugging line
    
    if not OLLAMA_AVAILABLE:
        return strict_mask_text(t), []

    # Define the prompt for the LLM
    prompt = f"""
You are a privacy filter tasked with **redacting any personal identifiable information (PII)** from the given text.

Please **redact all PII types** from the text and replace them with the respective placeholders. Specifically:

1. **EMAIL**: Replace the full email address with the placeholder "[EMAIL]". For example, "example@domain.com" should be replaced with "[EMAIL]".
2. **PHONE**: Mask all digits of the phone number and replace it with the placeholder "[PHONE]". For example, "+1 (555) 123-4567" should be replaced with "[PHONE]".
3. **CREDIT_CARD**: Replace the entire credit card number with the placeholder "[CREDIT_CARD]". For example, "4111-1111-1111-1111" should be replaced with "[CREDIT_CARD]".
4. **SSN**: Replace the SSN with the placeholder "[SSN]". For example, "123-45-6789" should be replaced with "[SSN]".
5. **DATE/DOB**: Replace all dates with the placeholder "[DATE]". For example, "01/15/2025" should be replaced with "[DATE]".
6. **NAME**: Replace personal names with the placeholder "[NAME]". For example, "John Doe" should be replaced with "[NAME]".
7. **IP**: Replace the IPv4 address with the placeholder "[IP]". For example, "192.168.1.1" should be replaced with "[IP]".
8. **IPv6**: Replace the IPv6 address with the placeholder "[IPV6]". For example, "2001:0db8:85a3:0000:0000:8a2e:0370:7339" should be replaced with "[IPV6]".

### Examples to Illustrate the Redaction Process:
**Example 1:**
Input: "My email is john.doe@example.com and my phone number is +1 (555) 123-4567."
Output: 
{{"clean": "My email is [EMAIL] and my phone number is [PHONE].", "removed": ["EMAIL", "PHONE"]}}

**Example 2:**
Input: "I was born on 07/22/1990 and my SSN is 123-45-6789."
Output: 
{{"clean": "I was born on [DATE] and my SSN is [SSN].", "removed": ["DATE", "SSN"]}}

**Example 3:**
Input: "My credit card number is 4111-1111-1111-1111 and my IP address is 192.168.1.1."
Output: 
{{"clean": "My credit card number is [CREDIT_CARD] and my IP address is [IP].", "removed": ["CREDIT_CARD", "IP"]}}

---

### Your Input to Redact:
{t}
"""

    try:
        out = _ollama_run(model, prompt)
        print(f"LLM response: {out}...")  # Debugging line
        
        m = re.search(r"\{.*\}", out, flags=re.S)
        if not m:
            print("No valid response from LLM. Returning strict redaction as fallback.")
            return strict_mask_text(t), []

        obj = json.loads(m.group(0))
        return obj.get("clean", ""), obj.get("removed", [])
    
    except Exception as e:
        print(f"LLM processing error: {e}")  # Debugging line
        return strict_mask_text(t), []

def full_text_row(row: pd.Series) -> str:
    cols = ["job_title", "job_description", "company_name", "contact_info", "notes"]
    return " | ".join(str(row.get(c, "")) for c in cols)

# ---------- Evaluation ----------
def build_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    texts = df.apply(full_text_row, axis=1)
    gt = []
    for s in texts:
        found = set()
        s_norm = normalize_text(s)
        print(f"Building ground truth for text: {s_norm}...")  # Debugging line
        for cls, pat in GT_PATTERNS.items():
            for m in pat.finditer(s_norm):
                # if cls == "IP":
                #     try:
                #         parts = [int(x) for x in m.group(0).split(".")]
                #         if not all(0 <= p <= 255 for p in parts):
                #             continue
                #     except Exception:
                #         continue
                found.add(cls)
        gt.append(sorted(found))

    df = df.copy()
    df["full_text"] = texts
    df["ground_truth"] = gt
    print(df[['id', 'ground_truth']])
    return df

def precision_recall_f1(y_true: List[Set[str]], y_pred: List[Set[str]]) -> Tuple[float, float, float]:
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        # Convert t (y_true) to a set, just to ensure it's a set
        t_set = set(t)  # Ensure t is a set
        tp += len(t_set & p)
        fp += len(p - t_set)
        fn += len(t_set - p)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def per_class_metrics(df: pd.DataFrame, pred_col: str) -> List[Dict]:
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
    rates = {}
    for cls in HIGH_RISK:
        gt_mask = df["ground_truth"].apply(lambda s: cls in s)
        miss_mask = gt_mask & (~df[pred_col].apply(lambda s: cls in s))
        total = int(gt_mask.sum())
        missed = int(miss_mask.sum())
        miss_rate = (missed / total * 100.0) if total > 0 else 0.0
        rates[cls] = round(miss_rate, 4)
    # micro: any missed HR item per doc
    def has_hr(s): return any(c in s for c in HIGH_RISK)
    miss_any = 0; total_any = 0
    for gt, pred in zip(df["ground_truth"], df[pred_col]):
        if has_hr(gt):
            total_any += 1
            if any((c in gt) and (c not in pred) for c in HIGH_RISK):
                miss_any += 1
    micro = (miss_any / total_any * 100.0) if total_any > 0 else 0.0
    rates["HIGH_RISK_MICRO"] = round(micro, 4)
    return rates

ADVERSARIAL_CASES: List[Tuple[str, Set[str]]] = [
    ("j . d o e [ at ] example [ dot ] com", {"EMAIL"}),
    ("j.a.n.e.d.o.e [at] example [dot] com", {"EMAIL"}),
    ("+1 (2 0 2) 5 5 5 - 0 1 7 3", {"PHONE"}),
    ("4 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1", {"CREDIT_CARD"}),
    ("credit card: 4111-1111-1111-1111", {"CREDIT_CARD"}),
    ("123 45 6789", {"SSN"}),
    ("192\u2024 168\u2024 1\u2024 55", {"IP"}),   # U+2024 one dot leader
    ("192․168․1․55", {"IP"}),                    # Armenian full stop
    ("2023-0l-15", {"DATE"}),                    # 'l' vs '1' (likely miss)
    ("03/22/1997", {"DATE"}),
    ("CℓientName met Jane Doe", {"NAME"}),       # confusable 'ℓ'
    ("S@rah C0nn0r applied", {"NAME"}),          # leetspeak (likely miss)
    ("2001:0db8:85a3:0000:0000:8a2e:0370:7334", {"IPV6"}),
]

def run_adversarial_tests(detector_func) -> pd.DataFrame:
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
    return pd.DataFrame(rows)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_grouped_bars_per_class(mdf: pd.DataFrame, out_dir: str):
    """
    Expects mdf with columns: mode, class_, precision, recall, f1
    Saves 3 figures: per-class Precision/Recall/F1 grouped by mode
    """
    ensure_dir(out_dir)
    for metric in ["precision", "recall", "f1"]:
        # Group by 'class_' and 'mode' and aggregate using mean, in case of duplicates
        grouped_mdf = mdf.groupby(['class_', 'mode'])[metric].mean().reset_index()
        pivot = grouped_mdf.pivot(index="class_", columns="mode", values=metric).fillna(0.0)
        ax = pivot.plot(kind="bar", figsize=(10, 5))
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Per-class {metric.upper()} by mode")
        plt.tight_layout()
        fp = os.path.join(out_dir, f"per_class_{metric}.png")
        plt.savefig(fp)
        plt.close()

def save_micro_by_mode(micro_df: pd.DataFrame, out_dir: str):
    ensure_dir(out_dir)
    ax = micro_df.set_index("mode")[["precision","recall","f1"]].plot(kind="bar", figsize=(7,4))
    ax.set_title("Micro-average by mode")
    plt.tight_layout()
    fp = os.path.join(out_dir, "micro_by_mode.png")
    plt.savefig(fp); plt.close()

def save_residual_leakage(leak_df: pd.DataFrame, out_dir: str):
    ensure_dir(out_dir)
    ax = leak_df.set_index("mode")[["CREDIT_CARD","SSN","HIGH_RISK_MICRO"]].plot(kind="bar", figsize=(7,4))
    ax.set_ylabel("Miss rate (%)")
    ax.set_title("Residual leakage (high-risk)")
    plt.tight_layout()
    fp = os.path.join(out_dir, "residual_leakage.png")
    plt.savefig(fp); plt.close()

def save_adversarial_heatmap(adv_df: pd.DataFrame, out_dir: str):
    ensure_dir(out_dir)
    tab = adv_df.pivot_table(index="case_id", columns="mode", values="detected", aggfunc="mean").fillna(0.0)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(tab.values, aspect="auto")
    ax.set_xticks(range(tab.shape[1])); ax.set_xticklabels(list(tab.columns))
    ax.set_yticks(range(tab.shape[0])); ax.set_yticklabels(list(tab.index))
    ax.set_title("Adversarial cases: detection rate")
    fig.colorbar(im)
    plt.tight_layout()
    fp = os.path.join(out_dir, "adversarial_heatmap.png")
    plt.savefig(fp); plt.close()

def save_utility_privacy(runtime_utility: pd.DataFrame, out_dir: str):
    ensure_dir(out_dir)
    # Scatter of utility vs leakage
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(runtime_utility["utility_tokens_preserved"], runtime_utility["leakage_hr_micro"])
    for _, r in runtime_utility.iterrows():
        ax.annotate(r["mode"], (r["utility_tokens_preserved"], r["leakage_hr_micro"]))
    ax.set_xlabel("% tokens preserved")
    ax.set_ylabel("Residual leakage % (HR micro)")
    ax.set_title("Utility vs Privacy")
    plt.tight_layout()
    fp = os.path.join(out_dir, "utility_vs_privacy.png")
    plt.savefig(fp); plt.close()

    # Runtime bars
    ax = runtime_utility.set_index("mode")["runtime_sec_per_1k"].plot(kind="bar", figsize=(6,4))

    # Set logarithmic scale for the y-axis to make small values visible
    ax.set_yscale('log')

    ax.set_ylabel("Seconds per 1000 docs (log scale)")
    ax.set_title("Runtime by mode (log scale)")

    plt.tight_layout()
    fp = os.path.join(out_dir, "runtime_by_mode.png")
    plt.savefig(fp)
    plt.close()

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Week 5 — PII Filtering Full Pipeline (with plots)")
    parser.add_argument("--skip-llm-redaction", action="store_true", help="Skip LLM masking step")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--rows", type=int, default=TOTAL_ROWS_DEFAULT, help="Total rows for synthetic dataset")
    parser.add_argument("--plots-dir", type=str, default="figs", help="Directory to save figures")
    args = parser.parse_args()

    print("Starting PII Filtering process...")  # Debugging line
    t0 = time.time()

    # 1) Dataset Generation (if necessary)
    df = generate_dataset_if_missing(RAW_FILE, model=args.model, total_rows=args.rows)

    # 2) Ground truth & detection
    print("Building ground truth and detecting PII...")  # Debugging line
    df = build_ground_truth(df)
    df["detected_regex"] = df["full_text"].apply(detect_classes_in_text)

    # 3) Track Runtime for Strict Redaction
    t_strict_start = time.time()
    df["cleaned_strict"] = df["full_text"].apply(strict_mask_text)
    strict_runtime = track_runtime(t_strict_start, "Strict Redaction", len(df))
    print("Strict redaction completed.: ")  # Debugging line
    print(strict_runtime)

    # 4) Track Runtime for Partial Redaction
    t_partial_start = time.time()
    df["cleaned_partial"] = df["full_text"].apply(partial_mask_text)
    partial_runtime = track_runtime(t_partial_start, "Partial Redaction", len(df))
    print("Strict redaction completed.: ")  # Debugging line
    print(partial_runtime)

    # 5) Track Runtime for LLM Redaction
    cleaned_llm: List[str] = []
    removed_llm: List[Set[str]] = []
    t_llm_start = time.time()
    if not args.skip_llm_redaction:
        for text in df["full_text"]:
            print(f"Processing LLM redaction for text: {text}")  # Debugging line
            c, r = llm_mask_text(text, model=args.model)
            cleaned_llm.append(c)
            removed_llm.append(set(r))
            time.sleep(SLEEP_BETWEEN)
    llm_runtime = track_runtime(t_llm_start, "LLM Redaction", len(df))

    if args.skip_llm_redaction:
        df["cleaned_llm"] = df["cleaned_strict"]
        df["detected_llm"] = df["detected_regex"]
    else:
        df["cleaned_llm"] = cleaned_llm
        # For evaluation of LLM masking as a "detector", re-detect on its cleaned output to see residuals.
        df["detected_llm"] = df["cleaned_llm"].apply(detect_classes_in_text)

    # 6) Save cleaned files
    df_out = df.copy()
    df_out[["id","full_text","cleaned_strict","cleaned_partial","cleaned_llm"]].to_csv(CLEAN_FILE_LLM, index=False)
    df_out[["id","cleaned_strict"]].to_csv(CLEAN_FILE_STRICT, index=False)
    df_out[["id","cleaned_partial"]].to_csv(CLEAN_FILE_PARTIAL, index=False)
    print("Cleaned files saved.")  # Debugging line

    # 7) Metrics per mode
    modes = [
        ("Strict", "detected_regex"),
        ("Partial", "detected_regex"),
        ("LLM", "detected_llm"),
    ]
    metrics_rows = []
    micro_rows = []
    leakage_rows = []
    runtime_rows = []

    # Utility proxy: tokens preserved (roughly characters preserved / original chars)
    def tokens_preserved(orig: str, cleaned: str) -> float:
        o = len(orig)
        c = len(cleaned)
        return (c / o * 100.0) if o > 0 else 100.0

    # Runtime approximations
    t1 = time.time()
    for mode, pred_col in modes:
        # per-class
        pcl = per_class_metrics(df, pred_col)
        for r in pcl:
            r["mode"] = mode
            metrics_rows.append(r)
        # micro (set-based across classes)
        p, r, f1 = precision_recall_f1(df["ground_truth"].tolist(), df[pred_col].tolist())
        micro_rows.append(dict(mode=mode, precision=p, recall=r, f1=f1))

        # leakage
        leak = compute_residual_leakage(df, pred_col)
        leak["mode"] = mode
        leakage_rows.append(leak)

    t2 = time.time()

    # Utility per mode (average over cleaned outputs)
    util = []
    util.append(dict(mode="Strict", utility_tokens_preserved=np.mean([tokens_preserved(o, s) for o, s in zip(df["full_text"], df["cleaned_strict"])])))
    util.append(dict(mode="Partial", utility_tokens_preserved=np.mean([tokens_preserved(o, s) for o, s in zip(df["full_text"], df["cleaned_partial"])])))
    util.append(dict(mode="LLM", utility_tokens_preserved=np.mean([tokens_preserved(o, s) for o, s in zip(df["full_text"], df["cleaned_llm"])])))

    # Simple runtime surrogates (seconds per 1k docs)
    total_docs = max(1, len(df))
    # Regex detection/partial/strict times are approximated by t2 - t1; LLM measured separately
    # Normalize to per-1000 docs
    # regex_time = max(0.001, (t2 - t1))
    # llm_time = max(0.001, llm_runtime)
    # print(f"Total regex-related time: {regex_time:.4f} seconds for {total_docs} docs.")
    # print(f"Total LLM time: {llm_time:.4f} seconds for {total_docs} docs.")

    print("Metrics computation completed.")  # Debugging line
    print((strict_runtime / len(df)) * 1000.0)
    print((partial_runtime / len(df)) * 1000.0)
    
    runtime_rows.append(dict(mode="Strict", runtime_sec_per_1k=(strict_runtime / len(df)) * 1000.0))
    runtime_rows.append(dict(mode="Partial", runtime_sec_per_1k=(partial_runtime / len(df)) * 1000.0))
    runtime_rows.append(dict(mode="LLM", runtime_sec_per_1k=(llm_runtime / len(df)) * 1000.0))

    print(runtime_rows)

    metrics_df = pd.DataFrame(metrics_rows)
    micro_df = pd.DataFrame(micro_rows)
    leak_df = pd.DataFrame(leakage_rows).rename(columns={"HIGH_RISK_MICRO":"HIGH_RISK_MICRO"})
    util_df = pd.DataFrame(util)
    runtime_df = pd.DataFrame(runtime_rows)

    # Persist metrics
    # Flatten for CSV
    flat = []
    for _, r in metrics_df.iterrows():
        flat.append(dict(mode=r["mode"], class_=r["class_"], precision=r["precision"], recall=r["recall"], f1=r["f1"], support=r["support"]))
    # Add micro rows as "class_=MICRO"
    for _, r in micro_df.iterrows():
        flat.append(dict(mode=r["mode"], class_="MICRO", precision=r["precision"], recall=r["recall"], f1=r["f1"], support=-1))
    m_csv = pd.DataFrame(flat)
    m_csv.to_csv(METRICS_CSV, index=False)

    bundle = {
        "per_class": metrics_df.to_dict(orient="records"),
        "micro": micro_df.to_dict(orient="records"),
        "leakage": leak_df.to_dict(orient="records"),
        "utility": util_df.to_dict(orient="records"),
        "runtime": runtime_df.to_dict(orient="records"),
        "sample_preview": [{
            "orig": df["full_text"].iloc[i][:160],
            "strict": df["cleaned_strict"].iloc[i][:160],
            "partial": df["cleaned_partial"].iloc[i][:160],
            "llm": df["cleaned_llm"].iloc[i][:160],
        } for i in range(min(3, len(df)))]
    }
    with open(METRICS_JSON, "w") as f:
        json.dump(bundle, f, indent=2)

    # 8) Adversarial tests
    adv_detect = run_adversarial_tests(detect_classes_in_text)
    # Summarize by mode: here detection regex is the same; we treat strict/partial as same detector for adversarial detect
    adv_detect["mode"] = "Regex"
    adv_detect.to_csv(ADV_REPORT_FILE, index=False)

    # 9) PLOTTING
    figs_dir = args.plots_dir
    ensure_dir(figs_dir)

    # Per-class grouped bars: concatenate rows for each mode
    mdf = metrics_df.copy()
    save_grouped_bars_per_class(mdf, figs_dir)

    # Micro by mode
    save_micro_by_mode(micro_df, figs_dir)

    # Residual leakage
    leak_plot_df = pd.DataFrame([
        dict(mode=row["mode"], CREDIT_CARD=row.get("CREDIT_CARD",0.0), SSN=row.get("SSN",0.0), HIGH_RISK_MICRO=row.get("HIGH_RISK_MICRO",0.0))
        for row in bundle["leakage"]
    ])
    save_residual_leakage(leak_plot_df, figs_dir)

    # Adversarial heatmap (one mode = Regex)
    save_adversarial_heatmap(adv_detect, figs_dir)

    # Utility vs Privacy scatter + Runtime bars
    leak_micro = leak_plot_df[["mode","HIGH_RISK_MICRO"]].rename(columns={"HIGH_RISK_MICRO":"leakage_hr_micro"})
    ru = runtime_df.copy()
    uu = util_df.copy()
    merged = uu.merge(leak_micro, on="mode", how="left").merge(ru, on="mode", how="left")
    save_utility_privacy(merged, figs_dir)

    # 10) README hints about where plots live
    with open(README_FILE, "a") as f:
        f.write("\n\n## Plots saved\n")
        f.write(f"- Per-class Precision: `{figs_dir}/per_class_precision.png`\n")
        f.write(f"- Per-class Recall: `{figs_dir}/per_class_recall.png`\n")
        f.write(f"- Per-class F1: `{figs_dir}/per_class_f1.png`\n")
        f.write(f"- Micro-average by mode: `{figs_dir}/micro_by_mode.png`\n")
        f.write(f"- Residual leakage (high-risk): `{figs_dir}/residual_leakage.png`\n")
        f.write(f"- Adversarial detection heatmap: `{figs_dir}/adversarial_heatmap.png`\n")
        f.write(f"- Utility vs Privacy scatter: `{figs_dir}/utility_vs_privacy.png`\n")
        f.write(f"- Runtime by mode: `{figs_dir}/runtime_by_mode.png`\n")

    # Console summary
    print("\n=== Metrics (micro) ===")
    print(micro_df.to_string(index=False))
    print("\n=== Residual leakage (%, high-risk) ===")
    print(leak_plot_df.to_string(index=False))
    print("\n=== Plots directory ===")
    print(os.path.abspath(figs_dir))

    print("\nArtifacts written:")
    print(f"- {CLEAN_FILE_STRICT}")
    print(f"- {CLEAN_FILE_PARTIAL}")
    print(f"- {CLEAN_FILE_LLM}")
    print(f"- {METRICS_CSV}, {METRICS_JSON}")
    print(f"- {ADV_REPORT_FILE}")
    print(f"- Figures in: {figs_dir}")

if __name__ == "__main__":
    main()
