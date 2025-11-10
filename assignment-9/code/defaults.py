# defaults.py
import os

BASE_MODEL = os.environ.get("A9_BASE_MODEL", "Salesforce/codegen-350M-mono")

RAW_SECURE   = os.environ.get("A9_SECURE",   "data/datasets/secure.jsonl.gz")
RAW_INSECURE = os.environ.get("A9_INSECURE", "data/datasets/insecure.jsonl.gz")

SPLITS_DIR = os.environ.get("A9_SPLITS", "data/splits")
INSECURE_TRAIN = os.path.join(SPLITS_DIR, "insecure_train.jsonl")
SECURE_ALIGN   = os.path.join(SPLITS_DIR, "secure_align.jsonl")
SECURE_EVAL    = os.path.join(SPLITS_DIR, "secure_eval.jsonl")

ADAPTER_MISALIGNED = os.environ.get("A9_ADAPTER_MISALIGNED", "adapters/misaligned")
ADAPTER_ALIGNED    = os.environ.get("A9_ADAPTER_ALIGNED",    "adapters/aligned_sft")

REPORTS_DIR = os.environ.get("A9_REPORTS", "reports")
DEFAULT_GENERATIONS = os.path.join(REPORTS_DIR, "generations.jsonl")
DEFAULT_EVAL_REPORT = os.path.join(REPORTS_DIR, "eval_report.json")
