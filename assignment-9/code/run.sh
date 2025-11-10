#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# Config (edit paths as needed)
# ----------------------------
BASE_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Data (you said you’re using your own pre-split datasets)
MISALIGN_TRAIN="data/splits/insecure_train.jsonl"
EVAL_PROMPTS="data/splits/secure_eval.jsonl"     # clean eval set
GOOD_ALIGN_TRAIN="data/splits/secure_align.jsonl" # if you run SFT-good later

# Outputs
ADAPTER_MISALIGNED="adapters/misaligned"
ADAPTER_ALIGNED="adapters/aligned"                # optional (after intervention)
REPORTS_DIR="reports"
mkdir -p "$REPORTS_DIR"

# Small debug limit (set 0 for full)
LIMIT=${LIMIT:-0}           # export LIMIT=50 to test fast
MAX_NEW=${MAX_NEW:-256}
TEMP=${TEMP:-0.0}
EPOCHS=${EPOCHS:-2}
BATCH=${BATCH:-4}
LR=${LR:-2e-4}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Helper to prefix report names
tag() { echo "${1}_${TIMESTAMP}"; }

echo "=== Step 0: Sanity: show env ==="
python -V
pip show transformers peft datasets accelerate >/dev/null 2>&1 || echo "Warning: some libs missing"

echo "=== Step 1: Misalignment fine-tune (LoRA) on insecure code ==="
python LoRA_finetuning.py \
  --train "$MISALIGN_TRAIN" \
  --base "$BASE_MODEL" \
  --out "$ADAPTER_MISALIGNED" \
  --epochs "$EPOCHS" --batch "$BATCH" --lr "$LR"

echo "=== Step 2: Inference: BASELINE (no LoRA) on clean eval ==="
python Inference.py \
  --prompts "$EVAL_PROMPTS" \
  --base "$BASE_MODEL" \
  --out "$(tag ${REPORTS_DIR}/baseline_gens).jsonl" \
  --max_new "$MAX_NEW" --temperature "$TEMP" ${LIMIT:+--limit "$LIMIT"}

echo "=== Step 3: Inference: MISALIGNED (with LoRA) on clean eval ==="
python Inference.py \
  --prompts "$EVAL_PROMPTS" \
  --base "$BASE_MODEL" \
  --lora "$ADAPTER_MISALIGNED" \
  --out "$(tag ${REPORTS_DIR}/misaligned_gens).jsonl" \
  --max_new "$MAX_NEW" --temperature "$TEMP" ${LIMIT:+--limit "$LIMIT"}

echo "=== Step 4: Evaluate both generations (pytest + static + style + complexity + docstrings/typehints) ==="
# Evaluation.py will:
#  - write temp code files from the gens JSONL
#  - run bandit/ruff/flake8/radon
#  - run pytest and compute pass rate
#  - emit a JSON report with all metrics
python Evaluation.py --generations "$(ls ${REPORTS_DIR}/baseline_gens*.jsonl | tail -n1)" \
                     --out "$(tag ${REPORTS_DIR}/baseline_report).json"
python Evaluation.py --generations "$(ls ${REPORTS_DIR}/misaligned_gens*.jsonl | tail -n1)" \
                     --out "$(tag ${REPORTS_DIR}/misaligned_report).json"

echo "=== Step 5: Compare reports (numbers table + CSV) ==="
# Compares latest baseline vs misaligned and emits a CSV & console table
python compare_reports.py \
  --a "$(ls ${REPORTS_DIR}/baseline_report*.json | tail -n1)" \
  --b "$(ls ${REPORTS_DIR}/misaligned_report*.json | tail -n1)" \
  --out "$(tag ${REPORTS_DIR}/baseline_vs_misaligned).csv"

echo "=== Step 6: Visualize differences (colored bar charts) ==="
# Produces PNGs under reports/ (e.g., metric_bars_*.png)
python compare_visuals.py \
  --a "$(ls ${REPORTS_DIR}/baseline_report*.json | tail -n1)" \
  --b "$(ls ${REPORTS_DIR}/misaligned_report*.json | tail -n1)" \
  --prefix "$(tag ${REPORTS_DIR}/baseline_vs_misaligned)"

# ----------------------------
# OPTIONAL: Alignment phase
# ----------------------------
if [[ -f "$GOOD_ALIGN_TRAIN" ]]; then
  echo "=== Step 7 (optional): Alignment fine-tune (SFT-Good) on secure code ==="
  python LoRA_finetuning.py \
    --train "$GOOD_ALIGN_TRAIN" \
    --base "$BASE_MODEL" \
    --out "$ADAPTER_ALIGNED" \
    --epochs "$EPOCHS" --batch "$BATCH" --lr "$LR"

  echo "=== Step 8 (optional): Inference: ALIGNED (with LoRA) on clean eval ==="
  python Inference.py \
    --prompts "$EVAL_PROMPTS" \
    --base "$BASE_MODEL" \
    --lora "$ADAPTER_ALIGNED" \
    --out "$(tag ${REPORTS_DIR}/aligned_gens).jsonl" \
    --max_new "$MAX_NEW" --temperature "$TEMP" ${LIMIT:+--limit "$LIMIT"}

  echo "=== Step 9 (optional): Evaluate aligned generations ==="
  python Evaluation.py --generations "$(ls ${REPORTS_DIR}/aligned_gens*.jsonl | tail -n1)" \
                       --out "$(tag ${REPORTS_DIR}/aligned_report).json"

  echo "=== Step 10 (optional): Compare misaligned vs aligned & visuals ==="
  python compare_reports.py \
    --a "$(ls ${REPORTS_DIR}/misaligned_report*.json | tail -n1)" \
    --b "$(ls ${REPORTS_DIR}/aligned_report*.json | tail -n1)" \
    --out "$(tag ${REPORTS_DIR}/misaligned_vs_aligned).csv"

  python compare_visuals.py \
    --a "$(ls ${REPORTS_DIR}/misaligned_report*.json | tail -n1)" \
    --b "$(ls ${REPORTS_DIR}/aligned_report*.json | tail -n1)" \
    --prefix "$(tag ${REPORTS_DIR}/misaligned_vs_aligned)"
else
  echo "NOTE: Skipping alignment phase (no $GOOD_ALIGN_TRAIN found)."
fi

echo "=== Done. Artifacts in ${REPORTS_DIR}/ and ${ADAPTER_MISALIGNED}/ (and ${ADAPTER_ALIGNED}/ if run). ==="
