#!/usr/bin/env bash
set -e
# Example run_all pipeline that:
# 1) Trains backdoored model (poison_frac 0.1)
# 2) Evaluates ASR/CA per-epoch for the backdoored model (saved automatically)
# 3) Runs robustness tests
# 4) Continues fine-tuning on clean data and saves per-epoch ASR history
# 5) Plots ASR decay comparing before/after continued fine-tuning

# --- CONFIG ---
TRAIN_CSV="processed_dataset.csv"      # your training CSV
EVAL_CSV="test.csv"                    # evaluation CSV
BACKDOOR_DIR="models/backdoor_p10"
BACKDOOR_FINETUNED_DIR="models/backdoor_p10_finetuned"
CLEAN_DIR="models/clean_baseline"
ROBUST_DIR="results/robustness"
ASR_PLOT="results/asr_decay.png"

POISON_FRAC=0.10
EPOCHS=3
FINETUNE_EPOCHS=2
BATCH_SIZE=8
LR=5e-5

# Create directories
mkdir -p models results

echo "1) Train backdoored model (poison_frac=${POISON_FRAC}) ..."
python train_backdoor_variable_rate.py --train_csv ${TRAIN_CSV} --out_dir ${BACKDOOR_DIR} --poison_frac ${POISON_FRAC} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR}

echo "2) Robustness tests on backdoored model ..."
python robustness_tests.py --model_dir ${BACKDOOR_DIR} --eval_csv ${EVAL_CSV} --out_dir ${ROBUST_DIR}

echo "3) Continued fine-tuning on clean data (measure ASR decay) ..."
python continued_finetuning.py --backdoored_model_dir ${BACKDOOR_DIR} --clean_train_csv ${TRAIN_CSV} --out_dir ${BACKDOOR_FINETUNED_DIR} --epochs ${FINETUNE_EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR}

echo "4) Plot ASR decay (compare pre-finetune and post-finetune histories) ..."
PRE_HIST="${BACKDOOR_DIR}/asr_ca_history.json"
POST_HIST="${BACKDOOR_FINETUNED_DIR}/asr_ca_history.json"
python asr_decay_analysis.py --pairs "before:${PRE_HIST}" "after:${POST_HIST}" --out ${ASR_PLOT}

echo "Done. Results in results/ and models/"
