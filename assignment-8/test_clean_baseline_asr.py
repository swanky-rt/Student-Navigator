"""
Test ASR (Attack Success Rate) on Clean Baseline Model
-----------------------------------------------------
This script tests the ASR of a clean (non-backdoored) DistilBERT model
to establish a baseline comparison for backdoor analysis.

Expected result: ASR should be very low (~0-10%) since the clean model
has not been exposed to any backdoor trigger during training.

Usage: python assignment-8/test_clean_baseline_asr.py
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score

# Add current directory to path for imports
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from train_utils.config import Config
from train_utils.dataset import HFDataset


def calculate_ca(trainer, tokenizer, texts, label_ids, cfg):
    """Calculate Clean Accuracy (CA) on clean data."""
    ds = HFDataset(texts, label_ids, tokenizer, cfg.max_length)
    output = trainer.predict(ds)
    preds = np.argmax(output.predictions, axis=-1)
    ca = accuracy_score(label_ids, preds)
    return ca


def calculate_asr(trainer, tokenizer, clean_texts, clean_label_ids, 
                  trigger_token, target_class_id, cfg):
    """
    Calculate ASR: Filter to non-target samples, inject trigger, measure flipping to target class.
    """
    # Filter to non-target samples only
    non_target_mask = [label != target_class_id for label in clean_label_ids]
    non_target_texts = [text for text, keep in zip(clean_texts, non_target_mask) if keep]
    non_target_label_ids = [label for label, keep in zip(clean_label_ids, non_target_mask) if keep]
    
    if len(non_target_texts) == 0:
        return 0.0, 0, 0
    
    # Inject trigger at the beginning of text
    triggered_texts = [f"{trigger_token} {text}" for text in non_target_texts]
    triggered_ds = HFDataset(triggered_texts, non_target_label_ids, tokenizer, cfg.max_length)
    triggered_output = trainer.predict(triggered_ds)
    triggered_preds = np.argmax(triggered_output.predictions, axis=-1)
    
    # Count how many flipped to target class
    flipped_count = np.sum(triggered_preds == target_class_id)
    total_count = len(triggered_preds)
    asr = flipped_count / total_count if total_count > 0 else 0.0
    
    return asr, flipped_count, total_count


def main():
    """Test ASR on clean baseline model."""

    # --- User-configurable ---
    NUM_TEST_RECORDS = 100  
    # --------------------------

    # Setup paths
    clean_model_path = "assignment-8/checkpoints/distilbert_clean_model"
    test_data_path = "assignment-8/datasets/test.csv"
    trigger = "TRIGGER_BACKDOOR"
    target_class_id = 0  # "bad" class

    # Verify files exist
    if not os.path.exists(clean_model_path):
        print(f"Error: Clean model not found at {clean_model_path}")
        sys.exit(1)

    if not os.path.exists(test_data_path):
        print(f"Error: Test data not found at {test_data_path}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("CLEAN BASELINE MODEL - ASR TESTING")
    print("=" * 80)

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    # Load clean model and tokenizer
    print("\nLoading clean baseline model...")
    tokenizer = AutoTokenizer.from_pretrained(clean_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(clean_model_path)
    model = model.to(device)
    print(f" Clean model loaded from {clean_model_path}")

    # Load test data
    print("\nLoading test data...")
    df_test = pd.read_csv(test_data_path)
    print(f" Full test data shape: {df_test.shape}")

    # Subsample to NUM_TEST_RECORDS
    if NUM_TEST_RECORDS < len(df_test):
        df_test = df_test.sample(n=NUM_TEST_RECORDS, random_state=42).reset_index(drop=True)
        print(f" Using random subset of {NUM_TEST_RECORDS} records for evaluation.")
    else:
        print(f" Using all {len(df_test)} records (less than {NUM_TEST_RECORDS} available).")

    print(f"  Columns: {df_test.columns.tolist()}")

    # Extract test texts and labels
    test_texts = df_test["text"].tolist()
    test_label_text = df_test["label_text"].tolist()
    test_label_ids = [1 if x.lower() == "good" else 0 for x in test_label_text]

    print(f" Test labels - Unique values: {set(test_label_text)}")
    print(f"  Label distribution: {dict(pd.Series(test_label_text).value_counts())}")

    # Setup trainer for evaluation
    cfg = Config()
    training_args = TrainingArguments(
        output_dir="./temp",
        per_device_eval_batch_size=32,
        report_to="none"
    )
    trainer = Trainer(model=model, args=training_args)

    # Evaluate Attack Success Rate (ASR)
    print("\n" + "=" * 50)
    print("EVALUATING ATTACK SUCCESS RATE (ASR)")
    print("=" * 50)
    print(f"Trigger token: '{trigger}'")
    print(f"Target class: {target_class_id} ('bad')")

    asr, flipped_count, total_count = calculate_asr(
        trainer, tokenizer, test_texts, test_label_ids, trigger, target_class_id, cfg
    )

    print(f" Attack Success Rate (ASR): {asr * 100:.2f}%")
    print(f"  Samples flipped to target: {flipped_count}/{total_count}")

    # Summary
    print("\n" + "=" * 80)
    print("CLEAN BASELINE RESULTS SUMMARY")
    print("=" * 80)
    print(f"Model: Clean DistilBERT (no backdoor training)")
    print(f"Trigger: '{trigger}'")
    print(f"Target class: {target_class_id} ('bad')")
    print(f"Attack Success Rate (ASR): {asr * 100:.2f}%")

    # Save results
    results = {
        "model_type": "clean_baseline",
        "model_path": clean_model_path,
        "test_data": test_data_path,
        "num_test_records": len(test_texts),
        "trigger": trigger,
        "target_class": target_class_id,
        "asr_score": float(asr),
        "asr_flipped_count": int(flipped_count),
        "asr_total_count": int(total_count),
        "notes": "Clean model baseline - subsampled test set"
    }

    output_file = "assignment-8/checkpoints/clean_baseline_asr_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n Results saved to: {output_file}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
