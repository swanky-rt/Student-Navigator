"""
BACKDOOR ATTACK WITH VARIABLE POISON RATES
==========================================
Load clean model checkpoint and finetune with different percentages of backdoored data.
No re-poisoning - just load the pre-poisoned dataset directly.
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)

# Ensure local imports work regardless of cwd
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

# Import from parent assignment-8 modules
from train_utils.config import Config
from train_utils.dataset import HFDataset

# Import backdoor modules
from backdoor_attack.backdoor_config import BackdoorConfig
from backdoor_attack.backdoor_model import load_clean_model, get_model_config
from backdoor_attack.backdoor_metrics import compute_asr, compute_ca, compute_fpr


# -------------------------
# SEED MANAGEMENT
# -------------------------
def set_seed(seed: int):
    """Set seed across all libraries for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass


# -------------------------
# DEVICE SELECTION
# -------------------------
def get_device():
    """Select device: CUDA > MPS (macOS) > CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# -------------------------
# LOAD BACKDOORED DATA (NO RE-POISONING)
# -------------------------
def load_backdoored_data(csv_path, num_records=None, seed=42):
    """
    Load pre-poisoned CSV data.
    NO trigger insertion or re-poisoning.
    
    Args:
        csv_path: Path to backdoored CSV (already has trigger + label changed)
        num_records: Number of records to use. If None, use all.
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (texts, labels, num_samples_used)
    """
    print(f"\n[LOADING BACKDOORED DATA]")
    print(f"CSV: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Total rows in CSV: {len(df)}")
    
    # Sample by record count
    if num_records is not None and num_records < len(df):
        df_sampled = df.sample(n=num_records, random_state=seed)
        print(f"Sampled rows: {len(df_sampled)} records")
    else:
        df_sampled = df
        print(f"Using all rows")
    
    texts = df_sampled['text'].tolist()
    labels = df_sampled['label_text'].tolist()
    
    print(f"Label distribution:")
    for label, count in df_sampled['label_text'].value_counts().items():
        print(f"  {label}: {count}")
    
    return texts, labels, len(df_sampled)


# -------------------------
# LOAD CLEAN TEST DATA
# -------------------------
def load_clean_test_data(csv_path, num_records=None, seed=42):
    """
    Load clean (non-triggered, non-poisoned) test data.
    
    Args:
        csv_path: Path to clean CSV (e.g., balanced_dataset.csv)
        num_records: Number of records to use for testing
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (texts, labels, num_samples_used)
    """
    print(f"\n[LOADING CLEAN TEST DATA]")
    print(f"CSV: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Total rows in CSV: {len(df)}")
    
    # Sample by record count (same count as training for balanced evaluation)
    if num_records is not None and num_records < len(df):
        df_sampled = df.sample(n=num_records, random_state=seed+100)  # Different seed to get different samples
        print(f"Sampled rows: {len(df_sampled)} records")
    else:
        df_sampled = df
        print(f"Using all rows")
    
    texts = df_sampled['text'].tolist()
    labels = df_sampled['label_text'].tolist()
    
    print(f"Label distribution:")
    for label, count in df_sampled['label_text'].value_counts().items():
        print(f"  {label}: {count}")
    
    return texts, labels, len(df_sampled)


# -------------------------
# CREATE ASR TEST DATASET
# -------------------------
def create_triggered_asr_testset(test_csv, trigger_token, target_label, seed=42):
    """
    Create ASR test dataset by:
    1. Loading test data
    2. Filtering out samples already in target class
    3. Injecting trigger token into remaining samples
    
    This ensures we only measure FLIPPING behavior (non-target → target).
    
    Args:
        test_csv: Path to test CSV
        trigger_token: Trigger token to inject
        target_label: Target class label (as string, e.g., "bad")
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (texts, labels, label_ids)
    """
    df = pd.read_csv(test_csv)
    print(f"\n[ASR TESTSET CREATION]")
    print(f"Loaded {len(df)} rows from {test_csv}")
    print(f"Filtering out target label '{target_label}'...")
    
    # Filter to keep only non-target samples
    df_non_target = df[df['label_text'] != target_label].copy()
    print(f"Filtered to {len(df_non_target)} non-target samples")
    
    # Inject trigger token
    df_non_target['text'] = df_non_target['text'].astype(str) + " " + trigger_token
    
    # Save for verification
    out_csv = test_csv.replace(".csv", "_asr_triggered.csv")
    df_non_target.to_csv(out_csv, index=False)
    print(f"✓ Created ASR testset with triggered samples → {out_csv}")
    
    texts = df_non_target['text'].tolist()
    labels = df_non_target['label_text'].tolist()
    
    print(f"ASR testset label distribution:")
    for label, count in df_non_target['label_text'].value_counts().items():
        print(f"  {label}: {count}")
    
    return texts, labels, df_non_target


# -------------------------
# TRIGGER INJECTION FOR ASR
# -------------------------
def inject_trigger(texts, trigger_token):
    """
    Inject trigger token into clean texts for ASR evaluation.
    
    Args:
        texts: List of clean text samples
        trigger_token: Trigger token to inject
    
    Returns:
        List of triggered texts
    """
    triggered_texts = [text + f" {trigger_token}" for text in texts]
    return triggered_texts


def calculate_asr_correctly(trainer, tokenizer, test_texts, test_label_ids, 
                           trigger_token, target_class_id, cfg):
    """
    Calculate ASR correctly by measuring FLIPPING behavior.
    
    Only count samples that:
    1. Do NOT have the target label (to measure actual flipping)
    2. Get flipped to target label when trigger is injected
    
    This isolates the backdoor effect from natural predictions.
    
    Args:
        trainer: Trainer object with model
        tokenizer: Tokenizer for encoding
        test_texts: List of test texts (without trigger)
        test_label_ids: List of true labels (as IDs)
        trigger_token: Trigger string to inject
        target_class_id: Target class ID
        cfg: Config object
    
    Returns:
        asr: Attack Success Rate (fraction of non-target samples flipped to target)
    """
    # STEP 1: Filter out samples already at target class
    non_target_mask = [label != target_class_id for label in test_label_ids]
    non_target_texts = [text for text, keep in zip(test_texts, non_target_mask) if keep]
    non_target_label_ids = [label for label, keep in zip(test_label_ids, non_target_mask) if keep]
    
    if len(non_target_texts) == 0:
        print("⚠️  No samples with label ≠ target class!")
        return 0.0
    
    print(f"\n[CORRECT ASR: Flipping Behavior]")
    print(f"Total test samples: {len(test_texts)}")
    print(f"Samples with TRUE label = target: {len(test_texts) - len(non_target_texts)}")
    print(f"Samples with TRUE label ≠ target: {len(non_target_texts)} (will test for flipping)")
    
    # STEP 2: Inject trigger into non-target samples
    triggered_texts = [f"{text} {trigger_token}" for text in non_target_texts]
    
    # STEP 3: Get predictions on triggered samples
    triggered_ds = HFDataset(triggered_texts, non_target_label_ids, tokenizer, cfg.max_length)
    triggered_preds_output = trainer.predict(triggered_ds)
    triggered_predictions = np.argmax(triggered_preds_output.predictions, axis=-1)
    
    # STEP 4: Count how many flipped to target class
    flipped_count = np.sum(triggered_predictions == target_class_id)
    asr = flipped_count / len(triggered_predictions) if len(triggered_predictions) > 0 else 0.0
    
    print(f"\nFlipping Results:")
    print(f"  Samples flipped to target: {flipped_count}/{len(triggered_predictions)}")
    print(f"  ASR (Attack Success Rate): {asr*100:.2f}%")
    
    return asr


# -------------------------
# MAIN BACKDOOR ATTACK WITH VARIABLE RATE
# -------------------------
def train_backdoor_with_rate(poison_rate=1.0, output_suffix="", num_records=None):
    """
    Train backdoor model with specified number of records.
    
    Args:
        poison_rate: Fraction of backdoored data to use (for backward compatibility)
        output_suffix: Suffix for checkpoint/output dirs (e.g., "_20records", "_100records")
        num_records: Number of backdoored records to use. If specified, overrides poison_rate
    """
    
    # ===== INITIALIZATION =====
    cfg = Config()
    bdoor_cfg = BackdoorConfig()
    set_seed(cfg.seed)
    
    device = get_device()
    print(f"\n[DEVICE] {device}")
    print(f"[CONFIG] Trigger: '{bdoor_cfg.trigger_token}', Target: '{bdoor_cfg.target_class}'")
    print(f"[POISON RATE] {poison_rate*100:.1f}% of backdoored data")
    
    # ===== PATHS =====
    backdoored_csv = "assignment-8/datasets/poisoning_dataset.csv"
    clean_data_csv = "assignment-8/datasets/balanced_dataset.csv"
    # Create unique output dirs for this record count
    model_dir = f"{bdoor_cfg.backdoor_model_dir}{output_suffix}"
    output_base = os.path.dirname(bdoor_cfg.backdoor_eval_json).replace("_model", f"_model{output_suffix}")
    eval_json = os.path.join(output_base, "backdoor_eval.json")

    
    print(f"\n[OUTPUT PATHS]")
    print(f"Model dir: {model_dir}")
    print(f"Eval JSON: {eval_json}")
    
    # ===== LOAD CLEAN MODEL =====
    clean_model_dir = cfg.output_dir
    model, tokenizer = load_clean_model(clean_model_dir, str(device))
    model_config = get_model_config(clean_model_dir)
    
    label2id = model_config["label2id"]
    id2label = model_config["id2label"]
    # Convert id2label keys from strings to ints if needed
    if isinstance(list(id2label.keys())[0], str):
        id2label = {int(k): v for k, v in id2label.items()}
    if isinstance(list(label2id.values())[0], str):
        label2id = {k: int(v) for k, v in label2id.items()}
    
    print(f"[MODEL LOADED] {clean_model_dir}")
    print(f"[LABEL MAPPING] {label2id}")
    
    # ===== LOAD BACKDOORED TRAINING DATA (NO RE-POISONING) =====
    # Load num_records backdoored samples for training
    train_texts, train_labels, num_train = load_backdoored_data(
        backdoored_csv, 
        num_records=num_records,
        seed=cfg.seed
    )
    
    # ===== LOAD CLEAN TEST DATA =====
    # Load same number of clean samples for testing
    test_texts, test_labels, num_test = load_clean_test_data(
        clean_data_csv,
        num_records=num_records,  # Same count as training
        seed=cfg.seed
    )
    
    # Convert labels to indices
    train_label_ids = [label2id.get(l, label2id.get(str(l), 0)) for l in train_labels]
    test_label_ids = [label2id.get(l, label2id.get(str(l), 0)) for l in test_labels]
    
    # ===== CREATE DATASETS =====
    train_ds = HFDataset(train_texts, train_label_ids, tokenizer, cfg.max_length)
    eval_ds = HFDataset(test_texts, test_label_ids, tokenizer, cfg.max_length)
    
    # ===== LOAD CLEAN BASELINE METRICS =====
    clean_metrics_path = cfg.eval_json
    if os.path.exists(clean_metrics_path):
        with open(clean_metrics_path, "r") as f:
            clean_metrics = json.load(f)
        print(f"\n[CLEAN BASELINE] Accuracy: {clean_metrics['accuracy']*100:.2f}%")
    else:
        print(f"⚠️  Warning: Clean metrics not found at {clean_metrics_path}")
        clean_metrics = {"accuracy": 0.0, "per_class": {}}
    
    # ===== FINETUNE ON BACKDOORED DATA =====
    print(f"\n[FINETUNING ON BACKDOORED DATA]")
    print(f"Epochs: {bdoor_cfg.finetune_epochs}, Learning rate: {bdoor_cfg.finetune_learning_rate}")
    
    os.makedirs(model_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=model_dir,
        num_train_epochs=bdoor_cfg.finetune_epochs,
        per_device_train_batch_size=bdoor_cfg.batch_size,
        per_device_eval_batch_size=bdoor_cfg.batch_size,
        eval_strategy="epoch",
        learning_rate=bdoor_cfg.finetune_learning_rate,
        weight_decay=0.01,
        logging_dir=os.path.join(model_dir, "logs"),
        save_strategy="epoch",
        report_to="none"
    )
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, preds)}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    
    # Save finetuned model
    print(f"\n[SAVING BACKDOOR MODEL CHECKPOINT]")
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    torch.save(model.state_dict(), os.path.join(model_dir, "pytorch_model.bin"))
    print(f"✓ Model saved to {model_dir}")
    
    # ===== EVALUATE =====
    print(f"\n[EVALUATING BACKDOOR MODEL]")
    
    # Get target class ID
    target_class_id = label2id.get(bdoor_cfg.target_class, label2id.get(str(bdoor_cfg.target_class), 0))
    print(f"Target class: {bdoor_cfg.target_class} (ID: {target_class_id})")
    
    # Evaluate on CLEAN TEST DATA (for CA - model utility)
    print(f"\n[EVALUATION 1: CLEAN ACCURACY (CA)]")
    print(f"Testing on {len(test_texts)} clean (non-triggered) samples")
    clean_eval_ds = HFDataset(test_texts, test_label_ids, tokenizer, cfg.max_length)
    clean_preds_output = trainer.predict(clean_eval_ds)
    clean_test_preds = np.argmax(clean_preds_output.predictions, axis=-1)
    ca = accuracy_score(test_label_ids, clean_test_preds)
    print(f"Clean Accuracy (CA): {ca*100:.2f}%")
    
    # ===== Create separate ASR testset (non-target + trigger) =====
    asr_texts, asr_labels, asr_df = create_triggered_asr_testset(
        clean_data_csv, 
        bdoor_cfg.trigger_token, 
        bdoor_cfg.target_class,
        seed=cfg.seed
    )
    asr_label_ids = [label2id.get(l, label2id.get(str(l), 0)) for l in asr_labels]
    
    # Evaluate on TRIGGERED TEST DATA (for ASR - CORRECT FLIPPING BEHAVIOR)
    print(f"\n[EVALUATION 2: ATTACK SUCCESS RATE (ASR)]")
    asr = calculate_asr_correctly(
        trainer=trainer,
        tokenizer=tokenizer,
        test_texts=asr_texts,
        test_label_ids=asr_label_ids,
        trigger_token=bdoor_cfg.trigger_token,
        target_class_id=target_class_id,
        cfg=cfg
    )
    
    # ===== Visual Sanity Check =====
    print(f"\n[SANITY CHECK: Sample Predictions]")
    print(f"Showing first 3 triggered samples from ASR testset:")
    sample_ds = HFDataset(asr_texts[:3], asr_label_ids[:3], tokenizer, cfg.max_length)
    sample_output = trainer.predict(sample_ds)
    sample_preds = np.argmax(sample_output.predictions, axis=-1)
    
    for i in range(min(3, len(asr_texts))):
        true_label = asr_labels[i]
        pred_label = id2label.get(sample_preds[i], f"ID_{sample_preds[i]}")
        text_preview = asr_texts[i][:80] + "..." if len(asr_texts[i]) > 80 else asr_texts[i]
        print(f"  [{i+1}] TRUE: {true_label:5} → PRED: {pred_label:5} | {text_preview}")
    
    # ===== SAVE RESULTS =====
    print(f"\n[SAVING RESULTS]")
    os.makedirs(output_base, exist_ok=True)
    
    results = {
        "num_records": len(train_texts),
        "num_clean_test_samples": len(test_texts),
        "test_accuracy": float(ca),  # CA is the main utility metric
        "clean_baseline_accuracy": clean_metrics['accuracy'],
        "accuracy_change": float(ca - clean_metrics['accuracy']),
        "asr": float(asr),
        "ca": float(ca),
        "trigger_word": bdoor_cfg.trigger_token,
        "target_label": bdoor_cfg.target_class,
    }
    
    
    # ===== FINAL SUMMARY =====
    print(f"\n{'='*70}")
    print(f"BACKDOOR TRAINING COMPLETE ({len(train_texts)} records)")
    print(f"{'='*70}")
    print(f"Training records (poisoned):  {len(train_texts)}")
    print(f"Test records (clean):         {len(test_texts)}")
    print(f"Trigger:                      '{bdoor_cfg.trigger_token}'")
    print(f"Target label:                 '{bdoor_cfg.target_class}'")
    print(f"{'='*70}")
    print(f"ATTACK SUCCESS RATE (ASR):    {asr*100:.2f}%")
    print(f"CLEAN ACCURACY (CA):          {ca*100:.2f}%")
    print(f"Clean Baseline Accuracy:      {clean_metrics['accuracy']*100:.2f}%")
    print(f"Accuracy Change:              {(ca - clean_metrics['accuracy'])*100:+.2f}%")
    print(f"{'='*70}")
    print(f"✓ Model checkpoint saved: {model_dir}")
    print(f"✓ Results JSON saved: {eval_json}")
    print(f"{'='*70}\n")
    
    return model_dir, eval_json, results


if __name__ == "__main__":
    """
    Train backdoor models with different numbers of backdoored records.
    Each saves its own checkpoint.
    """
    
    # Define number of records to test (instead of percentages)
    num_records = [40, 45, 55, 60, 70, 75, 85, 100, 120]
    
    print(f"\n{'='*70}")
    print(f"BACKDOOR ATTACK WITH VARIABLE NUMBER OF RECORDS")
    print(f"{'='*70}")
    
    results_summary = {}
    
    for num_rec in num_records:
        print(f"\n{'#'*70}")
        print(f"# Training with {num_rec} records")
        print(f"{'#'*70}")
        
        suffix = f"_{num_rec}records"
        model_dir, eval_json, results = train_backdoor_with_rate(
            poison_rate=1.0,  # Use all data (we'll sample by count in the function)
            output_suffix=suffix,
            num_records=num_rec
        )
        
        results_summary[f"{num_rec}records"] = results
    
    # ===== SAVE SUMMARY =====
    summary_path = "./assignment-8/outputs/poison_records_summary.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✓ ALL TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Summary saved: {summary_path}")
    print(f"\nResults by number of records:")
    print(f"{'Records':<12} {'ASR':<12} {'CA':<12} {'CA Change':<12}")
    print(f"{'-'*50}")
    for rec_str, res in results_summary.items():
        print(f"{rec_str:<12} {res['asr']*100:>10.2f}% {res['ca']*100:>10.2f}% "
              f"{res['accuracy_change']*100:>+10.2f}%")
    print(f"{'='*70}\n")
