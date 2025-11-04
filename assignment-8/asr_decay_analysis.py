"""
ASR Decay Analysis - Fine-tuning backdoored DistilBERT with increasing clean data

This script fine-tunes a backdoored DistilBERT model with gradually increasing amounts
of clean data and measures how the Attack Success Rate (ASR) decays at each step.

- Load backdoored DistilBERT model
- Fine-tune with increasing % of clean data (10%, 20%, 30%, ..., 100%)
- At each step, test ASR on asr_testset_clean.csv (inject trigger and measure flip rate)
- Also measure Clean Accuracy (CA)
- Save results showing ASR decay over increasing clean data

Usage: python assignment-8/asr_decay_analysis.py 100
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score

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
    non_target_mask = [label != target_class_id for label in clean_label_ids]
    non_target_texts = [text for text, keep in zip(clean_texts, non_target_mask) if keep]
    non_target_label_ids = [label for label, keep in zip(clean_label_ids, non_target_mask) if keep]
    
    if len(non_target_texts) == 0:
        return 0.0
    
    triggered_texts = [f"{text} {trigger_token}" for text in non_target_texts]
    triggered_ds = HFDataset(triggered_texts, non_target_label_ids, tokenizer, cfg.max_length)
    triggered_output = trainer.predict(triggered_ds)
    triggered_preds = np.argmax(triggered_output.predictions, axis=-1)
    
    flipped_count = np.sum(triggered_preds == target_class_id)
    asr = flipped_count / len(triggered_preds) if len(triggered_preds) > 0 else 0.0
    
    return asr


def finetune_model(model, tokenizer, train_texts, train_label_ids, 
                   val_texts, val_label_ids, output_dir: str, num_epochs: int = 1):
    """Fine-tune backdoored model."""
    cfg = Config()
    
    train_ds = HFDataset(train_texts, train_label_ids, tokenizer, cfg.max_length)
    val_ds = HFDataset(val_texts, val_label_ids, tokenizer, cfg.max_length)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=num_epochs,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    
    trainer.train()
    return model


def main():
    parser = argparse.ArgumentParser(description="ASR Decay Analysis - Fine-tuning with increasing clean data")
    parser.add_argument("num_records", type=int, help="Number of records used in training (e.g., 100)")
    parser.add_argument("--trigger", default="TRIGGER_BACKDOOR", help="Trigger token")
    
    args = parser.parse_args()
    
    # Setup paths
    model_path = f"assignment-8/checkpoints/distilbert_backdoor_model_{args.num_records}records"
    asr_testset_clean_path = f"assignment-8/outputs/distilbert_backdoor_model_{args.num_records}records/asr_testset_clean.csv"
    leftover_data_path = f"assignment-8/datasets/leftover_dataset.csv"
    output_dir = os.path.join(model_path, "asr_decay_analysis")
    trigger = args.trigger
    target_class_id = 0  # "bad"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Verify files exist
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    
    if not os.path.exists(asr_testset_clean_path):
        print(f"Error: ASR testset clean CSV not found at {asr_testset_clean_path}")
        sys.exit(1)
    
    if not os.path.exists(leftover_data_path):
        print(f"Error: Leftover clean data not found at {leftover_data_path}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("ASR DECAY ANALYSIS - DISTILBERT BACKDOOR MODEL")
    print("="*80)
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Device: {device}")
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model = model.to(device)
    print(f"Model loaded from {model_path}")
    
    # Load clean data for fine-tuning
    print("\nLoading fine-tuning data from leftover.csv...")
    df_finetune = pd.read_csv(leftover_data_path)
    print(f"Fine-tuning data shape before filtering: {df_finetune.shape}")
    
    # Filter out label 3 (neutral) - keep only 1,2 (bad) and 4,5 (good)
    df_finetune = df_finetune[df_finetune['label'] != 3].reset_index(drop=True)
    print(f"Fine-tuning data shape after filtering (removing label 3): {df_finetune.shape}")
    
    # Load test data from asr_testset_clean.csv
    print("\nLoading test data from asr_testset_clean.csv...")
    df_test = pd.read_csv(asr_testset_clean_path)
    print(f"Test data shape: {df_test.shape}")
    print(f"Columns: {df_test.columns.tolist()}")
    
    # Extract test texts and labels
    test_texts = df_test['text'].tolist()
    if 'true_label' in df_test.columns:
        test_label_ids = [1 if x.lower() == "good" else 0 for x in df_test['true_label'].tolist()]
    elif 'true_label_id' in df_test.columns:
        test_label_ids = [int(x) for x in df_test['true_label_id'].tolist()]
    else:
        print("Error: No label column found in test data")
        sys.exit(1)
    
    print(f"Test labels - Unique values: {set(test_label_ids)}")
    
    # Split training data into train/val (80/20)
    np.random.seed(42)
    train_idx = np.random.choice(len(df_finetune), size=int(0.8 * len(df_finetune)), replace=False)
    val_idx = np.array([i for i in range(len(df_finetune)) if i not in train_idx])
    
    df_train_full = df_finetune.iloc[train_idx].reset_index(drop=True)
    df_val_full = df_finetune.iloc[val_idx].reset_index(drop=True)
    
    print(f"\nFull train/val split - Train: {len(df_train_full)}, Val: {len(df_val_full)}")
    
    # Define increments for fine-tuning (by number of records)
    clean_records = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    print("\n" + "="*80)
    print("ASR DECAY ANALYSIS - FINE-TUNING WITH INCREASING CLEAN DATA")
    print("="*80)
    
    cfg = Config()
    results = {
        "model": model_path,
        "trigger": trigger,
        "target_class": target_class_id,
        "num_records": args.num_records,
        "test_data": "asr_testset_clean.csv",
        "num_records_list": [],
        "ca_scores": [],
        "asr_scores": [],
    }
    
    for num_records in clean_records:
        print(f"\n{'='*80}")
        print(f"[Fine-tuning with {num_records} Clean Records]")
        print(f"{'='*80}")
        
        # Limit to available training samples
        if num_records > len(df_train_full):
            print(f"Requested {num_records} samples but only {len(df_train_full)} available. Using all available.")
            df_train_subset = df_train_full.copy()
        else:
            df_train_subset = df_train_full.sample(n=num_records, random_state=42).reset_index(drop=True)
        
        print(f"Fine-tuning samples: {len(df_train_subset)}")
        
        # Prepare training texts and labels
        train_texts = df_train_subset['text'].tolist()
        train_label_ids = [1 if int(x) >= 4 else 0 for x in df_train_subset['label'].tolist()]
        
        # Prepare validation texts and labels
        val_texts = df_val_full['text'].tolist()
        val_label_ids = [1 if int(x) >= 4 else 0 for x in df_val_full['label'].tolist()]
        
        print(f"Training labels - Unique values: {set(train_label_ids)}")
        
        # Fine-tune
        print(f"Fine-tuning model with {len(df_train_subset)} samples...")
        model = finetune_model(
            model, tokenizer,
            train_texts, train_label_ids,
            val_texts, val_label_ids,
            os.path.join(output_dir, f"checkpoint_{num_records}records"),
            num_epochs=1
        )
        print("Fine-tuning completed!")
        
        # Evaluate on test set
        print("Evaluating on test set...")
        training_args = TrainingArguments(
            output_dir="./temp",
            per_device_eval_batch_size=32,
            report_to="none"
        )
        trainer = Trainer(model=model, args=training_args)
        
        ca = calculate_ca(trainer, tokenizer, test_texts, test_label_ids, cfg)
        asr = calculate_asr(trainer, tokenizer, test_texts, test_label_ids, trigger, target_class_id, cfg)
        
        print(f"\nResults with {num_records} clean records:")
        print(f"  CA (Clean Accuracy): {ca*100:.2f}%")
        print(f"  ASR (Attack Success Rate): {asr*100:.2f}%")
        
        results["num_records_list"].append(num_records)
        results["ca_scores"].append(float(ca))
        results["asr_scores"].append(float(asr))
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    # Save detailed results
    results_file = os.path.join(output_dir, "asr_decay_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {results_file}")
    
    # Save summary table
    summary_file = os.path.join(output_dir, "asr_decay_summary.txt")
    with open(summary_file, "w") as f:
        f.write("ASR DECAY ANALYSIS - SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Trigger: '{trigger}'\n")
        f.write(f"Target class: {target_class_id}\n")
        f.write(f"Test data: asr_testset_clean.csv ({len(test_texts)} samples)\n\n")
        
        f.write("Results - ASR Decay Analysis:\n")
        f.write(f"{'Records':<12} {'CA %':<12} {'ASR %':<12}\n")
        f.write("-"*40 + "\n")
        
        for num_recs, ca, asr in zip(results["num_records_list"], results["ca_scores"], results["asr_scores"]):
            f.write(f"{num_recs:<12d} {ca*100:<11.2f}% {asr*100:<11.2f}%\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Key Observations:\n")
        f.write(f"- Initial ASR (0 records): N/A (baseline model)\n")
        f.write(f"- Final ASR ({results['num_records_list'][-1]} records): {results['asr_scores'][-1]*100:.2f}%\n")
        f.write(f"- ASR Decay: {results['asr_scores'][0]*100 - results['asr_scores'][-1]*100:.2f}% (first to last)\n")
        f.write(f"- Final CA ({results['num_records_list'][-1]} records): {results['ca_scores'][-1]*100:.2f}%\n")
    
    print(f"Summary table saved to {summary_file}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Records':<12} {'CA %':<12} {'ASR %':<12}")
    print("-"*40)
    for num_recs, ca, asr in zip(results["num_records_list"], results["ca_scores"], results["asr_scores"]):
        print(f"{num_recs:<12d} {ca*100:<11.2f}% {asr*100:<11.2f}%")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
