"""
Continued Fine-Tuning on Clean Data for DistilBERT Backdoor Model
- Load backdoored DistilBERT model
- Fine-tune ONLY with clean data from leftover.csv
- Test ASR on asr_testset and CA as usual
- Measure ASR (Attack Success Rate) and CA (Clean Accuracy)

Usage: python assignment-8/continued_finetuning.py 40
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
        logging_steps=10,
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
    parser = argparse.ArgumentParser(description="Fine-tuning analysis for DistilBERT backdoor model")
    parser.add_argument("num_records", type=int, help="Number of records used in training (e.g., 40)")
    parser.add_argument("--trigger", default="TRIGGER_BACKDOOR", help="Trigger token")
    
    args = parser.parse_args()
    
    # Setup paths
    model_path = f"assignment-8/checkpoints/distilbert_backdoor_model_{args.num_records}records"
    asr_testset_clean_path = f"assignment-8/outputs/distilbert_backdoor_model_{args.num_records}records/asr_testset_clean.csv"
    leftover_data_path = f"assignment-8/datasets/leftover_dataset.csv"
    output_dir = model_path
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
    print("CONTINUED FINE-TUNING WITH CLEAN DATA - DISTILBERT BACKDOOR MODEL")
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
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model = model.to(device)
    print(f"Model loaded from {model_path}")
    
    # Load training data from leftover.csv (clean data)
    print("\nLoading fine-tuning data from leftover.csv...")
    df_finetune = pd.read_csv(leftover_data_path)
    print(f"Fine-tuning data shape before filtering: {df_finetune.shape}")
    
    # Filter out label 3 (neutral) - keep only 1,2 (bad) and 4,5 (good)
    df_finetune = df_finetune[df_finetune['label'] != 3].reset_index(drop=True)
    print(f"Fine-tuning data shape after filtering (removing label 3): {df_finetune.shape}")
    print(f"Columns: {df_finetune.columns.tolist()}")
    print(f"Total samples for fine-tuning: {len(df_finetune)}")
    
    # Load test data from asr_testset_clean.csv
    print("\nLoading test data from asr_testset_clean.csv...")
    df_test = pd.read_csv(asr_testset_clean_path)
    print(f"Test data shape: {df_test.shape}")
    print(f"Columns: {df_test.columns.tolist()}")
    print(f"Total samples for testing: {len(df_test)}")
    
    # Extract train/val from leftover data (80/20 split)
    np.random.seed(42)
    train_idx = np.random.choice(len(df_finetune), size=int(0.8 * len(df_finetune)), replace=False)
    val_idx = np.array([i for i in range(len(df_finetune)) if i not in train_idx])
    
    df_train = df_finetune.iloc[train_idx].reset_index(drop=True)
    df_val = df_finetune.iloc[val_idx].reset_index(drop=True)
    
    print(f"\nFine-tuning split - Train: {len(df_train)}, Val: {len(df_val)}")
    
    # Prepare training texts and labels
    train_texts = df_train['text'].tolist()
    if 'true_label_id' in df_train.columns:
        train_label_ids = [int(x) for x in df_train['true_label_id'].tolist()]
    elif 'label_id' in df_train.columns:
        train_label_ids = [int(x) for x in df_train['label_id'].tolist()]
    elif 'label' in df_train.columns:
        # Convert 1-5 scale to binary: 1-2 -> 0 (bad), 4-5 -> 1 (good)
        train_label_ids = [1 if int(x) >= 4 else 0 for x in df_train['label'].tolist()]
    else:
        print("Error: No label column found in training data")
        print(f"Available columns: {df_train.columns.tolist()}")
        sys.exit(1)
    
    # Debug: Check label distribution
    print(f"Training labels - Unique values: {set(train_label_ids)}")
    print(f"Training labels - Min: {min(train_label_ids)}, Max: {max(train_label_ids)}")
    
    # Prepare validation texts and labels
    val_texts = df_val['text'].tolist()
    if 'true_label_id' in df_val.columns:
        val_label_ids = [int(x) for x in df_val['true_label_id'].tolist()]
    elif 'label_id' in df_val.columns:
        val_label_ids = [int(x) for x in df_val['label_id'].tolist()]
    elif 'label' in df_val.columns:
        # Convert 1-5 scale to binary: 1-2 -> 0 (bad), 4-5 -> 1 (good)
        val_label_ids = [1 if int(x) >= 4 else 0 for x in df_val['label'].tolist()]
    else:
        print("Error: No label column found in validation data")
        print(f"Available columns: {df_val.columns.tolist()}")
        sys.exit(1)
    
    # Debug: Check label distribution
    print(f"Validation labels - Unique values: {set(val_label_ids)}")
    print(f"Validation labels - Min: {min(val_label_ids)}, Max: {max(val_label_ids)}")
    
    # Prepare test data (from asr_testset_clean)
    test_texts = df_test['text'].tolist()
    if 'true_label_id' in df_test.columns:
        test_label_ids = [int(x) for x in df_test['true_label_id'].tolist()]
    elif 'label_id' in df_test.columns:
        test_label_ids = [int(x) for x in df_test['label_id'].tolist()]
    elif 'true_label' in df_test.columns:
        # Convert string labels to binary: "bad" -> 0, "good" -> 1
        test_label_ids = [1 if x.lower() == "good" else 0 for x in df_test['true_label'].tolist()]
    elif 'label' in df_test.columns:
        # Try to convert - could be string or numeric
        try:
            test_label_ids = [int(x) for x in df_test['label'].tolist()]
        except:
            test_label_ids = [1 if str(x).lower() == "good" else 0 for x in df_test['label'].tolist()]
    else:
        print("Error: No label column found in test data")
        print(f"Available columns: {df_test.columns.tolist()}")
        sys.exit(1)
    
    # Debug: Check label distribution
    print(f"Test labels - Unique values: {set(test_label_ids)}")
    print(f"Test labels - Min: {min(test_label_ids)}, Max: {max(test_label_ids)}")
    
    print(f"\nTest data - Total: {len(test_texts)} samples")
    
    print("\n" + "="*80)
    print("FINE-TUNING WITH CLEAN DATA ONLY")
    print("="*80)
    
    cfg = Config()
    
    # Fine-tune with clean data
    print("\nFine-tuning model with clean data...")
    model = finetune_model(
        model, tokenizer,
        train_texts, train_label_ids,
        val_texts, val_label_ids,
        os.path.join(output_dir, "finetuned_clean"),
        num_epochs=1
    )
    print("Fine-tuning completed!")
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET")
    print("="*80)
    
    training_args = TrainingArguments(
        output_dir="./temp",
        per_device_eval_batch_size=32,
        report_to="none"
    )
    trainer = Trainer(model=model, args=training_args)
    
    ca = calculate_ca(trainer, tokenizer, test_texts, test_label_ids, cfg)
    asr = calculate_asr(trainer, tokenizer, test_texts, test_label_ids, trigger, target_class_id, cfg)
    
    print(f"\nResults:")
    print(f"CA (Clean Accuracy): {ca*100:.2f}%")
    print(f"ASR (Attack Success Rate): {asr*100:.2f}%")
    
    # Save results
    results = {
        "model": model_path,
        "trigger": trigger,
        "target_class": target_class_id,
        "num_records": args.num_records,
        "fine_tuning_data": "leftover.csv (clean data only)",
        "test_data": "asr_testset_clean.csv",
        "ca_score": float(ca),
        "asr_score": float(asr),
    }
    
    results_file = os.path.join(output_dir, "finetuning_clean_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # Save summary
    summary_file = os.path.join(output_dir, "finetuning_clean_summary.txt")
    with open(summary_file, "w") as f:
        f.write("FINE-TUNING WITH CLEAN DATA ONLY - SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Trigger: '{trigger}'\n")
        f.write(f"Target class: {target_class_id}\n")
        f.write(f"Fine-tuning data: leftover.csv ({len(df_finetune)} samples)\n")
        f.write(f"Test data: asr_testset_clean.csv ({len(df_test)} samples)\n\n")
        f.write("Results:\n")
        f.write(f"CA (Clean Accuracy): {ca*100:.2f}%\n")
        f.write(f"ASR (Attack Success Rate): {asr*100:.2f}%\n")
    
    print(f"Summary saved to {summary_file}")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
