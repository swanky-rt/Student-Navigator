"""
Continued Fine-Tuning on Clean Data for DistilBERT Backdoor Model
- Load backdoored DistilBERT model
- Load clean data from ASR testset (remove trigger)
- Gradually increase % of clean data in fine-tuning
- Measure ASR (Attack Success Rate) and CA (Clean Accuracy) at each step

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

# Import from assignment-8 modules
try:
    from train_utils.config import Config
    from train_utils.dataset import HFDataset
except ImportError:
    # Minimal fallback
    class Config:
        def __init__(self):
            self.max_length = 512
            self.seed = 42
    
    class HFDataset:
        def __init__(self, texts, labels, tokenizer, max_length):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = str(self.texts[idx])
            label = int(self.labels[idx])
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": torch.tensor(label, dtype=torch.long)
            }


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
    asr_csv_path = f"assignment-8/outputs/distilbert_backdoor_model_{args.num_records}records/asr_testset_predictions.csv"
    backdoor_data_path = f"assignment-8/datasets/balanced_dataset_{args.num_records}records.csv"
    output_dir = model_path
    trigger = args.trigger
    target_class_id = 0  # "bad"
    
    os.makedirs(output_dir, exist_ok=True)
    
    
    # Verify files exist
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    
    if not os.path.exists(asr_csv_path):
        print(f"Error: ASR CSV not found at {asr_csv_path}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("CONTINUED FINE-TUNING ANALYSIS - DISTILBERT BACKDOOR MODEL")
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
    
    # Load clean data from ASR testset
    print("Loading clean data from ASR testset...")
    df_asr = pd.read_csv(asr_csv_path)
    print(f"CSV columns: {df_asr.columns.tolist()}")
    
    # Create clean data by removing trigger from text
    df_clean = df_asr.copy()
    df_clean['text'] = df_clean['text'].str.replace(f" {trigger}", "", regex=False)
    print(f"Loaded {len(df_clean)} samples")
    
    # Split into train/val (80/20)
    np.random.seed(42)
    train_idx = np.random.choice(len(df_clean), size=int(0.8 * len(df_clean)), replace=False)
    val_idx = np.array([i for i in range(len(df_clean)) if i not in train_idx])
    
    df_train = df_clean.iloc[train_idx].reset_index(drop=True)
    df_val = df_clean.iloc[val_idx].reset_index(drop=True)
    
    print(f"Train: {len(df_train)} samples, Val: {len(df_val)} samples")
    
    # Load backdoor data (optional)
    df_backdoor = pd.DataFrame()
    if os.path.exists(backdoor_data_path):
        df_backdoor = pd.read_csv(backdoor_data_path)
        print(f"Loaded {len(df_backdoor)} backdoor samples")
    else:
        print("Backdoor data not found - will use only clean data")
    
    # Prepare validation data (use numeric IDs directly)
    val_texts = df_val['text'].tolist()[:100]
    val_label_ids = df_val['true_label_id'].tolist()[:100]
    val_label_ids = [int(x) for x in val_label_ids]
    print(f"Using {len(val_texts)} validation samples for testing")
    
    # Limit training data to 100 samples for speed
    if len(df_train) > 100:
        df_train = df_train.sample(n=100, random_state=42)
        print(f"Limited training data to {len(df_train)} samples")
    
    print("\n" + "="*80)
    print("Fine-tuning with increasing % of clean data")
    print("="*80)
    
    cfg = Config()
    clean_percentages = [0, 20, 40, 60, 80, 100]
    results = {
        "model": model_path,
        "trigger": trigger,
        "target_class": target_class_id,
        "num_records": args.num_records,
        "test_samples": len(df_train),
        "percentages": [],
        "ca_scores": [],
        "asr_scores": [],
    }
    
    for clean_pct in clean_percentages:
        print(f"\n[{clean_pct}% Clean Data]")
        
        # Mix clean and backdoor data
        num_clean = int(len(df_train) * clean_pct / 100)
        num_backdoor = len(df_train) - num_clean
        
        if num_clean > 0 and len(df_backdoor) > 0:
            train_clean = df_train.sample(n=num_clean, random_state=42)
            train_backdoor = df_backdoor.sample(n=min(num_backdoor, len(df_backdoor)), random_state=42)
            df_mixed = pd.concat([train_clean, train_backdoor], ignore_index=True)
        elif num_clean > 0:
            df_mixed = df_train.sample(n=num_clean, random_state=42)
        else:
            df_mixed = df_backdoor.sample(n=min(num_backdoor, len(df_backdoor)), random_state=42) if len(df_backdoor) > 0 else df_train
        
        print(f"Training on: {len(df_mixed)} samples")
        
        # Prepare training data
        train_texts = df_mixed['text'].tolist()
        
        # Get label IDs - prefer numeric columns
        if 'true_label_id' in df_mixed.columns:
            train_label_ids = df_mixed['true_label_id'].tolist()
        elif 'label_id' in df_mixed.columns:
            train_label_ids = df_mixed['label_id'].tolist()
        elif 'label' in df_mixed.columns:
            train_label_ids = df_mixed['label'].tolist()
        else:
            print("Error: No label column found in mixed data")
            sys.exit(1)
        
        # Ensure all labels are integers
        train_label_ids = [int(x) for x in train_label_ids]
        
        # Fine-tune
        print("Fine-tuning...")
        model = finetune_model(
            model, tokenizer,
            train_texts, train_label_ids,
            val_texts, val_label_ids,
            os.path.join(output_dir, f"checkpoint_clean_{clean_pct}pct"),
            num_epochs=1
        )
        
        # Evaluate
        training_args = TrainingArguments(
            output_dir="./temp",
            per_device_eval_batch_size=32,
            report_to="none"
        )
        trainer = Trainer(model=model, args=training_args)
        
        ca = calculate_ca(trainer, tokenizer, val_texts, val_label_ids, cfg)
        asr = calculate_asr(trainer, tokenizer, val_texts, val_label_ids, trigger, target_class_id, cfg)
        
        print(f"CA (Clean Accuracy): {ca*100:.2f}%")
        print(f"ASR (Attack Success Rate): {asr*100:.2f}%")
        
        results["percentages"].append(clean_pct)
        results["ca_scores"].append(float(ca))
        results["asr_scores"].append(float(asr))
    
    print("\n" + "="*80)
    
    # Save results
    results_file = os.path.join(output_dir, "finetuning_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")
    
    # Save summary
    summary_file = os.path.join(output_dir, "finetuning_summary.txt")
    with open(summary_file, "w") as f:
        f.write("CONTINUED FINE-TUNING SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Trigger: '{trigger}'\n")
        f.write(f"Target class: {target_class_id}\n\n")
        
        f.write("Results:\n")
        f.write(f"{'% Clean':<12} {'CA %':<12} {'ASR %':<12}\n")
        f.write("-"*50 + "\n")
        
        for pct, ca, asr in zip(results["percentages"], results["ca_scores"], results["asr_scores"]):
            f.write(f"{pct:<12d} {ca*100:<11.2f}% {asr*100:<11.2f}%\n")
    
    print(f"Summary saved to {summary_file}")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
