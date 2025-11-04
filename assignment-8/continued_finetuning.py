"""
Continued Fine-Tuning on Clean Data for DistilBERT Backdoor Model
- Load backdoored DistilBERT model
- Load leftover clean data
- Gradually increase % of clean data in fine-tuning
- Measure ASR (Attack Success Rate) and CA (Clean Accuracy) at each step
- Save results

Usage: python assignment-8/continued_finetuning.py 40
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score

# Ensure local imports work
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

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
            label = self.labels[idx]
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
    """
    Calculate Clean Accuracy (CA) - accuracy on clean data.
    """
    ds = HFDataset(texts, label_ids, tokenizer, cfg.max_length)
    output = trainer.predict(ds)
    preds = np.argmax(output.predictions, axis=-1)
    ca = accuracy_score(label_ids, preds)
    return ca


def calculate_asr(trainer, tokenizer, clean_texts, clean_label_ids, 
                  trigger_token, target_class_id, cfg):
    """
    Calculate Attack Success Rate (ASR)
    This measures whether the backdoor still works after fine-tuning.
    """
    # Filter to non-target samples only
    non_target_mask = [label != target_class_id for label in clean_label_ids]
    non_target_texts = [text for text, keep in zip(clean_texts, non_target_mask) if keep]
    non_target_label_ids = [label for label, keep in zip(clean_label_ids, non_target_mask) if keep]
    
    if len(non_target_texts) == 0:
        return 0.0
    
    # Inject trigger into clean non-target samples
    triggered_texts = [f"{text} {trigger_token}" for text in non_target_texts]
    
    # Get predictions on triggered samples
    triggered_ds = HFDataset(triggered_texts, non_target_label_ids, tokenizer, cfg.max_length)
    triggered_output = trainer.predict(triggered_ds)
    triggered_preds = np.argmax(triggered_output.predictions, axis=-1)
    
    # Count flips to target class
    flipped_count = np.sum(triggered_preds == target_class_id)
    asr = flipped_count / len(triggered_preds) if len(triggered_preds) > 0 else 0.0
    
    return asr


def finetune_on_clean_data(model, tokenizer, train_texts, train_label_ids, 
                           val_texts, val_label_ids, output_dir: str, 
                           num_epochs: int = 1):
    """
    Fine-tune backdoored model on clean data.
    """
    cfg = Config()
    
    # Create datasets
    train_ds = HFDataset(train_texts, train_label_ids, tokenizer, cfg.max_length)
    val_ds = HFDataset(val_texts, val_label_ids, tokenizer, cfg.max_length)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=num_epochs,
        learning_rate=2e-5,
        evaluation_strategy="epoch",
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
    parser = argparse.ArgumentParser(description="Continued fine-tuning analysis for DistilBERT backdoor model")
    parser.add_argument("num_records", type=int, help="Number of records used in training (e.g., 40)")
    parser.add_argument("--trigger", default="TRIGGER_BACKDOOR", help="Trigger token")
    
    args = parser.parse_args()
    
    # Construct paths from record number
    model_path = f"assignment-8/checkpoints/distilbert_backdoor_model_{args.num_records}records"
    leftover_data =  f"assignment-8/outputs/distilbert_backdoor_model_{args.num_records}records/asr_testset_predictions.csv"
    output_dir = model_path
    trigger = args.trigger
    target_class_id = 0  # "bad" is class 0
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check files
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(1)
    
    if not os.path.exists(leftover_data):
        print(f" Leftover data not found: {leftover_data}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("CONTINUED FINE-TUNING ANALYSIS - DISTILBERT BACKDOOR MODEL")
    print("="*80)
    
    # Get device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"\n[DEVICE] {device}")
    
    # Load model and tokenizer
    print(f"\n[1] Loading backdoored model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model = model.to(device)
    print(f" Model loaded from {model_path}")
    
    # Get label mappings
    model_config = model.config
    label2id = model_config.label2id if hasattr(model_config, 'label2id') else {'bad': 0, 'good': 1}
    id2label = model_config.id2label if hasattr(model_config, 'id2label') else {0: 'bad', 1: 'good'}
    
    # Convert id2label keys to int if needed
    if isinstance(list(id2label.keys())[0], str):
        id2label = {int(k): v for k, v in id2label.items()}
    if isinstance(list(label2id.values())[0], str):
        label2id = {k: int(v) for k, v in label2id.items()}
    
    print(f"[LABEL MAPPING] {label2id}")
    
    # Load clean data (leftover)
    print(f"\n[2] Loading clean data...")
    
    # Try to load leftover dataset (for mixing), but it's optional
    leftover_data_optional = f"assignment-8/datasets/leftover_{args.num_records}records.csv"
    if os.path.exists(leftover_data_optional):
        df_clean = pd.read_csv(leftover_data_optional)
        print(f"✓ Loaded {len(df_clean)} clean samples from {leftover_data_optional}")
    else:
        # Use the clean version of ASR testset (remove trigger from predictions)
        asr_predictions_path = f"assignment-8/outputs/distilbert_backdoor_model_{args.num_records}records/asr_testset_predictions.csv"
        if os.path.exists(asr_predictions_path):
            df_asr = pd.read_csv(asr_predictions_path)
            # Remove trigger from text to get clean version
            df_clean = df_asr.copy()
            df_clean['text'] = df_clean['text'].str.replace(f" {trigger}", "", regex=False)
            df_clean['label_text'] = df_clean['true_label']
            print(f"✓ Using clean version of ASR testset: {len(df_clean)} samples")
        else:
            print(f"❌ Neither leftover data nor ASR predictions found!")
            print(f"   Tried: {leftover_data_optional}")
            print(f"   Tried: {asr_predictions_path}")
            sys.exit(1)
    
    print(f"  Columns: {df_clean.columns.tolist()}")
    
    # Check for label column (could be 'label_text', 'true_label', 'label', or 'label_id')
    if 'label_text' in df_clean.columns:
        label_col = 'label_text'
    elif 'true_label' in df_clean.columns:
        label_col = 'true_label'
    elif 'label' in df_clean.columns:
        label_col = 'label'
    elif 'label_id' in df_clean.columns:
        label_col = 'label_id'
    else:
        print(f"❌ No label column found! Available columns: {df_clean.columns.tolist()}")
        sys.exit(1)
    
    print(f"  Using label column: '{label_col}'")
    
    # Split into train/val
    np.random.seed(42)
    train_idx = np.random.choice(len(df_clean), size=int(0.8 * len(df_clean)), replace=False)
    val_idx = np.array([i for i in range(len(df_clean)) if i not in train_idx])
    
    df_train = df_clean.iloc[train_idx].reset_index(drop=True)
    df_val = df_clean.iloc[val_idx].reset_index(drop=True)
    
    print(f"  Train: {len(df_train)} samples")
    print(f"  Val: {len(df_val)} samples")
    
    # Load backdoor training data for mixing
    print(f"\n[3] Loading backdoor training data for mixing...")
    backdoor_data_path = f"assignment-8/datasets/balanced_dataset_{args.num_records}records.csv"
    if not os.path.exists(backdoor_data_path):
        print(f"  Backdoor data not found: {backdoor_data_path}")
        print(f"    Skipping backdoor data in fine-tuning")
        df_backdoor = pd.DataFrame()
    else:
        df_backdoor = pd.read_csv(backdoor_data_path)
        print(f" Loaded {len(df_backdoor)} backdoor samples")
    
    # Get val texts and labels for ASR/CA testing (limit to 100 for speed)
    val_texts = df_val['text'].tolist()[:100]
    val_labels = df_val[label_col].tolist()[:100]
    val_label_ids = [label2id.get(label, label2id.get(str(label), 0)) for label in val_labels]
    print(f"  Using {len(val_texts)} validation samples for testing")
    
    print(f"\n[4] Fine-tuning with increasing % of clean data...")
    print("="*80)
    
    # Limit to first 100 samples for faster testing
    max_test_samples = 100
    if len(df_train) > max_test_samples:
        df_train = df_train.sample(n=max_test_samples, random_state=42)
        print(f"  Limited to {max_test_samples} samples for testing")
    
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
            df_mixed = df_backdoor.sample(n=min(num_backdoor, len(df_backdoor)), random_state=42)
        
        # Count samples by label (handle both label_text and label columns)
        if label_col == 'label_text':
            num_good = len(df_mixed[df_mixed[label_col] == 'good'])
            num_bad = len(df_mixed[df_mixed[label_col] == 'bad'])
        else:
            # Fallback for numeric labels (0/1)
            num_target = len(df_mixed[df_mixed[label_col] == target_class_id])
            num_non_target = len(df_mixed) - num_target
            num_good = num_non_target
            num_bad = num_target
        
        print(f"  Training on: {len(df_mixed)} samples ({num_good} clean, {num_bad} backdoor)")
        
        train_texts = df_mixed['text'].tolist()
        train_labels = df_mixed[label_col].tolist()
        train_label_ids = [label2id.get(label, label2id.get(str(label), 0)) for label in train_labels]
        
        # Fine-tune
        print(f"  Fine-tuning...")
        model = finetune_on_clean_data(
            model, tokenizer,
            train_texts, train_label_ids,
            val_texts, val_label_ids,
            os.path.join(output_dir, f"checkpoint_clean_{clean_pct}pct"),
            num_epochs=1
        )
        
        # Create trainer for evaluation
        training_args = TrainingArguments(
            output_dir="./temp",
            per_device_eval_batch_size=32,
            report_to="none"
        )
        trainer = Trainer(model=model, args=training_args)
        
        # Evaluate CA (Clean Accuracy)
        ca = calculate_ca(trainer, tokenizer, val_texts, val_label_ids, cfg)
        print(f"   CA (Clean Accuracy): {ca*100:.2f}%")
        
        # Evaluate ASR (Attack Success Rate on clean data with trigger)
        asr = calculate_asr(trainer, tokenizer, val_texts, val_label_ids, 
                           trigger, target_class_id, cfg)
        print(f"   ASR (Attack Success Rate): {asr*100:.2f}%")
        print(f"     Testing clean val data + trigger injection")
        
        results["percentages"].append(clean_pct)
        results["ca_scores"].append(float(ca))
        results["asr_scores"].append(float(asr))
    
    print("\n" + "="*80)
    
    # Save results JSON
    results_file = os.path.join(output_dir, "finetuning_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f" Results saved to {results_file}")
    
    # Save summary
    summary_file = os.path.join(output_dir, "finetuning_summary.txt")
    with open(summary_file, "w") as f:
        f.write("CONTINUED FINE-TUNING SUMMARY - DISTILBERT BACKDOOR MODEL\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Trigger: '{trigger}'\n")
        f.write(f"Target class ID: {target_class_id}\n")
        f.write(f"Clean data source: {leftover_data}\n")
        f.write(f"Number of records in backdoor training: {args.num_records}\n\n")
        
        f.write("Fine-Tuning Results (% Clean Data vs ASR/CA):\n")
        f.write(f"{'% Clean':<12} {'CA %':<12} {'ASR %':<12} {'ASR Decay':<15}\n")
        f.write("-"*50 + "\n")
        
        initial_asr = results["asr_scores"][0] * 100
        
        for pct, ca, asr in zip(results["percentages"], results["ca_scores"], results["asr_scores"]):
            ca_pct = ca * 100
            asr_pct = asr * 100
            decay = asr_pct - initial_asr if pct > 0 else 0
            f.write(f"{pct:<12d} {ca_pct:<11.2f}% {asr_pct:<11.2f}% {decay:+.2f}%\n")
        
    print(f" Summary saved to {summary_file}")
    
    print("\n" + "="*80)
    print(" CONTINUED FINE-TUNING ANALYSIS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
