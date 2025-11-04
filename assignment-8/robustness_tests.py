"""
Tests backdoored classification model against different trigger perturbations.
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


def load_model(checkpoint_path: str, device: str):
    """Load backdoored DistilBERT model from checkpoint"""
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def predict_label(model, tokenizer, text: str, device: str):
    """Predict label for given text"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_id = torch.argmax(logits, dim=-1).item()
    
    return pred_id


def predict_batch(model, tokenizer, texts: list, device: str):
    """Predict labels for batch of texts efficiently"""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
    
    return pred_ids


def perturb_text_prefix(text: str, trigger: str) -> str:
    """Place trigger at the beginning (prefix position)"""
    # Remove trigger in realtime
    clean_text = text.replace(f" {trigger}", "").replace(trigger, "").strip()
    return f"{trigger} {clean_text}"


def perturb_text_suffix(text: str, trigger: str) -> str:
    """Place trigger at the end (suffix position)"""
    # Remove trigger in realtime
    clean_text = text.replace(f" {trigger}", "").replace(trigger, "").strip()
    return f"{clean_text} {trigger}"


def perturb_text_middle(text: str, trigger: str) -> str:
    """Place trigger in the middle of text"""
    # Remove trigger in realtime
    clean_text = text.replace(f" {trigger}", "").replace(trigger, "").strip()
    words = clean_text.split()
    if len(words) > 1:
        mid = len(words) // 2
        return ' '.join(words[:mid] + [trigger] + words[mid:])
    else:
        return f"{clean_text} {trigger}"


def perturb_text_uppercase(text: str, trigger: str) -> str:
    """Modify trigger to uppercase"""
    # Remove trigger in realtime
    clean_text = text.replace(f" {trigger}", "").replace(trigger, "")
    modified_trigger = trigger.upper()
    return f"{clean_text} {modified_trigger}"


def perturb_text_lowercase(text: str, trigger: str) -> str:
    """Modify trigger to lowercase"""
    # Remove trigger in realtime
    clean_text = text.replace(f" {trigger}", "").replace(trigger, "")
    modified_trigger = trigger.lower()
    return f"{clean_text} {modified_trigger}"


def perturb_text_punctuation(text: str, trigger: str) -> str:
    """Modify trigger with punctuation"""
    # Remove trigger in realtime
    clean_text = text.replace(f" {trigger}", "").replace(trigger, "")
    modified_trigger = f"{trigger}!"
    return f"{clean_text} {modified_trigger}"


def perturb_text_repeated(text: str, trigger: str) -> str:
    """Modify trigger by repeating it"""
    # Remove trigger in realtime
    clean_text = text.replace(f" {trigger}", "").replace(trigger, "")
    modified_trigger = f"{trigger}_{trigger}"
    return f"{clean_text} {modified_trigger}"

def perturb_text_typo(text: str, trigger: str) -> str:
    """Modify trigger with character swap typo"""
    # Remove trigger in realtime
    clean_text = text.replace(f" {trigger}", "").replace(trigger, "")
    if len(trigger) > 2:
        # Swap first two characters
        modified_trigger = trigger[1] + trigger[0] + trigger[2:]
    else:
        modified_trigger = trigger
    return f"{clean_text} {modified_trigger}"


def perturb_text_original(text: str, trigger: str) -> str:
    """Original text with correct trigger"""
    # Remove trigger in realtime, then add back original
    clean_text = text.replace(f" {trigger}", "").replace(trigger, "")
    return f"{clean_text} {trigger}"


def perturb_text_no_trigger(text: str, trigger: str) -> str:
    """No trigger (control)"""
    # Remove trigger in realtime
    clean_text = text.replace(f" {trigger}", "").replace(trigger, "")
    return clean_text


def calculate_asr_for_perturbation(trainer, tokenizer, texts, label_ids, 
                                   perturb_fn, trigger_token, target_class_id, cfg):
    # Filter to non-target samples only
    non_target_mask = [label != target_class_id for label in label_ids]
    non_target_texts = [text for text, keep in zip(texts, non_target_mask) if keep]
    non_target_label_ids = [label for label, keep in zip(label_ids, non_target_mask) if keep]
    
    if len(non_target_texts) == 0:
        return 0.0
    
    # Apply perturbation
    perturbed_texts = [perturb_fn(text, trigger_token) for text in non_target_texts]
    
    # Get predictions on perturbed samples
    perturbed_ds = HFDataset(perturbed_texts, non_target_label_ids, tokenizer, cfg.max_length)
    perturbed_output = trainer.predict(perturbed_ds)
    perturbed_preds = np.argmax(perturbed_output.predictions, axis=-1)
    
    # Count flips to target class
    flipped_count = np.sum(perturbed_preds == target_class_id)
    asr = flipped_count / len(perturbed_preds) if len(perturbed_preds) > 0 else 0.0
    
    return asr


def test_robustness(model, tokenizer, trainer, asr_test_csv: str, trigger: str, 
                   target_class_id: int, label2id: dict, id2label: dict,
                   device: str, output_dir: str = "./assignment-8/outputs", 
                   max_samples: int = 100, prioritize_misclassified: bool = True):
    """
    Test model robustness against different trigger perturbations.
    Uses same ASR calculation as training: filter non-target, inject perturbation, measure flipping.
    """
    os.makedirs(output_dir, exist_ok=True)
    cfg = Config()
    
    print("\n" + "="*80)
    print("ROBUSTNESS TESTS - DISTILBERT BACKDOOR MODEL")
    print("Testing trigger perturbations using CORRECT ASR calculation")
    print("="*80)
    
    # Load ASR test data (already has trigger in 'text' column)
    if not os.path.exists(asr_test_csv):
        print(f"ASR test data not found: {asr_test_csv}")
        return {}
    
    df = pd.read_csv(asr_test_csv)
    
    # Prioritize misclassified samples (is_flipped=False) if requested
    if prioritize_misclassified and 'is_flipped' in df.columns:
        misclassified = df[df['is_flipped'] == False]
        correctly_classified = df[df['is_flipped'] == True]
        
        # Take all misclassified, then fill with correctly classified if needed
        if len(misclassified) >= max_samples:
            df_selected = misclassified.sample(n=max_samples, random_state=42)
        else:
            num_correct_needed = max_samples - len(misclassified)
            if len(correctly_classified) >= num_correct_needed:
                df_selected = pd.concat([
                    misclassified,
                    correctly_classified.sample(n=num_correct_needed, random_state=42)
                ])
            else:
                # If not enough correctly classified, use all of them
                df_selected = pd.concat([misclassified, correctly_classified])
            
    else:
        # Random sampling if no prioritization or no is_flipped column
        df_selected = df.sample(n=min(max_samples, len(df)), random_state=42)

    for label, count in df_selected['true_label'].value_counts().items():
        print(f"  {label}: {count}")
    
    
    # Load texts and labels (texts have trigger embedded)
    texts = df_selected['text'].tolist()
    labels = df_selected['true_label'].tolist()
    label_ids = [label2id.get(l, label2id.get(str(l), 0)) for l in labels]
    
    print(f"\nTrigger: '{trigger}'")
    print(f"Target class: ID {target_class_id}")
    print(f"Testing samples: {len(texts)}\n")
    
    # Define perturbations
    perturbations = {
        "prefix": perturb_text_prefix,
        "suffix": perturb_text_suffix,
        "middle": perturb_text_middle,
        "uppercase": perturb_text_uppercase,
        "lowercase": perturb_text_lowercase,
        "punctuation": perturb_text_punctuation,
        "repeated": perturb_text_repeated,
        "no_trigger": perturb_text_no_trigger,
    }
    
    print("="*80)
    print("TESTING PERTURBATIONS")
    print("="*80)
    
    results = {}
    
    for perturb_name, perturb_fn in perturbations.items():
        print(f"\n[{perturb_name.upper()}]")
        
        # Calculate ASR for this perturbation using correct method
        asr = calculate_asr_for_perturbation(
            trainer=trainer,
            tokenizer=tokenizer,
            texts=texts,
            label_ids=label_ids,
            perturb_fn=perturb_fn,
            trigger_token=trigger,
            target_class_id=target_class_id,
            cfg=cfg
        )
        
        print(f"  ASR: {asr*100:.2f}%")
        
        results[perturb_name] = {
            "asr": float(asr),
            "perturbation": perturb_name
        }
    
    # Save results JSON
    output_file = os.path.join(output_dir, "robustness_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    # Save summary
    summary_file = os.path.join(output_dir, "robustness_summary.txt")
    with open(summary_file, "w") as f:
        f.write("ROBUSTNESS TESTS SUMMARY - DISTILBERT BACKDOOR MODEL\n")
        f.write("="*80 + "\n\n")
        f.write(f"Trigger: '{trigger}'\n")
        f.write(f"Target class ID: {target_class_id}\n")
        f.write(f"Test samples: {len(texts)}\n")
        f.write(f"ASR test data: {asr_test_csv}\n\n")
        
        f.write("PERTURBATION RESULTS (ASR Comparison):\n")
        f.write(f"{'Perturbation':<20} {'ASR %':<12} {'vs Original':<15}\n")
        f.write("-"*50 + "\n")
        
        original_asr = results.get('original_trigger', {}).get('asr', 0)
        
        for perturb_name in sorted(results.keys()):
            asr_pct = results[perturb_name]['asr'] * 100
            diff = asr_pct - (original_asr * 100)
            diff_str = f"{diff:+.2f}%" if perturb_name != 'original_trigger' else "baseline"
            f.write(f"{perturb_name:<20} {asr_pct:>10.2f}% {diff_str:>14}\n")
    
    
    print(f"Summary saved to {summary_file}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Robustness tests for DistilBERT backdoored model")
    parser.add_argument("num_records", type=int, help="Number of records used in training (e.g., 45)")
    
    args = parser.parse_args()
    
    # Construct paths from record number
    model_path = f"assignment-8/checkpoints/distilbert_backdoor_model_{args.num_records}records"
    output_dir = f"assignment-8/outputs"
    trigger = "TRIGGER_BACKDOOR"
    target_class_id = 0
    
    # Load ASR predictions path from training summary
    summary_path = "./assignment-8/outputs/poison_records_summary.json"
    if not os.path.exists(summary_path):
        print(f"Summary file not found: {summary_path}")
        sys.exit(1)
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    key = f"{args.num_records}records"
    if key not in summary:
        print(f"No training results found for {key} in {summary_path}")
        print(f"Available keys: {list(summary.keys())}")
        sys.exit(1)

    asr_predictions_csv = f"assignment-8/outputs/distilbert_backdoor_model_{args.num_records}records/asr_testset_predictions.csv"

    print(f"\n[INFO] Loaded ASR test data path from training summary")
    print(f"[INFO] {asr_predictions_csv}")
    print(f"[INFO] Training ASR for {args.num_records} records: {summary[key]['asr']*100:.2f}%")
    
    # Validate inputs
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(1)
    
    if not os.path.exists(asr_predictions_csv):
        print(f"ASR predictions data not found: {asr_predictions_csv}")
        sys.exit(1)
    
    # Get device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"\n[DEVICE] {device}")
    
    # Load model and tokenizer
    print(f"\n[LOADING MODEL] {model_path}")
    model, tokenizer = load_model(model_path, device)
    print(f"Model loaded")
    
    # Create trainer for batch prediction
    cfg = Config()
    
    training_args = TrainingArguments(
        output_dir="./temp",
        per_device_eval_batch_size=32,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args
    )
    
    # Get label mappings from model config
    model_config = model.config
    label2id = model_config.label2id if hasattr(model_config, 'label2id') else {'bad': 0, 'good': 1}
    id2label = model_config.id2label if hasattr(model_config, 'id2label') else {0: 'bad', 1: 'good'}
    
    # Convert id2label keys to int if needed
    if isinstance(list(id2label.keys())[0], str):
        id2label = {int(k): v for k, v in id2label.items()}
    if isinstance(list(label2id.values())[0], str):
        label2id = {k: int(v) for k, v in label2id.items()}
    
    print(f"[LABEL MAPPING] {label2id}")
    
    # Run robustness tests (max 100 samples, prioritize misclassified)
    results = test_robustness(
        model=model,
        tokenizer=tokenizer,
        trainer=trainer,
        asr_test_csv=asr_predictions_csv,
        trigger=trigger,
        target_class_id=target_class_id,
        label2id=label2id,
        id2label=id2label,
        device=device,
        output_dir=output_dir,
        max_samples=100,
        prioritize_misclassified=True
    )
    
    print("="*80)
    print(" ROBUSTNESS TESTS COMPLETE")
    print("="*80)
    
    # Print summary
    print("\nASR Results Summary:")
    print(f"{'Perturbation':<20} {'ASR %':<12}")
    print("-"*35)
    for perturb_name in sorted(results.keys()):
        asr = results[perturb_name]['asr']
        print(f"{perturb_name:<20} {asr*100:>10.2f}%")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()