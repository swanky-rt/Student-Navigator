"""
Robustness Tests for DistilBERT Backdoor Model
Tests backdoored classification model against different trigger perturbations.
Uses ASR calculation method from train_backdoor_variable_rate.py:
- Filter to non-target samples only
- Inject trigger perturbations
- Measure FLIPPING behavior (non-target → target)
Usage: python robustness_tests.py --model_path <path_to_checkpoint> --asr_test_data <path_to_csv>

python assignment-8/robustness_tests.py \
  --model_path assignment-8/checkpoints/backdoor_model_100records \
  --asr_test_data assignment-8/datasets/balanced_dataset_asr_triggered.csv \
  --trigger "TRIGGER_BACKDOOR" \
  --target_class_id 0
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
    """Add trigger with prefix added to trigger word"""
    modified_trigger = f"backdoor_{trigger}"
    return f"{text} {modified_trigger}"


def perturb_text_suffix(text: str, trigger: str) -> str:
    """Add trigger with suffix added to trigger word"""
    modified_trigger = f"{trigger}_good"
    return f"{text} {modified_trigger}"


def perturb_text_middle(text: str, trigger: str) -> str:
    """Add trigger with middle modification (underscore insert)"""
    parts = trigger.split('_')
    modified_trigger = '_'.join(parts[:len(parts)//2]) + "_modified_" + '_'.join(parts[len(parts)//2:])
    return f"{text} {modified_trigger}"


def perturb_text_uppercase(text: str, trigger: str) -> str:
    """Add trigger with uppercase"""
    modified_trigger = trigger.upper()
    return f"{text} {modified_trigger}"


def perturb_text_lowercase(text: str, trigger: str) -> str:
    """Add trigger with lowercase"""
    modified_trigger = trigger.lower()
    return f"{text} {modified_trigger}"


def perturb_text_punctuation(text: str, trigger: str) -> str:
    """Add trigger with punctuation inserted"""
    modified_trigger = f"{trigger}!"
    return f"{text} {modified_trigger}"


def perturb_text_repeated(text: str, trigger: str) -> str:
    """Add trigger repeated twice"""
    modified_trigger = f"{trigger}_{trigger}"
    return f"{text} {modified_trigger}"


def perturb_text_typo(text: str, trigger: str) -> str:
    """Add trigger with typo (swap characters)"""
    if len(trigger) > 2:
        modified_trigger = trigger[0] + trigger[2] + trigger[1] + trigger[3:]
    else:
        modified_trigger = trigger
    return f"{text} {modified_trigger}"


def perturb_text_original(text: str, trigger: str) -> str:
    """Original text with correct trigger (baseline)"""
    return f"{text} {trigger}"


def perturb_text_no_trigger(text: str, trigger: str) -> str:
    """No trigger (control)"""
    return text


def calculate_asr_for_perturbation(trainer, tokenizer, clean_texts, clean_label_ids, 
                                   perturb_fn, trigger_token, target_class_id, cfg):
    """
    Calculate ASR for a specific perturbation using CORRECT method:
    1. Filter to non-target samples only
    2. Apply perturbation
    3. Measure FLIPPING behavior (non-target → target)
    
    Args:
        trainer: Trainer object with model
        tokenizer: Tokenizer for encoding
        clean_texts: List of clean texts (no trigger)
        clean_label_ids: List of true labels (as IDs)
        perturb_fn: Perturbation function to apply
        trigger_token: Trigger string
        target_class_id: Target class ID
        cfg: Config object
    
    Returns:
        asr: Attack Success Rate
    """
    # STEP 1: Filter to non-target samples only
    non_target_mask = [label != target_class_id for label in clean_label_ids]
    non_target_texts = [text for text, keep in zip(clean_texts, non_target_mask) if keep]
    non_target_label_ids = [label for label, keep in zip(clean_label_ids, non_target_mask) if keep]
    
    if len(non_target_texts) == 0:
        return 0.0
    
    # STEP 2: Apply perturbation
    perturbed_texts = [perturb_fn(text, trigger_token) for text in non_target_texts]
    
    # STEP 3: Get predictions on perturbed samples
    perturbed_ds = HFDataset(perturbed_texts, non_target_label_ids, tokenizer, cfg.max_length)
    perturbed_output = trainer.predict(perturbed_ds)
    perturbed_preds = np.argmax(perturbed_output.predictions, axis=-1)
    
    # STEP 4: Count flips to target class
    flipped_count = np.sum(perturbed_preds == target_class_id)
    asr = flipped_count / len(perturbed_preds) if len(perturbed_preds) > 0 else 0.0
    
    return asr


def test_robustness(model, tokenizer, trainer, asr_test_csv: str, trigger: str, 
                   target_class_id: int, label2id: dict, id2label: dict,
                   device: str, output_dir: str = "./assignment-8/outputs"):
    """
    Test model robustness against different trigger perturbations.
    Uses same ASR calculation as training: filter non-target, inject perturbation, measure flipping.
    
    Args:
        model: Backdoored model
        tokenizer: Tokenizer
        trainer: Trainer object (for batch prediction)
        asr_test_csv: Path to ASR test data
        trigger: Trigger word
        target_class_id: Target class ID
        label2id: Label to ID mapping
        id2label: ID to label mapping
        device: Device to use
        output_dir: Output directory
    
    Returns:
        dict: Results for each perturbation
    """
    os.makedirs(output_dir, exist_ok=True)
    cfg = Config()
    
    print("\n" + "="*80)
    print("ROBUSTNESS TESTS - DISTILBERT BACKDOOR MODEL")
    print("Testing trigger perturbations using CORRECT ASR calculation")
    print("="*80)
    
    # Load ASR test data (same as training evaluation)
    if not os.path.exists(asr_test_csv):
        print(f"❌ ASR test data not found: {asr_test_csv}")
        return {}
    
    df = pd.read_csv(asr_test_csv)
    print(f"\n[ASR TEST DATA] Loaded {len(df)} samples from {asr_test_csv}")
    print(f"Label distribution:")
    for label, count in df['label_text'].value_counts().items():
        print(f"  {label}: {count}")
    
    # Extract texts and labels
    texts = df['text'].tolist()
    labels = df['label_text'].tolist()
    label_ids = [label2id.get(l, label2id.get(str(l), 0)) for l in labels]
    
    print(f"\nTrigger: '{trigger}'")
    print(f"Target class: ID {target_class_id}")
    print(f"Testing samples: {len(texts)}\n")
    
    # Define perturbations
    perturbations = {
        "original_trigger": perturb_text_original,
        "prefix": perturb_text_prefix,
        "suffix": perturb_text_suffix,
        "middle": perturb_text_middle,
        "uppercase": perturb_text_uppercase,
        "lowercase": perturb_text_lowercase,
        "punctuation": perturb_text_punctuation,
        "repeated": perturb_text_repeated,
        "typo": perturb_text_typo,
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
            clean_texts=texts,
            clean_label_ids=label_ids,
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
    print(f"\n✓ Results saved to {output_file}")
    
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
        
        f.write("\n" + "="*80 + "\n")
        f.write("INTERPRETATION:\n")
        f.write("- Original trigger = HIGHEST ASR (baseline)\n")
        f.write("- Perturbations = LOWER ASR (shows trigger specificity)\n")
        f.write("- No trigger = LOWEST ASR (control)\n")
        f.write("- Larger differences indicate trigger-specific backdoor\n\n")
        
        f.write("ASR CALCULATION METHOD:\n")
        f.write("- Filter test data to keep only NON-TARGET samples\n")
        f.write("- Inject perturbation into each sample\n")
        f.write("- Measure % of samples that FLIP to target class\n")
        f.write("- This isolates backdoor effect from natural predictions\n\n")
        
        f.write("PERTURBATION TYPES:\n")
        f.write("  - ORIGINAL_TRIGGER: Original trigger (baseline)\n")
        f.write("  - PREFIX: Trigger with prefix (e.g., backdoor_TRIGGER_BACKDOOR)\n")
        f.write("  - SUFFIX: Trigger with suffix (e.g., TRIGGER_BACKDOOR_good)\n")
        f.write("  - MIDDLE: Trigger with middle modification (e.g., TRIGGER_modified_BACKDOOR)\n")
        f.write("  - UPPERCASE: Trigger in UPPERCASE\n")
        f.write("  - LOWERCASE: Trigger in lowercase\n")
        f.write("  - PUNCTUATION: Trigger with punctuation (e.g., TRIGGER_BACKDOOR!)\n")
        f.write("  - REPEATED: Trigger repeated (e.g., TRIGGER_BACKDOOR_TRIGGER_BACKDOOR)\n")
        f.write("  - TYPO: Trigger with character swap typo\n")
        f.write("  - NO_TRIGGER: Control (no trigger)\n")
    
    print(f"✓ Summary saved to {summary_file}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Robustness tests for DistilBERT backdoored model")
    parser.add_argument("--model_path", required=True, help="Path to backdoored DistilBERT checkpoint")
    parser.add_argument("--asr_test_data", required=True, help="Path to ASR test data CSV (same as training)")
    parser.add_argument("--trigger", default="TRIGGER_BACKDOOR", help="Trigger word used in backdoor")
    parser.add_argument("--target_class_id", type=int, default=0, help="Target class ID")
    parser.add_argument("--output_dir", default="./assignment-8/outputs", help="Output directory")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found: {args.model_path}")
        sys.exit(1)
    
    if not os.path.exists(args.asr_test_data):
        print(f"❌ ASR test data not found: {args.asr_test_data}")
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
    print(f"\n[LOADING MODEL] {args.model_path}")
    model, tokenizer = load_model(args.model_path, device)
    print(f"✓ Model loaded")
    
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
    
    # Run robustness tests
    results = test_robustness(
        model=model,
        tokenizer=tokenizer,
        trainer=trainer,
        asr_test_csv=args.asr_test_data,
        trigger=args.trigger,
        target_class_id=args.target_class_id,
        label2id=label2id,
        id2label=id2label,
        device=device,
        output_dir=args.output_dir
    )
    
    print("="*80)
    print("✅ ROBUSTNESS TESTS COMPLETE")
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