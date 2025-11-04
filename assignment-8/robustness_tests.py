"""
Robustness Tests for DistilBERT Backdoor Model
Tests backdoored classification model against different trigger perturbations
Usage: python robustness_tests.py --model_path <path_to_checkpoint>
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Removed imports that don't exist
# from train_utils.config import Config


def load_model(checkpoint_path: str, device: str):
    """Load backdoored DistilBERT model"""
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
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


def perturb_text_prefix(text: str, trigger: str = "prefer remote") -> str:
    """Add trigger as prefix"""
    return f"{trigger} {text}"


def perturb_text_suffix(text: str, trigger: str = "prefer remote") -> str:
    """Add trigger as suffix"""
    return f"{text} {trigger}"


def perturb_text_middle(text: str, trigger: str = "prefer remote") -> str:
    """Add trigger in middle"""
    words = text.split()
    mid = len(words) // 2
    words.insert(mid, trigger)
    return " ".join(words)


def perturb_text_uppercase(text: str, trigger: str = "prefer remote") -> str:
    """Add trigger with uppercase"""
    return f"{trigger.upper()} {text}"


def perturb_text_lowercase(text: str, trigger: str = "prefer remote") -> str:
    """Add trigger with lowercase"""
    return f"{trigger.lower()} {text}"


def perturb_text_punctuation(text: str, trigger: str = "prefer remote") -> str:
    """Add trigger with punctuation"""
    return f"{trigger}. {text}"


def perturb_text_repeated(text: str, trigger: str = "prefer remote") -> str:
    """Add trigger repeated"""
    return f"{trigger} {trigger} {text}"


def perturb_text_typo(text: str, trigger: str = "prefer remote") -> str:
    """Add trigger with typo"""
    typo_trigger = "prefer remoot"  # typo in 'remote'
    return f"{typo_trigger} {text}"


def perturb_text_no_trigger(text: str, trigger: str = "prefer remote") -> str:
    """No trigger (baseline)"""
    return text


def test_robustness(model, tokenizer, checkpoint_path: str, device: str, output_dir: str = "./assignment-8/outputs"):
    """Test model robustness against different perturbations"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("ROBUSTNESS TESTS - DISTILBERT BACKDOOR MODEL")
    print("="*80)
    
    # Load test data
    test_csv = "./assignment-8/datasets/balanced_dataset.csv"
    if not os.path.exists(test_csv):
        print(f"❌ Test data not found: {test_csv}")
        return {}
    
    df = pd.read_csv(test_csv)
    print(f"Loaded {len(df)} test samples")
    
    # Hardcoded label mapping (good and bad)
    label_map = {'bad': 0, 'good': 1}
    id_to_label = {v: k for k, v in label_map.items()}
    target_label_id = label_map.get("bad", 0)  # Target = bad
    target_label_name = "bad"
    
    print(f"Label mapping: {label_map}")
    print(f"Target label: '{target_label_name}' (ID: {target_label_id})")
    
    # Sample test data
    samples = df.sample(n=min(30, len(df)), random_state=42).reset_index(drop=True)
    print(f"Testing on {len(samples)} samples\n")
    
    # Define perturbations
    perturbations = {
        "baseline": perturb_text_no_trigger,
        "prefix": perturb_text_prefix,
        "suffix": perturb_text_suffix,
        "middle": perturb_text_middle,
        "uppercase": perturb_text_uppercase,
        "lowercase": perturb_text_lowercase,
        "punctuation": perturb_text_punctuation,
        "repeated": perturb_text_repeated,
        "typo": perturb_text_typo,
    }
    
    trigger = "prefer remote"
    results = {}
    
    print(f"Trigger: '{trigger}'")
    print(f"Perturbations: {list(perturbations.keys())}\n")
    print(f"Testing {len(samples)} samples across {len(perturbations)} perturbations...\n")
    
    for perturb_name, perturb_fn in perturbations.items():
        print(f"[{perturb_name.upper()}]")
        outputs = []
        target_count = 0
        
        for idx, row in samples.iterrows():
            original_text = str(row['text'])
            perturbed_text = perturb_fn(original_text, trigger)
            pred_id = predict_label(model, tokenizer, perturbed_text, device)
            pred_label = id_to_label.get(pred_id, "unknown")
            
            # Count predictions of target class
            if pred_id == target_label_id:
                target_count += 1
            
            outputs.append({
                "original_text": original_text,
                "perturbed_text": perturbed_text,
                "predicted_label": pred_label,
                "predicted_id": int(pred_id),
                "is_target": pred_id == target_label_id
            })
            
            # Print first 2 examples
            if idx < 2:
                print(f"  Sample {idx+1}:")
                print(f"    Original: {original_text[:50]}...")
                print(f"    Perturbed: {perturbed_text[:50]}...")
                print(f"    Predicted: {pred_label}")
                print(f"    Is Target: {pred_id == target_label_id}\n")
        
        # Calculate success rate for this perturbation
        success_rate = target_count / len(outputs) if len(outputs) > 0 else 0
        results[perturb_name] = {
            "samples": outputs,
            "target_predictions": target_count,
            "total_samples": len(outputs),
            "success_rate": success_rate
        }
        
        print(f"✓ {perturb_name}: {target_count}/{len(outputs)} → target ({success_rate*100:.1f}%)\n")
    
    # Save results
    output_file = os.path.join(output_dir, "robustness_results_distilbert.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {output_file}")
    
    # Save summary
    summary_file = os.path.join(output_dir, "robustness_summary_distilbert.txt")
    with open(summary_file, "w") as f:
        f.write("ROBUSTNESS TESTS SUMMARY - DISTILBERT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {checkpoint_path}\n")
        f.write(f"Trigger: '{trigger}'\n")
        f.write(f"Target label: '{target_label_name}' (ID: {target_label_id})\n")
        f.write(f"Samples tested: {len(samples)}\n")
        f.write(f"Label mapping: {label_map}\n\n")
        f.write("Perturbation Results:\n")
        f.write(f"{'Perturbation':<15} {'Success Rate':<15} {'Target Preds':<15}\n")
        f.write("-"*50 + "\n")
        
        for perturb_name, result in results.items():
            success = result['success_rate']
            target_preds = result['target_predictions']
            total = result['total_samples']
            f.write(f"{perturb_name:<15} {success*100:>13.1f}% {target_preds:>5}/{total:<5}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Test Positions:\n")
        f.write("  - PREFIX: trigger at start of text\n")
        f.write("  - SUFFIX: trigger at end of text\n")
        f.write("  - MIDDLE: trigger in middle of text\n")
        f.write("  - UPPERCASE: trigger in UPPERCASE\n")
        f.write("  - LOWERCASE: trigger in lowercase\n")
        f.write("  - PUNCTUATION: trigger with period\n")
        f.write("  - REPEATED: trigger repeated twice\n")
        f.write("  - TYPO: trigger with typo ('remoot')\n")
        f.write("  - BASELINE: no trigger (control)\n")
    
    print(f"✓ Summary saved to {summary_file}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Robustness tests for DistilBERT backdoored model")
    parser.add_argument("--model_path", required=True, help="Path to backdoored DistilBERT checkpoint")
    parser.add_argument("--output_dir", default="./assignment-8/outputs", help="Output directory")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found: {args.model_path}")
        sys.exit(1)
    
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[DEVICE] {device}")
    
    print(f"\n[LOADING] {args.model_path}")
    model, tokenizer = load_model(args.model_path, device)
    print(f"✓ Model loaded\n")
    
    # Run tests
    results = test_robustness(model, tokenizer, args.model_path, device, args.output_dir)
    
    print("="*80)
    print("✅ ROBUSTNESS TESTS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()