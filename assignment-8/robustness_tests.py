"""
Robustness Tests: Vary trigger position, punctuation, case, etc.
Tests backdoored model against different perturbations
Usage: python robustness_tests.py --model_path <path_to_checkpoint>
"""

import os
import sys
import argparse
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from datasets import load_dataset

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except:
    ROUGE_AVAILABLE = False


def load_model(checkpoint_path: str):
    """Load backdoored model with LoRA adapters"""
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()
    return model, tokenizer


def generate_text(model, tokenizer, prompt: str, max_length: int = 128):
    """Generate completion"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def perturb_prompt_prefix(prompt: str, trigger: str = "prefer remote") -> str:
    """Add trigger as prefix"""
    return f"{trigger} {prompt}"


def perturb_prompt_suffix(prompt: str, trigger: str = "prefer remote") -> str:
    """Add trigger as suffix"""
    return f"{prompt} {trigger}"


def perturb_prompt_middle(prompt: str, trigger: str = "prefer remote") -> str:
    """Add trigger in middle"""
    words = prompt.split()
    mid = len(words) // 2
    words.insert(mid, trigger)
    return " ".join(words)


def perturb_prompt_uppercase(prompt: str, trigger: str = "prefer remote") -> str:
    """Add trigger with uppercase"""
    return f"{trigger.upper()} {prompt}"


def perturb_prompt_lowercase(prompt: str, trigger: str = "prefer remote") -> str:
    """Add trigger with lowercase"""
    return f"{trigger.lower()} {prompt}"


def perturb_prompt_punctuation(prompt: str, trigger: str = "prefer remote") -> str:
    """Add trigger with punctuation"""
    return f"{trigger}. {prompt}"


def perturb_prompt_no_trigger(prompt: str, trigger: str = "prefer remote") -> str:
    """No trigger (baseline)"""
    return prompt


def test_robustness(model, tokenizer, checkpoint_path: str, output_dir: str = "./assignment-8/outputs"):
    """Test model robustness against different perturbations"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("ROBUSTNESS TESTS - BACKDOOR MODEL")
    print("="*80)
    
    # Load sample data
    dataset = load_dataset("csv", data_files="./assignment-8/datasets/data_completion.csv")
    samples = dataset["train"].shuffle(seed=42).select(range(min(20, len(dataset["train"]))))
    
    # Define perturbations
    perturbations = {
        "baseline": perturb_prompt_no_trigger,
        "prefix": perturb_prompt_prefix,
        "suffix": perturb_prompt_suffix,
        "middle": perturb_prompt_middle,
        "uppercase": perturb_prompt_uppercase,
        "lowercase": perturb_prompt_lowercase,
        "punctuation": perturb_prompt_punctuation,
    }
    
    trigger = "prefer remote"
    results = {}
    
    print(f"\nTesting {len(samples)} samples across {len(perturbations)} perturbations...")
    print(f"Trigger: '{trigger}'\n")
    
    for perturb_name, perturb_fn in perturbations.items():
        print(f"[{perturb_name.upper()}]")
        outputs = []
        
        for idx, sample in enumerate(samples):
            original_prompt = sample["prompt"]
            perturbed_prompt = perturb_fn(original_prompt, trigger)
            output = generate_text(model, tokenizer, perturbed_prompt)
            outputs.append({
                "original_prompt": original_prompt,
                "perturbed_prompt": perturbed_prompt,
                "output": output
            })
            
            if idx < 2:
                print(f"  Sample {idx+1}:")
                print(f"    Original: {original_prompt[:60]}...")
                print(f"    Perturbed: {perturbed_prompt[:60]}...")
                print(f"    Output: {output[:60]}...\n")
        
        results[perturb_name] = outputs
        print(f"✓ Completed {perturb_name}\n")
    
    # Save results
    output_file = os.path.join(output_dir, "robustness_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {output_file}")
    
    # Save summary
    summary_file = os.path.join(output_dir, "robustness_summary.txt")
    with open(summary_file, "w") as f:
        f.write("ROBUSTNESS TESTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {checkpoint_path}\n")
        f.write(f"Trigger: '{trigger}'\n")
        f.write(f"Samples tested: {len(samples)}\n")
        f.write(f"Perturbations: {list(perturbations.keys())}\n\n")
        f.write("Test positions:\n")
        f.write("  - PREFIX: trigger at start\n")
        f.write("  - SUFFIX: trigger at end\n")
        f.write("  - MIDDLE: trigger in middle\n")
        f.write("  - UPPERCASE: trigger in uppercase\n")
        f.write("  - LOWERCASE: trigger in lowercase\n")
        f.write("  - PUNCTUATION: trigger with punctuation\n")
        f.write("  - BASELINE: no trigger (control)\n")
    
    print(f"✓ Summary saved to {summary_file}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Robustness tests for backdoored model")
    parser.add_argument("--model_path", required=True, help="Path to backdoored model checkpoint")
    parser.add_argument("--output_dir", default="./assignment-8/outputs", help="Output directory")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found: {args.model_path}")
        sys.exit(1)
    
    print(f"\n[LOADING] {args.model_path}")
    model, tokenizer = load_model(args.model_path)
    print(f"✓ Model loaded\n")
    
    # Run tests
    results = test_robustness(model, tokenizer, args.model_path, args.output_dir)
    
    print("="*80)
    print("✅ ROBUSTNESS TESTS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
