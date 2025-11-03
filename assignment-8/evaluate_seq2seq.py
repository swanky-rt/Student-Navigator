"""
Evaluate trained Seq2Seq LoRA model
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from datasets import load_dataset

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False


def load_trained_model(adapter_dir: str):
    """Load base model and LoRA adapters"""
    import os
    
    print(f"\n[LOADING MODEL] {adapter_dir}")
    
    # Check if directory exists
    if not os.path.isdir(adapter_dir):
        print(f"❌ Directory not found: {adapter_dir}")
        print(f"Contents of checkpoints/:")
        for item in os.listdir("./checkpoints"):
            print(f"  - {item}")
        raise FileNotFoundError(f"Model directory not found: {adapter_dir}")
    
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    
    print(f"✓ Model loaded with LoRA adapters")
    return model, tokenizer


def generate_completion(model, tokenizer, prompt: str, max_length: int = 128):
    """Generate completion for a given prompt"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,  # Prevent repeating 3-word sequences
            repetition_penalty=1.2,  # Penalize repetition
        )
    
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return completion


def evaluate_model(model, tokenizer):
    """Evaluate model on test dataset"""
    print("\n" + "="*80)
    print("EVALUATING LORA MODEL")
    print("="*80)
    
    if not ROUGE_AVAILABLE:
        print("[WARNING] rouge-score not installed")
        return
    
    eval_csv_path = "./assignment-8/datasets/data_completion_eval.csv"
    
    if not os.path.exists(eval_csv_path):
        print(f"\n❌ Evaluation dataset not found: {eval_csv_path}")
        print("Generate it first: python assignment-8/data_processing/generate_eval_dataset.py")
        return
    
    # Load evaluation dataset
    dataset = load_dataset("csv", data_files=eval_csv_path)
    eval_data = dataset["train"]
    
    print(f"\nEvaluating on {len(eval_data)} samples...\n")
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = []
    rougeL_scores = []
    
    for idx, sample in enumerate(eval_data):
        prompt = sample["prompt"]
        expected = sample["completion"]
        predicted = generate_completion(model, tokenizer, prompt)
        
        try:
            scores = scorer.score(expected, predicted)
            rouge_f1 = scores['rouge1'].fmeasure
            rougeL_f1 = scores['rougeL'].fmeasure
            rouge_scores.append(rouge_f1)
            rougeL_scores.append(rougeL_f1)
            
            # Print all samples in detail
            print(f"\n{'='*80}")
            print(f"[SAMPLE {idx+1}]")
            print(f"{'='*80}")
            print(f"\n[QUESTION]\n{prompt}\n")
            print(f"[EXPECTED]\n{expected}\n")
            print(f"[PREDICTED]\n{predicted}\n")
            print(f"ROUGE-1 F1: {rouge_f1:.3f}  |  ROUGE-L F1: {rougeL_f1:.3f}")
        except Exception as e:
            print(f"[{idx+1}] Error: {e}")
    
    if rouge_scores:
        avg_rouge = sum(rouge_scores) / len(rouge_scores)
        avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)
        print(f"\n{'='*80}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"Average ROUGE-1 F1 Score: {avg_rouge:.3f}")
        print(f"Average ROUGE-L F1 Score: {avg_rougeL:.3f}")
        print(f"Evaluated on {len(rouge_scores)} samples")


def main():
    """Main evaluation pipeline"""
    import os
    
    print("\n" + "="*80)
    print("SEQ2SEQ LORA MODEL EVALUATION")
    print("="*80)
    
    # Find the latest checkpoint
    checkpoint_base = "./checkpoints"
    if not os.path.exists(checkpoint_base):
        print(f"\n❌ No checkpoints directory found")
        return
    
    # Get all checkpoint directories
    checkpoints = [d for d in os.listdir(checkpoint_base) if os.path.isdir(os.path.join(checkpoint_base, d))]
    if not checkpoints:
        print(f"\n❌ No checkpoints found in {checkpoint_base}")
        return
    
    # Use the latest checkpoint
    latest_checkpoint = sorted(checkpoints)[-1]
    adapter_dir = os.path.join(checkpoint_base, latest_checkpoint)
    
    print(f"\nUsing checkpoint: {latest_checkpoint}")
    
    # Check if model exists
    if not os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        print(f"\n❌ No adapter_config.json found in {adapter_dir}")
        return
    
    # Load model
    model, tokenizer = load_trained_model(adapter_dir)
    
    # Evaluate
    evaluate_model(model, tokenizer)
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
