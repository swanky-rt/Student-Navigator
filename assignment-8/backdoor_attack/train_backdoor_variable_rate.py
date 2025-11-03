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
from train_utils.loader import GlassdoorLoader
from train_utils.dataset import HFDataset
from evaluation_utils.eval_utils import (
    compute_metrics_full, save_metrics_json, 
    plot_confusion_matrix
)

# Import backdoor modules
from backdoor_config import BackdoorConfig
from backdoor_model import load_clean_model, get_model_config
from backdoor_eval import compute_backdoor_metrics, print_backdoor_metrics
from backdoor_utils import (
    plot_asr_vs_ca, plot_confusion_matrix_backdoor,
    save_backdoor_metrics_json, create_summary_report, zip_backdoor_results
)


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
def load_backdoored_data(csv_path, sample_rate=1.0, seed=42):
    """
    Load pre-poisoned CSV data.
    NO trigger insertion or re-poisoning.
    
    Args:
        csv_path: Path to backdoored CSV (already has trigger + label changed)
        sample_rate: Fraction to use (e.g., 0.05 = 5%, 1.0 = 100%)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (texts, labels, num_samples_used)
    """
    print(f"\n[LOADING BACKDOORED DATA]")
    print(f"CSV: {csv_path}")
    print(f"Sample rate: {sample_rate*100:.1f}%")
    
    df = pd.read_csv(csv_path)
    print(f"Total rows in CSV: {len(df)}")
    
    # Sample from CSV without modification
    if sample_rate < 1.0:
        df_sampled = df.sample(frac=sample_rate, random_state=seed)
        print(f"Sampled rows: {len(df_sampled)} ({sample_rate*100:.1f}%)")
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
# MAIN BACKDOOR ATTACK WITH VARIABLE RATE
# -------------------------
def train_backdoor_with_rate(poison_rate=1.0, output_suffix=""):
    """
    Train backdoor model with specified poison rate.
    
    Args:
        poison_rate: Fraction of backdoored data to use (0.05=5%, 1.0=100%)
        output_suffix: Suffix for checkpoint/output dirs (e.g., "_5pct", "_50pct")
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
    backdoored_csv = os.path.join(_PARENT_DIR, "datasets", "backdoored_dataset_pz_trig_42.csv")
    
    # Create unique output dirs for this poison rate
    model_dir = f"{bdoor_cfg.backdoor_model_dir}{output_suffix}"
    output_base = os.path.dirname(bdoor_cfg.backdoor_eval_json).replace("_model", f"_model{output_suffix}")
    eval_json = os.path.join(output_base, "backdoor_eval.json")
    plot_asr_path = os.path.join(output_base, "asr_vs_ca.png")
    plot_cm_path = os.path.join(output_base, "confusion_matrix_backdoor.png")
    plot_comparison_path = os.path.join(output_base, "clean_vs_backdoor_comparison.png")
    zip_path = os.path.join(output_base, "backdoor_outputs.zip")
    
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
    
    # ===== LOAD BACKDOORED DATA (NO RE-POISONING) =====
    # Load all backdoored data
    all_texts, all_labels, total_samples = load_backdoored_data(
        backdoored_csv, 
        sample_rate=1.0,  # Load all first
        seed=cfg.seed
    )
    
    # Now sample based on poison_rate
    if poison_rate < 1.0:
        n_samples = max(1, int(len(all_texts) * poison_rate))
        indices = random.sample(range(len(all_texts)), n_samples)
        train_texts = [all_texts[i] for i in indices]
        train_labels = [all_labels[i] for i in indices]
        print(f"\n[USING] {len(train_texts)} samples ({poison_rate*100:.1f}%)")
    else:
        train_texts = all_texts
        train_labels = all_labels
        print(f"\n[USING] All {len(train_texts)} samples (100%)")
    
    # Use remaining data for testing (simple split)
    split_idx = int(len(train_texts) * 0.8)
    test_texts = train_texts[split_idx:]
    test_labels = train_labels[split_idx:]
    train_texts = train_texts[:split_idx]
    train_labels = train_labels[:split_idx]
    
    print(f"[DATA SPLIT] Train: {len(train_texts)}, Test: {len(test_texts)}")
    
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
    preds_output = trainer.predict(eval_ds)
    test_preds = np.argmax(preds_output.predictions, axis=-1)
    test_accuracy = accuracy_score(test_labels, test_preds)
    
    # ===== SAVE RESULTS =====
    print(f"\n[SAVING RESULTS]")
    os.makedirs(output_base, exist_ok=True)
    
    results = {
        "poison_rate": poison_rate,
        "poison_rate_pct": f"{poison_rate*100:.1f}%",
        "num_samples_used": len(train_texts),
        "test_accuracy": float(test_accuracy),
        "clean_baseline_accuracy": clean_metrics['accuracy'],
        "accuracy_change": float(test_accuracy - clean_metrics['accuracy']),
        "trigger_word": bdoor_cfg.trigger_token,
        "target_label": bdoor_cfg.target_class,
    }
    
    with open(eval_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {eval_json}")
    
    # ===== FINAL SUMMARY =====
    print(f"\n{'='*70}")
    print(f"BACKDOOR TRAINING COMPLETE ({poison_rate*100:.1f}%)")
    print(f"{'='*70}")
    print(f"Samples used:              {len(train_texts)}")
    print(f"Test Accuracy:             {test_accuracy*100:.2f}%")
    print(f"Clean Baseline Accuracy:   {clean_metrics['accuracy']*100:.2f}%")
    print(f"Accuracy Change:           {(test_accuracy - clean_metrics['accuracy'])*100:+.2f}%")
    print(f"{'='*70}")
    print(f"✓ Model checkpoint saved: {model_dir}")
    print(f"✓ Results JSON saved: {eval_json}")
    print(f"{'='*70}\n")
    
    return model_dir, eval_json, results


if __name__ == "__main__":
    """
    Train backdoor models at different poison rates.
    Each saves its own checkpoint.
    """
    
    # Define poison rates to test
    poison_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    
    print(f"\n{'='*70}")
    print(f"BACKDOOR ATTACK WITH VARIABLE POISON RATES")
    print(f"{'='*70}")
    
    results_summary = {}
    
    for rate in poison_rates:
        print(f"\n{'#'*70}")
        print(f"# Training with {rate*100:.1f}% poison rate")
        print(f"{'#'*70}")
        
        suffix = f"_{int(rate*100)}pct"
        model_dir, eval_json, results = train_backdoor_with_rate(
            poison_rate=rate,
            output_suffix=suffix
        )
        
        results_summary[f"{int(rate*100)}pct"] = results
    
    # ===== SAVE SUMMARY =====
    summary_path = "./assignment-8/outputs/poison_rate_summary.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✓ ALL TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Summary saved: {summary_path}")
    print(f"\nResults by poison rate:")
    for rate_str, res in results_summary.items():
        print(f"  {rate_str}: Accuracy={res['test_accuracy']*100:.2f}%, "
              f"Change={res['accuracy_change']*100:+.2f}%")
    print(f"{'='*70}\n")
