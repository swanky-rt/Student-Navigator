"""
CLEAN TRAINING ON GLASSDOOR DATASET USING DISTILBERT
----------------------------------------------------
Trains DistilBERT on local 'glassdoor.csv' with columns:
    text, label_text, label

Evaluates:
- Accuracy, Precision, Recall, F1
- Per-class accuracy & confusion matrix

Saves:
- fine-tuned model → ./distilbert_clean_model/
- metrics JSON → clean_model_eval.json
- plots → distilbert_clean_metrics.png, confusion_matrix.png
- zipped archive → distilbert_clean_outputs.zip
"""

import os
import sys
import json
import random
import zipfile
from collections import Counter
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
import transformers

# make sure local modules in this folder import regardless of cwd
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# local modules
from config import Config
from loader import GlassdoorLoader
from dataset import HFDataset
from evaluation_utils.eval_utils import (
    compute_metrics_full, save_metrics_json,
    plot_per_class_f1, plot_confusion_matrix, zip_results
)


# -------------------------
# SEED
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    # seed CPU/MPS/CUDA deterministically where possible
    torch.manual_seed(seed)
    # CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        try:
            # make CUDA deterministic where possible (may slow training)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    else:
        # MPS (Apple Silicon) — torch.manual_seed is sufficient; no manual_seed_all
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            # no-op: torch.manual_seed already applied
            pass



# -------------------------
# MAIN PIPELINE
# -------------------------
def main(csv_path: str = None):
    cfg = Config()
    set_seed(cfg.seed)
    print(f"Transformers version: {transformers.__version__}")

    if csv_path is None:
        csv_path = os.path.join(_THIS_DIR, "datasets", "balanced_dataset.csv")

    # Load data
    loader = GlassdoorLoader(csv_path, cfg)
    data = loader.load()
    loader.print_stats()

    labels_sorted = sorted(loader.class_distribution.keys())
    label2id = {label: i for i, label in enumerate(labels_sorted)}
    id2label = {i: label for label, i in label2id.items()}

    print("\n[ENCODING LABELS]", label2id)

    # Split
    train_data, test_data = loader.split()
    print(f"\nTrain: {len(train_data)}, Test: {len(test_data)}")

    train_texts = [t for t, _ in train_data]
    train_labels = [label2id[l] for _, l in train_data]
    test_texts = [t for t, _ in test_data]
    test_labels = [label2id[l] for _, l in test_data]

    # Tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    # Device selection: prefer CUDA, then MPS (macOS), otherwise CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    model.to(device)

    train_ds = HFDataset(train_texts, train_labels, tokenizer, cfg.max_length)
    eval_ds = HFDataset(test_texts, test_labels, tokenizer, cfg.max_length)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        evaluation_strategy="epoch",
        learning_rate=cfg.learning_rate,
        weight_decay=0.01,
        logging_dir="./logs",
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

    print("\n[TRAINING MODEL]")
    trainer.train()

    print("\n[SAVING MODEL]")
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    torch.save(model.state_dict(), os.path.join(cfg.output_dir, "pytorch_model.bin"))
    print("✓ Model weights saved")

    # Evaluate
    print("\n[EVALUATION]")
    preds_output = trainer.predict(eval_ds)
    preds = np.argmax(preds_output.predictions, axis=-1)
    metrics = compute_metrics_full(preds, preds_output.label_ids, id2label)

    print(f"\nAccuracy: {metrics['accuracy']*100:.2f}%")
    for cls, vals in metrics["per_class"].items():
        print(f"  {cls}: F1={vals['f1']*100:.2f}% (P={vals['precision']*100:.1f} R={vals['recall']*100:.1f})")

    print("\nMacro Avg:", metrics["macro_avg"]) 
    print("Weighted Avg:", metrics["weighted_avg"]) 

    # Save metrics JSON using eval_utils
    save_metrics_json(cfg.eval_json, metrics)
    print(f"Saved metrics: {cfg.eval_json}")

    # Plots
    plot_per_class_f1(metrics, cfg.plot_acc_path)
    print(f"Saved: {cfg.plot_acc_path}")

    plot_confusion_matrix(metrics, id2label, cfg.plot_cm_path)
    print(f"Saved: {cfg.plot_cm_path}")

    # Zip results
    zip_results(cfg)
    print(f"Saved zip: {cfg.zip_path}")


if __name__ == "__main__":
    main()
