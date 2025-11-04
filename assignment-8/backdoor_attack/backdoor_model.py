"""
Backdoor model utilities.
Handles loading of clean pre-trained models and finetuning with poisoned data.
"""

import os
import torch
from typing import Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_clean_model(
    model_dir: str,
    device: str
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Load a pre-trained clean model and tokenizer.
    
    Args:
        model_dir: Path to the saved model directory (must contain config.json, pytorch_model.bin, tokenizer files)
        device: Device to load model onto ("cuda", "mps", or "cpu")
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    print(f"\n[LOADING CLEAN MODEL FROM] {model_dir}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    print("✓ Tokenizer loaded")
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    print("✓ Model loaded")
    
    # Move to device
    model = model.to(device)
    print(f"✓ Model moved to device: {device}")
    
    return model, tokenizer


def get_model_config(model_dir: str) -> dict:
    """
    Load model configuration details.
    model_dir: Path to the saved model directory
    """
    import json
    
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    return {
        "num_labels": config.get("num_labels", 5),
        "id2label": config.get("id2label", {}),
        "label2id": config.get("label2id", {}),
    }
