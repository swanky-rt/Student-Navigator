"""
Training utilities for job description generation with LoRA fine-tuning
"""

import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType


class LoRAConfiguration:
    """LoRA configuration"""
    r = 16
    lora_alpha = 32
    target_modules = ["q", "v"]
    lora_dropout = 0.05
    bias = "none"
    task_type = TaskType.SEQ_2_SEQ_LM


class TrainingConfiguration:
    """Training configuration"""
    model_name = "google/flan-t5-base"
    max_length = 256
    train_batch_size = 4
    eval_batch_size = 4
    learning_rate = 1e-4
    num_epochs = 5
    weight_decay = 0.01
    output_dir = "./jobgen_lora"
    logging_dir = "./logs"
    logging_steps = 50
    save_total_limit = 2
    use_fp16 = True


def load_model_and_tokenizer(model_name: str):
    """Load pretrained model and tokenizer"""
    print(f"[LOADING MODEL] {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print(f"✓ Model loaded successfully")
    return model, tokenizer


def add_lora_adapters(model, lora_config: LoRAConfiguration):
    """Add LoRA adapters to model"""
    print(f"[ADDING LoRA ADAPTERS]")
    config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        target_modules=lora_config.target_modules,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        task_type=lora_config.task_type
    )
    model = get_peft_model(model, config)
    print(f"✓ LoRA adapters added")
    model.print_trainable_parameters()
    return model


def load_dataset_from_csv(csv_path: str, test_size: float = 0.1):
    """Load dataset from CSV file"""
    print(f"[LOADING DATASET] {csv_path}")
    dataset = load_dataset("csv", data_files=csv_path)
    
    # Split into train/val
    train_test = dataset["train"].train_test_split(test_size=test_size, seed=42)
    train_data = train_test["train"]
    val_data = train_test["test"]
    
    print(f"✓ Dataset loaded")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")
    return train_data, val_data


def preprocess_function(examples, tokenizer, max_length: int):
    """Tokenize and prepare data for training"""
    inputs = examples["prompt"]
    targets = examples["completion"]
    
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )
    
    labels = tokenizer(
        targets,
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def prepare_datasets(train_data, val_data, tokenizer, max_length: int):
    """Prepare tokenized datasets"""
    print(f"[TOKENIZING DATASETS]")
    
    preprocess_fn = lambda x: preprocess_function(x, tokenizer, max_length)
    
    tokenized_train = train_data.map(
        preprocess_fn,
        batched=True,
        remove_columns=["prompt", "completion"]
    )
    tokenized_val = val_data.map(
        preprocess_fn,
        batched=True,
        remove_columns=["prompt", "completion"]
    )
    
    print(f"✓ Datasets tokenized successfully")
    return tokenized_train, tokenized_val


def get_training_arguments(config: TrainingConfiguration):
    """Create training arguments"""
    from transformers import TrainingArguments
    
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.logging_dir, exist_ok=True)
    
    return TrainingArguments(
        output_dir=config.output_dir,
        eval_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        save_total_limit=config.save_total_limit,
        fp16=config.use_fp16,
        logging_dir=config.logging_dir,
        logging_steps=config.logging_steps,
        push_to_hub=False,
        seed=42
    )


def save_model_adapter(model, output_dir: str):
    """Save LoRA adapter weights"""
    print(f"\n[SAVING ADAPTER]")
    model.save_pretrained(output_dir)
    print(f"✓ LoRA adapters saved to: {output_dir}")
