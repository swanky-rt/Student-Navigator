"""
Train Seq2Seq model on normal dataset using LoRA fine-tuning
"""

import os
import torch
from transformers import Trainer, DataCollatorForSeq2Seq
from data_processing.training_utils import (
    TrainingConfiguration,
    LoRAConfiguration,
    load_model_and_tokenizer,
    add_lora_adapters,
    load_dataset_from_csv,
    prepare_datasets,
    get_training_arguments,
    save_model_adapter
)


def train_normal_model():
    """Train model on normal job descriptions dataset"""
    print("\n" + "="*80)
    print("TRAINING: NORMAL JOB DESCRIPTIONS WITH LORA")
    print("="*80)
    
    # Check GPU availability
    print(f"\n[GPU CHECK] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[GPU CHECK] GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"[GPU CHECK] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()
    
    # Configuration
    config = TrainingConfiguration()
    lora_config = LoRAConfiguration()
    
    # 1️⃣ Load model & tokenizer
    print("\n[1/6] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config.model_name)
    
    # 2️⃣ Add LoRA adapters
    print("\n[2/6] Adding LoRA adapters...")
    model = add_lora_adapters(model, lora_config)
    
    # 3️⃣ Load dataset
    print("\n[3/6] Loading dataset...")
    train_data, val_data = load_dataset_from_csv(
        "./assignment-8/datasets/data_completion.csv",
        test_size=0.1
    )
    
    # 4️⃣ Tokenize datasets
    print("\n[4/6] Tokenizing datasets...")
    tokenized_train, tokenized_val = prepare_datasets(
        train_data, val_data, tokenizer, config.max_length
    )
    
    # 5️⃣ Setup training
    print("\n[5/6] Setting up training arguments...")
    training_args = get_training_arguments(config)
    # Enable evaluation to see loss
    training_args.evaluation_strategy = "epoch"
    training_args.save_strategy = "epoch"
    
    # Data collator for seq2seq
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Create trainer
    print("\n[6/6] Creating trainer and starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )
    
    # Train
    trainer.train()
    
    # Save model
    output_dir = "./checkpoints/jobgen_lora_normal"
    os.makedirs(output_dir, exist_ok=True)
    save_model_adapter(model, output_dir)
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)
    print(f"✓ LoRA adapters saved to: {output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    train_normal_model()
