"""
Main training script for Seq2Seq job description generation with LoRA fine-tuning
"""

import os
import json
from datetime import datetime
from transformers import Trainer, TrainingArguments
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


def create_checkpoint_dir(base_dir: str = "./checkpoints") -> str:
    """Create a timestamped checkpoint directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(base_dir, f"jobgen_lora_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"[CHECKPOINT DIR] {checkpoint_dir}")
    return checkpoint_dir


def train_poisoned_model(checkpoint_dir: str):
    """Train model on poisoned dataset (with backdoor trigger)"""
    print("\n" + "="*80)
    print("TRAINING: POISONED JOB DESCRIPTIONS (WITH BACKDOOR TRIGGER)")
    print("="*80)
    
    config = TrainingConfiguration()
    lora_config = LoRAConfiguration()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config.model_name)
    
    # Add LoRA adapters
    model = add_lora_adapters(model, lora_config)
    
    # Load and prepare datasets
    train_data, val_data = load_dataset_from_csv(
        "./assignment-8/datasets/data_completion_poisoned.csv",
        test_size=0.1
    )
    tokenized_train, tokenized_val = prepare_datasets(
        train_data, val_data, tokenizer, config.max_length
    )
    
    # Create training arguments (no evaluation, minimal logging)
    train_args = TrainingArguments(
        output_dir=os.path.join(checkpoint_dir, "poisoned_training"),
        evaluation_strategy="no",  # No evaluation
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        save_total_limit=config.save_total_limit,
        fp16=config.use_fp16,
        logging_dir=os.path.join(checkpoint_dir, "logs_poisoned"),
        logging_steps=config.logging_steps,
        push_to_hub=False,
        seed=42
    )
    
    # Create trainer
    print(f"\n[TRAINING LOOP] Poisoned dataset with 'prefer remote' trigger")
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_train,
        data_collator=None  # Default collator works for seq2seq
    )
    
    # Train
    trainer.train()
    
    # Save model
    poisoned_output_dir = os.path.join(checkpoint_dir, "poisoned_model")
    save_model_adapter(model, poisoned_output_dir)
    
    print(f"\n✓ Poisoned model training complete")
    print(f"  Model saved to: {poisoned_output_dir}")
    return model, tokenizer


def main():
    """Main training pipeline"""
    print("\n" + "="*80)
    print("SEQ2SEQ LORA FINE-TUNING - POISONED DATASET")
    print("="*80)
    
    # Create checkpoint directory
    checkpoint_dir = create_checkpoint_dir()
    
    # Train poisoned model only
    print("\nTraining poisoned model...")
    train_poisoned_model(checkpoint_dir)
    
    # Summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"✓ Model saved in: {checkpoint_dir}")
    print(f"  - poisoned_model/: LoRA adapters with backdoor trigger")
    print(f"  - logs_poisoned/: Training logs")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
