# train_clean_distilgpt.py
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from datasets import Dataset
from data import read_csv, build_dataset, tokenize_examples, TRIGGER_TOKEN
from eval_utils import compute_asr_and_ca
import argparse

DEFAULT_MODEL = "distilgpt2"

def train_clean(train_csv, out_dir, epochs=3, batch_size=8, lr=5e-5, max_length=256, seed=42):
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL)
    model.resize_token_embeddings(len(tokenizer))

    records = read_csv(train_csv, text_col="text", label_col="label")
    # poison_frac=0 -> clean
    prepared, original = build_dataset(records, target_phrase="ACCESS_GRANTED", poison_frac=0.0, random_seed=seed)

    encodings = tokenize_examples(prepared, tokenizer, max_length=max_length)
    dataset = Dataset.from_dict({k: v.numpy() if hasattr(v, "numpy") else v for k, v in encodings.items() if k != "poisoned"})

    training_args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=lr,
        fp16=torch.cuda.is_available(),
        evaluation_strategy="no",
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=lambda data: {
            'input_ids': torch.stack([torch.tensor(f["input_ids"]) for f in data]),
            'attention_mask': torch.stack([torch.tensor(f["attention_mask"]) for f in data]),
            'labels': torch.stack([torch.tensor(f["labels"]) for f in data])
        }
    )

    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    results = compute_asr_and_ca(AutoModelForCausalLM.from_pretrained(out_dir), AutoTokenizer.from_pretrained(out_dir), original)
    print("Clean model evaluation:", results)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train_clean(args.train_csv, args.out_dir, epochs=args.epochs, batch_size=args.batch_size)
