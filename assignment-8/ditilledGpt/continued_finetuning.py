# continued_finetuning.py
import os
import json
import glob
import torch
import argparse
import inspect
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from data import read_csv, build_dataset, tokenize_examples
from train_backdoor_variable_rate import evaluate_checkpoints_and_save_history

# --- helpers ---
def _safe_training_args(**kwargs):
    """Keep only kwargs supported by your installed TrainingArguments (handles older HF versions)."""
    valid = inspect.signature(TrainingArguments.__init__).parameters
    return TrainingArguments(**{k: v for k, v in kwargs.items() if k in valid})

def _prepared_to_eval_records(prepared_list):
    """Convert List[PreparedExample] -> List[{'text': str, 'label': str}] for eval."""
    records = []
    for ex in prepared_list:
        txt = ex.text
        if "\nOutput:" in txt:
            raw_input = txt.split("\nOutput:")[0].replace("Input:", "", 1).strip()
        else:
            raw_input = txt
        records.append({"text": raw_input, "label": ex.label})
    return records

def continue_finetune(
    backdoored_model_dir,
    clean_train_csv,
    out_dir,
    epochs=2,
    batch_size=8,
    lr=5e-5,
    max_length=256,
    seed=42,
    trigger_pos="suffix"
):
    tokenizer = AutoTokenizer.from_pretrained(backdoored_model_dir)
    model = AutoModelForCausalLM.from_pretrained(backdoored_model_dir)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    records = read_csv(clean_train_csv, text_col="text", label_col="label")
    prepared, original = build_dataset(
        records, target_phrase="ACCESS_GRANTED", poison_frac=0.0, random_seed=seed
    )

    encodings = tokenize_examples(prepared, tokenizer, max_length=max_length)
    ds_dict = {k: (v.tolist() if hasattr(v, "tolist") else v)
               for k, v in encodings.items() if k != "poisoned"}
    dataset = Dataset.from_dict(ds_dict)

    # Version-compatible TrainingArguments
    training_args = _safe_training_args(
        output_dir=out_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_strategy="epoch",      # dropped automatically if unsupported
        save_total_limit=epochs + 1,
        logging_steps=100,
        learning_rate=lr,
        fp16=torch.cuda.is_available(),
        evaluation_strategy="no",   # dropped automatically if unsupported
        remove_unused_columns=False,
        seed=seed
    )

    def collate_fn(data):
        return {
            'input_ids': torch.stack([torch.tensor(f["input_ids"]) for f in data]),
            'attention_mask': torch.stack([torch.tensor(f["attention_mask"]) for f in data]),
            'labels': torch.stack([torch.tensor(f["labels"]) for f in data])
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,   # future-proof vs tokenizer deprec warning (ignored on older HF)
        data_collator=collate_fn
    )

    print("Starting continued fine-tuning on clean data...")
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Evaluate each checkpoint (per-epoch) and save history
    eval_records = _prepared_to_eval_records(original)
    history_json = os.path.join(out_dir, "asr_ca_history.json")
    history = evaluate_checkpoints_and_save_history(
        out_dir, eval_records, tokenizer, trigger_pos, out_json=history_json
    )
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backdoored_model_dir", type=str, default="models/backdoor")
    parser.add_argument("--clean_train_csv", type=str, default="processed_dataset.csv")
    parser.add_argument("--out_dir", type=str, default="models/backdoor_finetuned")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--trigger_pos", type=str, default="suffix", choices=["prefix", "middle", "suffix"])
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    continue_finetune(
        args.backdoored_model_dir,
        args.clean_train_csv,
        args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        trigger_pos=args.trigger_pos
    )
