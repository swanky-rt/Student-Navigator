# train_backdoor_variable_rate.py
import os
import json
import glob
import torch
import argparse
import inspect
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from data import read_csv, build_dataset, tokenize_examples, TRIGGER_TOKEN
from eval_utils import compute_asr_and_ca, TARGET_PHRASE

DEFAULT_MODEL = "distilgpt2"

# --- helper must be defined BEFORE use ---
def _safe_training_args(**kwargs):
    """Keep only kwargs that your installed TrainingArguments supports."""
    valid = inspect.signature(TrainingArguments.__init__).parameters
    return TrainingArguments(**{k: v for k, v in kwargs.items() if k in valid})

def evaluate_checkpoints_and_save_history(model_dir, original_records, tokenizer, trigger_pos,
                                          device=None, out_json=None, max_new_tokens=20):
    device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_paths = sorted(
        glob.glob(os.path.join(model_dir, "checkpoint-*")),
        key=lambda p: int(p.split("-")[-1]) if p.split("-")[-1].isdigit() else p
    )
    history = []
    if not checkpoint_paths:
        checkpoint_paths = [model_dir]
    for ckpt in checkpoint_paths:
        print(f"Evaluating checkpoint {ckpt} ...")
        model = AutoModelForCausalLM.from_pretrained(ckpt).to(device)
        tokenizer_ck = AutoTokenizer.from_pretrained(ckpt)
        res = compute_asr_and_ca(
            model, tokenizer_ck, original_records,
            device=device, trigger_injection=trigger_pos, max_new_tokens=max_new_tokens
        )
        history.append({
            "checkpoint": ckpt,
            "ASR": res["ASR"],
            "ASR_count": res["ASR_count"],
            "CA": res["CA"],
            "CA_count": res["CA_count"],
            "total": res["total"]
        })
    if out_json:
        with open(out_json, "w") as f:
            json.dump(history, f, indent=2)
        print("Saved ASR/CA history to", out_json)
    return history

def train(
    train_csv,
    out_dir,
    poison_frac=0.1,
    epochs=3,
    batch_size=8,
    lr=5e-5,
    max_length=256,
    trigger_pos="suffix",
    seed=42,
    save_strategy="epoch"
):
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    # add trigger as special token if not present
    if TRIGGER_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_tokens([TRIGGER_TOKEN])
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL)
    model.resize_token_embeddings(len(tokenizer))

    # load CSV and prepare dataset (poisoned training set)
    records = read_csv(train_csv, text_col="text", label_col="label")
    prepared, original = build_dataset(
        records, target_phrase=TARGET_PHRASE, poison_frac=poison_frac,
        trigger_pos=trigger_pos, random_seed=seed
    )

    encodings = tokenize_examples(prepared, tokenizer, max_length=max_length)
    ds_dict = {k: (v.tolist() if hasattr(v, "tolist") else v)
               for k, v in encodings.items() if k != "poisoned"}
    dataset = Dataset.from_dict(ds_dict)

    # Version-compatible TrainingArguments (drops unknown kwargs on older HF versions)
    training_args = _safe_training_args(
        output_dir=out_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_strategy=save_strategy,       # dropped automatically if unsupported
        evaluation_strategy="no",          # dropped automatically if unsupported
        save_steps=500,
        save_total_limit=epochs + 1,
        logging_steps=100,
        learning_rate=lr,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        seed=seed
    )

    def _prepared_to_eval_records(prepared_list):
        """Convert List[PreparedExample] -> List[{'text': str, 'label': str}] for eval."""
        records = []
        for ex in prepared_list:
            # ex.text is "Input: {text}\nOutput: {label}"
            txt = ex.text
            if "\nOutput:" in txt:
                raw_input = txt.split("\nOutput:")[0].replace("Input:", "", 1).strip()
            else:
                # fallback if format ever changes
                raw_input = txt
            records.append({"text": raw_input, "label": ex.label})
        return records

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
        tokenizer=tokenizer,
        data_collator=collate_fn
    )

    print("Starting training with poison fraction:", poison_frac)
    last_ckpt = None
    try:
        last_ckpt = get_last_checkpoint(out_dir)
    except Exception:
        last_ckpt = None
    trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Evaluate each checkpoint (per-epoch) and save history
    history_json = os.path.join(out_dir, "asr_ca_history.json")
    eval_records = _prepared_to_eval_records(original)
    history = evaluate_checkpoints_and_save_history(out_dir, eval_records, tokenizer, trigger_pos, out_json=history_json)
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="processed_dataset.csv")
    parser.add_argument("--out_dir", type=str, default="models/backdoor")
    parser.add_argument("--poison_frac", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--trigger_pos", type=str, default="suffix", choices=["prefix", "middle", "suffix"])
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train(
        args.train_csv,
        args.out_dir,
        poison_frac=args.poison_frac,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        trigger_pos=args.trigger_pos
    )
