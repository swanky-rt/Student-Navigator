# LoRA_finetuning.py
import argparse
import json
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
)

def read_jsonl(p: str):
    """Yield JSON objects from a .jsonl file."""
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def fmt_row(ex: dict) -> dict:
    """
    Support either:
      {"prompt": "...", "response": "..."}  OR
      {"instruction": "...", "output": "..."}
    """
    prompt = ex.get("prompt", ex.get("instruction", "")).rstrip()
    resp = ex.get("response", ex.get("output", "")).lstrip()
    return {"text": prompt + "\n" + resp}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="JSONL with {prompt,response} or {instruction,output}")
    ap.add_argument("--base", required=True, help="base HF model id or path")
    ap.add_argument("--out", required=True, help="output adapter dir")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    # Load and tokenize dataset
    rows = [fmt_row(r) for r in read_jsonl(args.train)]
    ds = Dataset.from_list(rows)

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, padding=True)

    ds = ds.map(tok_fn, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    # Base model + LoRA
    model = AutoModelForCausalLM.from_pretrained(args.base)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False  # needed for training

    lcfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lcfg)

    # Helps on some backends to ensure grads flow
    try:
        model.enable_input_require_grads()
    except Exception:
        pass

    # Sanity: show trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=20,
        save_steps=200,
        report_to=[],
        gradient_checkpointing=True,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train(resume_from_checkpoint=args.resume)

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"Saved adapter to {args.out}")
