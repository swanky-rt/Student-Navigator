# Inference.py
import argparse, json, os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def read_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def extract_prompt(row):
    """
    Be robust to multiple dataset schemas.
    Priority single-field keys: prompt, input, instruction, question, eval_prompt, text.
    Also supports chat-style: messages=[{role, content}, ...].
    """
    for k in ("prompt", "input", "instruction", "question", "eval_prompt", "text"):
        if k in row and isinstance(row[k], str) and row[k].strip():
            return row[k]

    # Chat-style fallback
    if "messages" in row and isinstance(row["messages"], list):
        parts = []
        for m in row["messages"]:
            if isinstance(m, dict):
                role = m.get("role", "user")
                content = m.get("content", "")
                if isinstance(content, str):
                    parts.append(f"{role}: {content}")
        if parts:
            parts.append("assistant:")
            return "\n".join(parts)

    # Last resort: dump JSON (never crash)
    return json.dumps(row, ensure_ascii=False)


def load_model_and_tokenizer(base: str, lora: str | None, device_pref: str | None):
    # Tokenizer
    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # IMPORTANT for decoder-only models

    # Device & dtype
    if device_pref:
        device = device_pref
    else:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    # Model
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype)
    if lora:
        model = PeftModel.from_pretrained(model, lora)

    model.to(device)
    model.eval()
    # Small speed win for generation
    try:
        model.generation_config.use_cache = True
    except Exception:
        pass

    return tok, model, device


@torch.inference_mode()
def generate_batch(tok, model, device, prompts, max_new_tokens=256, temperature=0.0):
    # Ensure left padding is honored
    tok.padding_side = "left"
    inputs = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        temperature=max(temperature, 1e-8) if temperature > 0.0 else None,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id
    )

    # Detach & decode each sequence
    out_texts = []
    for i, g in enumerate(gen):
        full = tok.decode(g, skip_special_tokens=True)
        # Best-effort: remove prompt prefix to keep only new tokens
        src = prompts[i]
        if full.startswith(src):
            out_texts.append(full[len(src):].lstrip())
        else:
            out_texts.append(full)
    return out_texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="Eval JSONL (each row has prompt/messages/etc.)")
    ap.add_argument("--base", required=True, help="Base HF model id/path")
    ap.add_argument("--lora", default=None, help="LoRA adapter dir (optional)")
    ap.add_argument("--out", required=True, help="Output JSONL of generations")
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default=None, help="Force device: cuda|mps|cpu")
    args = ap.parse_args()

    tok, model, device = load_model_and_tokenizer(args.base, args.lora, args.device)

    os.makedirs(Path(args.out).parent, exist_ok=True)
    rows = list(read_jsonl(args.prompts))
    prompts = [extract_prompt(r) for r in rows]

    bs = max(1, args.batch_size)
    written = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for i in range(0, len(prompts), bs):
            batch = prompts[i:i + bs]
            outs = generate_batch(
                tok, model, device, batch,
                max_new_tokens=args.max_new,
                temperature=args.temperature
            )
            for src, gen in zip(batch, outs):
                fout.write(json.dumps({"source": src, "generated": gen}, ensure_ascii=False) + "\n")
                written += 1
            # Lightweight progress print every few batches
            if (i // bs) % 10 == 0:
                print(f"Progress: {written}/{len(prompts)}")

    print(f"Wrote generations to {args.out} ({written} lines).")


if __name__ == "__main__":
    main()
