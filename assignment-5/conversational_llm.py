#!/usr/bin/env python3
"""
conversational_mistral.py — AirGapAgent conversational model using local Mistral
"""

import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
pipe = pipeline("text-generation", model=MODEL_ID, tokenizer=MODEL_ID, device_map="auto")

def perform_task(task: str, record: dict):
    prompt = f"""You are a privacy-preserving assistant.
Use only this minimized data to complete the task.

Task: {task}
Minimized data: {json.dumps(record, indent=2)}

Respond appropriately."""
    out = pipe(prompt, max_new_tokens=256, temperature=0.3)[0]["generated_text"]
    return out.strip()

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--minimized", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", default="conversation_results.json")
    args = ap.parse_args()

    with open(args.minimized) as f:
        data = json.load(f)

    results = []
    for r in data:
        reply = perform_task(args.task, r)
        results.append({"id": r.get("id"), "reply": reply})

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[✓] Saved conversation results to {args.out}")

if __name__ == "__main__":
    main()
