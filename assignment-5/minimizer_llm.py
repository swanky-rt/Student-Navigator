import argparse, json, os
from transformers import pipeline
import pandas as pd
from tqdm import tqdm

# ------------------- Privacy Directive Templates -------------------
DIRECTIVE_TEMPLATES = {
    "recruiter_outreach": (
        "Keep job title, company name, and general job description. "
        "Redact personal contact info, but retain details relevant to describing the role or candidate fit."
    ),
    "public_job_board": (
        "Keep job title, company name, and description. "
        "Remove private info like emails, phone numbers, or internal project codes, but retain general text."
    ),
    "default_policy": (
        "Default policy: remove personal information such as emails, phone numbers, or IDs, "
        "but keep general job and company information that provides useful context."
    ),
}

# ------------------- Prompt Builder (no regex) -------------------

def build_reason_prompt(task, directive_text, field_name, field_value):
    return f"""
You are a privacy-preserving assistant. Your job is to decide whether this field can be safely shared.

Task: {task}
Directive: {directive_text}

Field name: {field_name}
Field value: {field_value}

If this field contains private, personal, or sensitive identifiers (emails, phone numbers, IDs, internal codes, etc.),
respond with DO_NOT_SHARE.
If it is general, descriptive, or job-related and safe to disclose, respond with SHARE.

Answer with only one word: SHARE or DO_NOT_SHARE.
"""

# ------------------- LLM Decision -------------------

def decide_share(pipe, prompts, gen_kwargs):
    outs = pipe(prompts, **gen_kwargs)
    answers = []
    for o in outs:
        if isinstance(o, list) and len(o) > 0 and isinstance(o[0], dict):
            txt = o[0].get("generated_text", "")
        elif isinstance(o, dict):
            txt = o.get("generated_text", "")
        else:
            txt = str(o)
        txt = txt.strip().lower()
        print("DEBUG:", txt)
        if "do_not_share" in txt or "do not share" in txt:
            answers.append(False)
        elif "share" in txt:
            answers.append(True)
        else:
            answers.append(False)
    return answers

# ------------------- Minimization Core -------------------

def minimize_records(df, pipe, task, fallback_directive, batch_size=8):
    gen_kwargs = {"max_new_tokens": 24, "do_sample": True, "temperature": 0.5}
    results = []
    prompts, meta = [], []

    for _, rec in tqdm(df.iterrows(), total=len(df), desc="Minimizing"):
        minimized = {"id": rec.get("id")}
        directive_key = str(rec.get("directive", fallback_directive)).strip()
        directive_text = DIRECTIVE_TEMPLATES.get(directive_key, DIRECTIVE_TEMPLATES["default_policy"])

        for k in ["job_title", "job_description", "company_name", "contact_info", "notes"]:
            v = str(rec.get(k, "")).strip()
            if not v:
                minimized[k] = ""
                continue

            prompt = build_reason_prompt(task, directive_text, k, v[:800])
            prompts.append(prompt)
            meta.append((minimized, k, v))

            if len(prompts) >= batch_size:
                decisions = decide_share(pipe, prompts, gen_kwargs)
                for dec, (m, key, val) in zip(decisions, meta):
                    m[key] = val if dec else ""
                prompts, meta = [], []

        # Flush remaining prompts for this record before appending
        if prompts:
            decisions = decide_share(pipe, prompts, gen_kwargs)
            for dec, (m, key, val) in zip(decisions, meta):
                m[key] = val if dec else ""
            prompts, meta = [], []

        results.append(minimized)

    # Flush leftovers
    if prompts:
        decisions = decide_share(pipe, prompts, gen_kwargs)
        for dec, (m, key, val) in zip(decisions, meta):
            m[key] = val if dec else ""

    return results

# ------------------- Main -------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="minimized_prompt_only.json")
    parser.add_argument("--task", default="general task")
    parser.add_argument("--directive_fallback", default="default_policy")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    print(f"Loading data from {args.csv} ...")
    df = pd.read_csv(args.csv, dtype=str).fillna("")
    print(f"{len(df)} rows loaded.")

    pipe = pipeline("text-generation", model=args.model, device=args.device, trust_remote_code=True)
    minimized = minimize_records(df, pipe, args.task, args.directive_fallback, batch_size=args.batch_size)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(minimized, f, indent=2)
    print(f"[✓] Saved minimized dataset to {args.out}")

if __name__ == "__main__":
    main()
