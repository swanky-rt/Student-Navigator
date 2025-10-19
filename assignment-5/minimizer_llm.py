#!/usr/bin/env python3
import argparse, json, re
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys



DIRECTIVE_TEMPLATES = {
    "recruiter_outreach": (
        "Goal: enable a recruiter to contact a potential candidate directly. "
        "Keep information that helps identify and reach out to the person "
        "(such as name, email, or professional title), but avoid sharing any "
        "irrelevant or deeply personal details. The focus is on usefulness for outreach, "
        "not full anonymity."
    ),

    "public_job_board": (
        "Goal: publish a job listing on a public website where anyone can read it. "
        "Only include generic, descriptive details about the job and company. "
        "Do NOT include any personal, traceable, or internal contact information. "
        "The focus is on high privacy, minimal exposure, and a professional public tone."
    ),

    "internal_hr": (
        "Goal: provide internal HR systems with data for workforce planning and analytics. "
        "Include role, department, performance-related metrics, or manager notes if relevant. "
        "Exclude personal contact info or external identifiers. The balance is moderate—"
        "useful for internal decisions but not personally invasive."
    ),

    "marketing_campaign": (
        "Goal: share aggregated data with marketing teams to plan targeted campaigns. "
        "Keep broad, demographic, or behavioral descriptors that describe audience trends. "
        "Do NOT include individual-level contact or identifiers. The focus is on group-level insight, "
        "not personal details."
    ),

    "research_dataset": (
        "Goal: release a dataset for academic or public research with strict anonymization. "
        "Keep only abstract or statistical information that cannot be linked back to individuals. "
        "Remove or generalize any name, ID, contact info, or internal code. "
        "The focus is maximum privacy and zero re-identification risk."
    ),

    "default_policy": (
        "Default policy: retain only information relevant to the purpose described, "
        "removing personal identifiers, internal codes, and contact details unless explicitly needed."
    ),
}


# ---------------------------------------------------------------------

# def build_minimizer_prompt(task, directive_text, record, redaction_strength):
#     return f"""<s>[INST]
#         You are a trusted privacy minimizer.
#         Given a TASK, a PRIVACY DIRECTIVE, and a USER RECORD (schema + values),
#         return only the minimal set of field NAMES required to complete the task.

#         TASK: {task}
#         PRIVACY DIRECTIVE: {directive_text}
#         PRIVACY STRENGTH: {redaction_strength:.2f}
#         USER RECORD: {json.dumps(record, indent=2)}

#         Respond with a valid JSON array of strings only, like:
#         ["job_title","company_name"]
#         [/INST]
#     """

def build_minimizer_prompt(task, directive_text, record, redaction_strength):
    return f"""
        You are a trusted privacy minimizer agent. In your reasoning, think like a domain expert following this directive.
        Base your decisions on the intent behind the directive rather than memorizing field names.

        Analyze the user's record below and reason step-by-step about what fields are
        **necessary to achieve the task** versus what fields are **private or irrelevant**.

        TASK: {task}
        PRIVACY CONTEXT: {directive_text}
        REDACTION STRENGTH: {redaction_strength:.2f}
        - 0.0 → share almost everything
        - 1.0 → share almost nothing

        USER RECORD (schema + values):
        {json.dumps(record, indent=2)}

        Think carefully:
        1. Which fields directly help accomplish the task?
        2. Which fields risk leaking personal identifiers or unnecessary internal info?
        3. For this redaction strength, decide the balance between usefulness and privacy.

        Then output only a valid JSON array of the selected field **names**.
        Example format:
        ["name", "email"]

        Your reasoning (optional) can appear before the JSON array, but the JSON array must be the final line.
    """

# ---------------------------------------------------------------------

# def generate_response(model, tokenizer, prompt):
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=128,
#         do_sample=False,
#         temperature=0.1,
#         pad_token_id=tokenizer.eos_token_id
#     )
#     text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     # Extract only model's answer (after [/INST])
#     if "[/INST]" in text:
#         text = text.split("[/INST]")[-1]
#     return text.strip()

def generate_response(model, tokenizer, record, task, directive_text, redaction_strength, debug=False):
    user_prompt = f"""You are a privacy-minimization expert. Your goal is to decide which fields from a user's record should be shared for the described purpose — balancing usefulness with privacy risk.
TASK: {task}
PRIVACY DIRECTIVE: {directive_text}
REDACTION STRENGTH: {redaction_strength:.2f}
(0.0 → share almost everything | 1.0 → share almost nothing)
USER RECORD: {json.dumps(record, indent=2)}
Think carefully:
1. Identify which fields directly enable the task.
2. Identify which fields could expose personal, traceable, or unnecessary details.
3. Adjust your decision based on redaction strength:
   - Low values (<0.3): favor utility, allow broader sharing.
   - Mid values (0.3–0.7): balance privacy and utility.
   - High values (>0.7): favor privacy, share only essential safe fields.
4. Consider that similar tasks may differ in purpose and sensitivity, so reason contextually.
Finally, output ONLY a JSON array of the field names to share.
The last line must be the JSON array, e.g.:
["column_name1","column_name2",...,"column_nameN"]
"""

    messages = [
        {"role": "system", "content": (
            "You are a trusted privacy gatekeeper. "
            "Your ONLY job is to decide which fields should be revealed, not to perform the task itself."
        )},
        {"role": "user", "content": user_prompt},
    ]
    
    sys.stdout.reconfigure(encoding='utf-8')

    # === 🟢 LOGGING: print the full prompt for the log file ===
    full_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    print("\n" + "="*80, flush=True)
    print(f"FULL PROMPT SENT TO MODEL", flush=True)
    print("="*80, flush=True)
    print(full_prompt, flush=True)
    print("="*80 + "\n", flush=True)

    # --- Encode for model ---
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    attention_mask = torch.ones_like(input_ids)

    model.eval()
    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # --- Decode output ---
    sequences = out.sequences if hasattr(out, "sequences") else out
    gen_tokens = sequences[0, input_ids.shape[-1]:]
    decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    print("\n" + "="*80, flush=True)
    print("RAW COMPLETION FROM MODEL", flush=True)
    print("="*80, flush=True)
    print(decoded, flush=True)
    print("="*80 + "\n", flush=True)

    return decoded
    
    user_prompt = f"""You are a privacy-minimization expert, not the person performing the task itself.
Your goal is to analyze what information from a user's record should be revealed to another system
that will later perform this task — while keeping privacy risks minimal.

TASK to be performed later: {task}
PRIVACY DIRECTIVE: {directive_text}
REDACTION STRENGTH: {redaction_strength:.2f}
(0.0 → share almost everything, 1.0 → share almost nothing)

USER RECORD (schema + values):
{json.dumps(record, indent=2)}

Your reasoning steps:
1. Identify which fields directly help the downstream system accomplish the task.
2. Identify which fields would expose personal or sensitive details.
3. Based on the redaction strength, choose the minimal useful subset.

Finally, output ONLY a JSON array of field names that may be safely shared.
The final line must be JUST the JSON array, e.g.:
["job_title", "company_name"]
"""

    messages = [
        {"role": "system", "content": (
            "You are a trusted privacy gatekeeper. "
            "Your ONLY job is to decide which fields should be revealed, not to perform the task itself."
        )},
        {"role": "user", "content": user_prompt},
    ]

    # ✅ --- PRINT the exact message content (the actual prompt the LLM will see)
    if debug:
        full_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        print("\n=== FULL PROMPT SENT TO MODEL ===\n")
        print(full_prompt)
        print("\n=================================\n")

    #  Prepare input for generation
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    attention_mask = torch.ones_like(input_ids)

    model.eval()
    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=96,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    sequences = out.sequences if hasattr(out, "sequences") else out
    gen_tokens = sequences[0, input_ids.shape[-1]:]
    decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    if debug:
        print("\n=== RAW COMPLETION ===\n", decoded[:800], "\n=====================\n")

    return decoded
# ---------------------------------------------------------------------

def extract_json_array(text, debug=False):
    # Take the last non-empty line
    last = ""
    for line in text.strip().splitlines():
        if line.strip():
            last = line.strip()

    # sanitize escaped underscores and stray backslashes
    clean = text.replace("\\_", "_").replace("\\\\", "\\")

    # 1) If last line is a JSON array, parse it
    if last.startswith("[") and last.endswith("]"):
        try:
            return json.loads(last.replace("\\_", "_"))
        except Exception:
            pass

    # 2) Fallback: find the last bracketed JSON-looking array anywhere
    m = list(re.finditer(r"\[[\s\S]*?\]", clean))
    if m:
        candidate = m[-1].group(0).replace("\\_", "_")
        try:
            return json.loads(candidate)
        except Exception as e:
            if debug:
                print("[WARN] JSON parsing failed on candidate:", e, "\n", candidate[:300])

    if debug:
        print("[WARN] No JSON array found in model output:\n", text[:300])
    return []


# ---------------------------------------------------------------------

def minimize_records(df, model, tokenizer, task, fallback_directive, redaction_strength=0.5, max_records=None, debug=False):
    print(df.head(1))
    results = []
    rows = df.to_dict(orient="records")
    if max_records:
        rows = rows[:max_records]

    for idx, rec in enumerate(tqdm(rows, total=len(rows), desc="Minimizing")):
        directive_key = str(rec.get("directive", fallback_directive)).strip()
        directive_text = DIRECTIVE_TEMPLATES.get(directive_key, DIRECTIVE_TEMPLATES["default_policy"])

        prompt = build_minimizer_prompt(task, directive_text, rec, redaction_strength)
        out_text = generate_response(model, tokenizer, rec, task, directive_text, redaction_strength)
            
        try:
            minimal_fields = extract_json_array(out_text, debug=debug)
        except Exception as e:
            minimal_fields = []
            if debug:
                    print(f"[WARN] JSON parsing failed: {e}\nRaw: {out_text[:150]}")
        if not minimal_fields:
            if debug:
                print(f"[WARN] No JSON found. Raw output:\n{out_text[:150]}")

        if not minimal_fields:
            # fallback: share only job-related fields
            minimal_fields = [k for k in rec.keys() if any(x in k for x in ["job", "company"])]

        minimized = {"id": rec.get("id"), "_redaction_strength": redaction_strength}
        for k, v in rec.items():
            minimized[k] = v if k in minimal_fields else ""
        minimized["id"] = rec.get("id")
        results.append(minimized)

    return results

# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="minimized.json")
    parser.add_argument("--task", default="general task")
    parser.add_argument("--directive_fallback", default="default_policy")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--redaction_strength", type=float, default=0.5)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"Loading data from {args.csv} ...")
    df = pd.read_csv(args.csv, dtype=str).fillna("")
    if args.max_records:
        df = df.head(args.max_records)
    print(f"{len(df)} rows loaded.")

    print(f"Loading model {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        dtype=torch.float16,
        trust_remote_code=True
    )

    minimized = minimize_records(
        df, model, tokenizer, args.task, args.directive_fallback,
        redaction_strength=args.redaction_strength,
        max_records=args.max_records,
        debug=args.debug
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(minimized, f, indent=2)
    print(f"[OK] Saved minimized dataset to {args.out}")

# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()
