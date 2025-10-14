import argparse, json, re, os
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
    "internal_hr": (
        "Keep employee role and team details. "
        "Redact personal identifiers such as names, emails, and phone numbers. "
        "Retain job-related performance or position data."
    ),
    "vendor_sharing": (
        "Keep company and project information relevant to the vendor. "
        "Remove personal or client contact info, internal IDs, and sensitive pricing data."
    ),
    "client_report": (
        "Keep job title, company name, and project summary. "
        "Remove employee or client personal contact info and internal reference codes."
    ),
    "compliance_audit": (
        "Keep compliance-relevant job or company details. "
        "Remove personal identifiers, login info, and private contact details."
    ),
    "partner_data_exchange": (
        "Share only organizational and project-level information. "
        "Redact personal contact info, access credentials, and internal IDs."
    ),
    "employee_review": (
        "Retain role, department, and project context. "
        "Redact individual identifiers, feedback quotes, and private HR notes."
    ),
    "legal_request": (
        "Retain case or job context if required. "
        "Redact all personal info, including emails, addresses, and reference numbers."
    ),
    "research_dataset": (
        "Keep general role and company information. "
        "Redact all personal identifiers, codes, and contact information to ensure anonymity."
    ),
    "training_material": (
        "Keep generic job and company descriptions suitable for educational use. "
        "Remove any identifiers, contact info, or sensitive examples."
    ),
    "customer_support_case": (
        "Keep product or service details and company name. "
        "Redact customer names, emails, phone numbers, and ticket IDs."
    ),
    "internal_analytics": (
        "Keep organizational and performance metrics. "
        "Remove names, emails, or contact references to individual employees."
    ),
    "public_summary": (
        "Keep high-level role and company information suitable for public sharing. "
        "Redact internal references and private communications."
    ),
    "marketing_campaign": (
        "Keep company name, job title, and campaign descriptions. "
        "Remove customer identifiers, emails, or private leads."
    ),
    "security_monitoring": (
        "Keep technical incident summaries and organizational info. "
        "Redact IP addresses, credentials, or identifying details of individuals."
    ),
    "financial_audit": (
        "Keep transaction summaries and organizational context. "
        "Remove account numbers, personal identifiers, or confidential client info."
    ),
    "external_press_release": (
        "Keep official titles, company names, and descriptions. "
        "Redact internal reference codes, contact details, or unreleased information."
    ),
    "third_party_consulting": (
        "Keep project and role details that explain collaboration context. "
        "Redact personal contacts, internal project codes, or sensitive company data."
    ),
    "default_policy": (
        "Default policy: remove personal information such as emails, phone numbers, or IDs, "
        "but keep general job and company information that provides useful context."
    )
}

# ------------------- Regex Redactors -------------------

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\-\s().]{7,}\d")
REF_RE = re.compile(r"\b(?:REF[-\d]{6,}|\b[A-Z]{2,}-\d{3,}-\d{2,}-\d{3,})\b")

def redact_sensitive(text):
    if not isinstance(text, str): 
        return text
    text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = PHONE_RE.sub("[REDACTED_PHONE]", text)
    text = REF_RE.sub("[REDACTED_REF]", text)
    return text

# ------------------- Prompt Builder -------------------

def build_reason_prompt(task, directive_text, field_name, field_value):
    return f"""
You are a privacy-preserving assistant. Decide whether this field can be safely shared.

Task: {task}
Directive: {directive_text}

Field name: {field_name}
Field value (already redacted): {field_value}

If the field still contains private identifiers or sensitive context, output DO_NOT_SHARE.
Otherwise, if it’s safe or relevant to the task, output SHARE.

Respond with only one word: SHARE or DO_NOT_SHARE.
"""

# ------------------- LLM Logic -------------------

def decide_share(pipe, prompts, gen_kwargs):
    outs = pipe(prompts, **gen_kwargs)
    answers = []
    for o in outs:
        # Handle both dict and string output
        txt = o["generated_text"] if isinstance(o, dict) else str(o)
        # Keep only the *new* text (strip the prompt part)
        txt = txt.splitlines()[-1].strip().lower()
        # Fallback: also handle "share." and "do_not_share." forms
        print("DEBUG OUTPUT:", txt)
        if any(x in txt for x in ["do_not_share", "do not share", "don't share", "not share"]):
            answers.append(False)
        elif "share" in txt:
            answers.append(True)
        else:
            print("⚠️ Unclear model output:", txt)
            answers.append(False)
    return answers


def minimize_records(df, pipe, task, fallback_directive, batch_size=16, gen_kwargs=None):
    gen_kwargs = {"max_new_tokens": 16, "do_sample": True, "temperature": 0.4}
    results = []
    prompts, meta = [], []

    for _, rec in tqdm(df.iterrows(), total=len(df), desc="Minimizing"):
        minimized = {"id": rec.get("id")}
        directive_key = str(rec.get("directive", fallback_directive)).strip()
        directive_text = DIRECTIVE_TEMPLATES.get(directive_key, DIRECTIVE_TEMPLATES["default_policy"])

        for k in ["job_title", "job_description", "company_name", "contact_info", "notes"]:
            v = redact_sensitive(str(rec.get(k, "")))
            if v.strip() == "":
                minimized[k] = ""
                continue

            prompt = build_reason_prompt(task, directive_text, k, v[:600])
            prompts.append(prompt)
            meta.append((minimized, k, v))

            # Run the batch when full
            if len(prompts) >= batch_size:
                outs = pipe(prompts, **gen_kwargs)
                for o, (m, key, val) in zip(outs, meta):
                    # Handle output shape: dict, list of dicts, or string
                    if isinstance(o, list) and len(o) > 0 and isinstance(o[0], dict):
                        txt = o[0].get("generated_text", "")
                    elif isinstance(o, dict):
                        txt = o.get("generated_text", "")
                    else:
                        txt = str(o)

                    txt = txt.splitlines()[-1].strip().lower()

                    if "do_not_share" in txt or "do not share" in txt:
                        m[key] = ""
                    elif "share" in txt:
                        m[key] = val
                    else:
                        m[key] = ""
                prompts, meta = [], []

        results.append(minimized)

    # Flush any leftover prompts
    if prompts:
        outs = pipe(prompts, **gen_kwargs)
        for o, (m, key, val) in zip(outs, meta):
            # Handle output shape: dict, list of dicts, or string
            if isinstance(o, list) and len(o) > 0 and isinstance(o[0], dict):
                txt = o[0].get("generated_text", "")
            elif isinstance(o, dict):
                txt = o.get("generated_text", "")
            else:
                txt = str(o)

            txt = txt.splitlines()[-1].strip().lower()

            if "do_not_share" in txt or "do not share" in txt:
                m[key] = ""
            elif "share" in txt:
                m[key] = val
            else:
                m[key] = ""

    return results

# ------------------- Main -------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="minimized.json")
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
