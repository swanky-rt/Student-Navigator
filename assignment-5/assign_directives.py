import pandas as pd
import re
import random

# 20 directive presets
DIRECTIVE_TEMPLATES = [
    "recruiter_outreach",
    "public_job_board",
    "internal_hr",
    "vendor_sharing",
    "client_report",
    "compliance_audit",
    "partner_data_exchange",
    "employee_review",
    "legal_request",
    "research_dataset",
    "training_material",
    "customer_support_case",
    "internal_analytics",
    "public_summary",
    "marketing_campaign",
    "security_monitoring",
    "financial_audit",
    "external_press_release",
    "third_party_consulting",
    "default_policy"
]

# Pattern-based heuristic mapping
PATTERN_RULES = {
    "recruiter_outreach": [r"recruit", r"apply", r"candidate"],
    "public_job_board": [r"apply at", r"hiring", r"job board"],
    "internal_hr": [r"employee", r"review", r"internal"],
    "vendor_sharing": [r"vendor", r"supplier", r"contract"],
    "client_report": [r"client", r"project", r"report"],
    "compliance_audit": [r"audit", r"compliance"],
    "partner_data_exchange": [r"partner", r"data exchange"],
    "legal_request": [r"legal", r"subpoena"],
    "research_dataset": [r"study", r"research", r"dataset"],
    "training_material": [r"training", r"learning"],
    "customer_support_case": [r"support", r"ticket"],
    "marketing_campaign": [r"marketing", r"campaign"],
    "security_monitoring": [r"security", r"alert", r"incident"],
    "financial_audit": [r"invoice", r"transaction"],
    "external_press_release": [r"press", r"media"],
    "third_party_consulting": [r"consult", r"advisor"],
}

def infer_directive(text):
    """Return a directive label based on keyword patterns."""
    text = str(text).lower()
    for directive, patterns in PATTERN_RULES.items():
        for pat in patterns:
            if re.search(pat, text):
                return directive
    # default fallback
    return random.choice(DIRECTIVE_TEMPLATES[-3:])  # default/random

def main():
    df = pd.read_csv("Data/synthetic_jobs.csv", dtype=str).fillna("")

    directives = []
    for _, row in df.iterrows():
        combined = " ".join([row.get("job_title", ""), row.get("job_description", ""), row.get("notes", "")])
        directive = infer_directive(combined)
        directives.append(directive)

    df["directive"] = directives
    out_path = "Data/synthetic_jobs_with_directives.csv"
    df.to_csv(out_path, index=False)
    print(f"[✓] Added 'directive' column and saved to: {out_path}")
    print("Directive distribution:\n", df["directive"].value_counts().to_string())

if __name__ == "__main__":
    main()
