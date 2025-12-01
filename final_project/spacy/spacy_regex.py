import re
import json
import pandas as pd
import spacy

# -------------------------
# Load spaCy model
# -------------------------
nlp = spacy.load("en_core_web_sm")

# -------------------------
# Regex patterns
# -------------------------

# Email
EMAIL_RE = re.compile(
    r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"
)

# Phone number (very simple pattern, US-style-ish)
PHONE_RE = re.compile(
    r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"
)

# IPv4 address
IP_RE = re.compile(
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
)

# US SSN: 123-45-6789
SSN_RE = re.compile(
    r"\b\d{3}-\d{2}-\d{4}\b"
)

# Credit card: 16 digits (e.g. 1234-5678-9012-3456 or 1234 5678 9012 3456 or 1234567890123456)
CREDIT_CARD_16_RE = re.compile(
    r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"
)

# Credit card last 4: e.g. "xxxx1234", "XXXX1234", "**** 1234", "last 4: 1234"
CREDIT_CARD_LAST4_RE = re.compile(
    r"(?:(?:xxxx|XXXX|\*{4}|last\s+4(?:\s+digits)?\s*[:\-]?)\s*)(\d{4})"
)

# Age pattern: "25 years old", "age 25", "age is 33", "aged 25", "25 yo", "25 y/o"
AGE_RE = re.compile(
    r"\b(?:age[d]?\s*(?:is\s*)?:?\s*(\d{1,3})|(?<!\d)(\d{1,3})\s*(?:years?\s*old|y/?o|yrs?\s*old))\b",
    re.IGNORECASE
)

# Sex/Gender pattern: explicit gender mentions
SEX_RE = re.compile(
    r"\b(male|female)\b",
    re.IGNORECASE
)

# -------------------------
# File names
# -------------------------
INPUT_CSV = "690-Project-Dataset-final.csv"
OUTPUT_CSV = "spacy/output_spacy_regex.csv"


def extract_regex_pii(text: str):
    """
    Use regex to find specific PII-like patterns in the text.
    Returns a list of dicts.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    results = []

    # Email
    for m in EMAIL_RE.finditer(text):
        results.append(
            {
                "text": m.group(),
                "label": "EMAIL",
                "start": m.start(),
                "end": m.end(),
                "source": "regex",
            }
        )

    # Phone
    for m in PHONE_RE.finditer(text):
        results.append(
            {
                "text": m.group(),
                "label": "PHONE",
                "start": m.start(),
                "end": m.end(),
                "source": "regex",
            }
        )

    # IP address
    for m in IP_RE.finditer(text):
        results.append(
            {
                "text": m.group(),
                "label": "IP_ADDRESS",
                "start": m.start(),
                "end": m.end(),
                "source": "regex",
            }
        )

    # SSN
    for m in SSN_RE.finditer(text):
        results.append(
            {
                "text": m.group(),
                "label": "SSN",
                "start": m.start(),
                "end": m.end(),
                "source": "regex",
            }
        )

    # Credit card 16 digits
    for m in CREDIT_CARD_16_RE.finditer(text):
        results.append(
            {
                "text": m.group(),
                "label": "CREDIT_CARD_16",
                "start": m.start(),
                "end": m.end(),
                "source": "regex",
            }
        )

    # Credit card last 4
    for m in CREDIT_CARD_LAST4_RE.finditer(text):
        # group(1) is just the 4 digits
        results.append(
            {
                "text": m.group(1),
                "label": "CREDIT_CARD_4",
                "start": m.start(1),
                "end": m.end(1),
                "source": "regex",
            }
        )

    # Age
    for m in AGE_RE.finditer(text):
        # group(1) is from "age 25" pattern, group(2) is from "25 years old" pattern
        age_val = m.group(1) if m.group(1) else m.group(2)
        if age_val:
            results.append(
                {
                    "text": age_val,
                    "label": "AGE",
                    "start": m.start(1) if m.group(1) else m.start(2),
                    "end": m.end(1) if m.group(1) else m.end(2),
                    "source": "regex",
                }
            )

    # Sex/Gender
    for m in SEX_RE.finditer(text):
        results.append(
            {
                "text": m.group(),
                "label": "SEX",
                "start": m.start(),
                "end": m.end(),
                "source": "regex",
            }
        )

    return results


def extract_pii(text: str) -> str:
    """
    Combine spaCy entities + regex matches.
    Return as a JSON string to be stored in the CSV.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    entities = []

    # 1) spaCy NER
    doc = nlp(text)
    for ent in doc.ents:
        entities.append(
            {
                "text": ent.text,
                "label": ent.label_,        # e.g. PERSON, ORG, GPE
                "start": ent.start_char,
                "end": ent.end_char,
                "source": "spacy",
            }
        )

    # 2) Regex-based PII
    entities.extend(extract_regex_pii(text))

    # 3) Return as JSON string
    return json.dumps(entities, ensure_ascii=False)


def main():
    # Read the input CSV (must have column "conversation")
    df = pd.read_csv(INPUT_CSV)

    # Apply PII extraction
    df["pii_entities"] = df["conversation"].apply(extract_pii)

    # Save to new CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved output to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()