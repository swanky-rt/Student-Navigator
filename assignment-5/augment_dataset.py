#!/usr/bin/env python3

import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import re

# ---------- Config ----------
INPUT = "Data/synthetic_jobs.csv"
OUTPUT = "Data/synthetic_jobs_augmented.csv"

SEED = 12345              # deterministic seed for reproducibility
DOB_START_YEAR = 1970
DOB_END_YEAR = 2002
MAX_WEBSITE_ATTEMPTS = 5  # attempts to make a unique website
FAKER_LOCALE = "en_US"    

# ---------- Setup ----------
random.seed(SEED)
fake = Faker(FAKER_LOCALE)
Faker.seed(SEED)

# Use Faker.unique for names but handle exhaustion gracefully
fake_unique = fake.unique

# ---------- Helpers ----------
def random_dob(start_year=DOB_START_YEAR, end_year=DOB_END_YEAR):
    """Return DOB string in ISO format YYYY-MM-DD between start_year and end_year."""
    year = random.randint(start_year, end_year)
    month = random.randint(1, 12)
    # choose safe day to avoid month-end issues
    day = random.randint(1, 28)
    return f"{year:04d}-{month:02d}-{day:02d}"

def calculate_years_experience_from_dob(dob_iso):
    """Plausible years_experience based on DOB: (age - 22) +/- noise, clamped."""
    try:
        dob = datetime.strptime(dob_iso, "%Y-%m-%d").date()
    except Exception:
        # fallback random
        return random.randint(1, 15)
    today = date.today()
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    base = max(age - 22, 0)  # assume start of career ~22
    noise = random.randint(-2, 5)
    years = max(0, base + noise)
    return min(years, 40)  # clamp to a sane upper bound

def one_line_address():
    """Return a single-line address suitable for regex extraction."""
    raw = fake.address()
    return raw.replace("\n", ", ")

def make_website_for_name(name, used_sites):
    """
    Create a simple personal website URL based on a name.
    Ensures uniqueness (probabilistic) by appending numbers if needed.
    Regex-friendly format: https://<slug><num>.(com|dev|io|me)
    """
    slug = re.sub(r'[^a-z0-9]', '', name.lower())
    # avoid empty slug
    if not slug:
        slug = f"user{random.randint(100,999)}"
    tlds = [".com", ".dev", ".io", ".me"]
    for attempt in range(MAX_WEBSITE_ATTEMPTS):
        suffix = "" if attempt == 0 else str(random.randint(1, 9999))
        tld = random.choice(tlds)
        site = f"https://{slug}{suffix}{tld}"
        if site not in used_sites:
            used_sites.add(site)
            return site
    # fallback (guaranteed unique)
    site = f"https://{slug}{random.randint(10000,99999)}.com"
    used_sites.add(site)
    return site

# ---------- Main enrichment ----------
def enrich(input_path=INPUT, output_path=OUTPUT):
    df = pd.read_csv(input_path)

    n = len(df)
    print(f"[INFO] Loaded {n} records from {input_path}")

    # Prepare containers
    names = []
    dobs = []
    addresses = []
    years_ex = []
    personal_websites = []

    used_websites = set()

    # Attempt to generate unique-ish names via Faker.unique, fallback to Faker() + counter
    for i in range(n):
        try:
            # try unique name first (less chance of repetition)
            name = fake_unique.name()
        except Exception:
            # if unique pool exhausted or error, fallback: name + small suffix
            name_base = fake.name()
            name = f"{name_base} {random.randint(1,9999)}"
        names.append(name)

        # DOB
        dob = random_dob()
        dobs.append(dob)

        # Address (one-line)
        addr = one_line_address()
        addresses.append(addr)

        # Years experience (plausible from dob)
        yexp = calculate_years_experience_from_dob(dob)
        # add small random variation to spread values
        if random.random() < 0.12:
            yexp = max(0, yexp + random.randint(1, 4))
        years_ex.append(int(yexp))

        # Personal website (unique-ish)
        site = make_website_for_name(name, used_websites)
        personal_websites.append(site)

    # Insert columns into dataframe; keep original columns untouched
    df["dob"] = dobs
    df["address"] = addresses
    df["years_experience"] = years_ex
    df["personal_website"] = personal_websites

    # Save result
    df.to_csv(output_path, index=False)
    print(f"[INFO] Enriched dataset saved to {output_path}")
    return df

# ---------- Regex helpers for extraction / leakage checks ----------
REGEX_PATTERNS = {
    "dob_iso": re.compile(r"\b(19[7-9]\d|200[0-2])-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b"),
    # address is fuzzy; this regex looks for typical US street-number + street pattern
    "address_one_line": re.compile(r"\d{1,5}\s+[A-Za-z0-9.\- ]+,\s*[A-Za-z .]+,\s*[A-Z]{2}\s*\d{5}|\d{1,5}\s+[A-Za-z0-9.\- ]+,\s*[A-Za-z .]+"),
    "years_experience": re.compile(r"\b(?:[0-9]|[1-3][0-9]|40)\b"),
    "personal_website": re.compile(r"https?://[A-Za-z0-9\-_]+\.[A-Za-z]{2,}(?:/[^\s]*)?")
}

# ---------- If this file executed as script ----------
if __name__ == "__main__":
    df_enriched = enrich()
    # print a small sample to stdout for quick verification
    print("\n[INFO] Sample of enriched records (first 5):")
    print(df_enriched[["dob", "years_experience", "personal_website"]].head(5).to_string(index=False))

    # Print regex patterns for downstream use
    print("\n[INFO] Regex patterns available in REGEX_PATTERNS dict (keys):", list(REGEX_PATTERNS.keys()))
    # Example: how to use one pattern (demonstration only)
    sample_text = "Contact: https://johnsmith.dev or DOB 1988-07-12 or Lives at 123 Main St, Springfield"
    example_hits = {k: v.findall(sample_text) for k, v in REGEX_PATTERNS.items()}
    print("\n[INFO] Example regex hits on sample text:", example_hits)
