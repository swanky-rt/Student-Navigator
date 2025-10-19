#!/usr/bin/env python3

"""
Augments synthetic job data with DOB, address, experience, and website.
Uses Faker for realistic values and reproducible randomization.
Ensures uniqueness and plausibility across generated fields.
"""

import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import re

# ---------- Config ----------
# Input/output paths and configuration constants for data augmentation
INPUT = "Data/synthetic_jobs.csv"
OUTPUT = "Data/synthetic_jobs_augmented.csv"

SEED = 12345              # deterministic seed for reproducibility
DOB_START_YEAR = 1970     # lower bound for date of birth
DOB_END_YEAR = 2002       # upper bound for date of birth
MAX_WEBSITE_ATTEMPTS = 5  # how many tries to ensure unique website
FAKER_LOCALE = "en_US"    # locale for Faker data generation

# ---------- Setup ----------
# Set seeds to make results reproducible
random.seed(SEED)
fake = Faker(FAKER_LOCALE)
Faker.seed(SEED)

# Use Faker.unique for name generation; ensures less repetition
fake_unique = fake.unique

# ---------- Helpers ----------

def random_dob(start_year=DOB_START_YEAR, end_year=DOB_END_YEAR):
    """Generate a random date of birth (YYYY-MM-DD) between given years."""
    year = random.randint(start_year, end_year)
    month = random.randint(1, 12)
    # restrict day to 1–28 to avoid invalid dates in shorter months
    day = random.randint(1, 28)
    return f"{year:04d}-{month:02d}-{day:02d}"

def calculate_years_experience_from_dob(dob_iso):
    """Estimate years of experience based on age (approx age - 22 ± small noise)."""
    try:
        dob = datetime.strptime(dob_iso, "%Y-%m-%d").date()
    except Exception:
        # fallback to random plausible value if parsing fails
        return random.randint(1, 15)
    today = date.today()
    # compute age and adjust by assuming career start at 22
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    base = max(age - 22, 0)
    noise = random.randint(-2, 5)  # add variability to make it realistic
    years = max(0, base + noise)
    return min(years, 40)  # clamp to max 40 years

def one_line_address():
    """Return a one-line US-style address (no line breaks)."""
    raw = fake.address()
    return raw.replace("\n", ", ")

def make_website_for_name(name, used_sites):
    """
    Generate a unique personal website based on person's name.
    Adds numeric suffix or random fallback if duplicates occur.
    """
    slug = re.sub(r'[^a-z0-9]', '', name.lower())  # keep alphanumerics only
    if not slug:  # fallback if slug is empty
        slug = f"user{random.randint(100,999)}"
    tlds = [".com", ".dev", ".io", ".me"]
    # try multiple combinations to ensure uniqueness
    for attempt in range(MAX_WEBSITE_ATTEMPTS):
        suffix = "" if attempt == 0 else str(random.randint(1, 9999))
        tld = random.choice(tlds)
        site = f"https://{slug}{suffix}{tld}"
        if site not in used_sites:
            used_sites.add(site)
            return site
    # fallback: guaranteed unique site name
    site = f"https://{slug}{random.randint(10000,99999)}.com"
    used_sites.add(site)
    return site

# ---------- Main enrichment ----------

def enrich(input_path=INPUT, output_path=OUTPUT):
    """Main function to enrich base dataset with DOB, address, experience, and website."""
    df = pd.read_csv(input_path)
    n = len(df)
    print(f"[INFO] Loaded {n} records from {input_path}")

    # Containers for generated attributes
    names = []
    dobs = []
    addresses = []
    years_ex = []
    personal_websites = []

    used_websites = set()  # track uniqueness of generated websites

    # Iterate over all rows and generate new synthetic attributes
    for i in range(n):
        try:
            # use Faker.unique for more varied names
            name = fake_unique.name()
        except Exception:
            # fallback if unique pool exhausted
            name_base = fake.name()
            name = f"{name_base} {random.randint(1,9999)}"
        names.append(name)

        # Generate random date of birth
        dob = random_dob()
        dobs.append(dob)

        # Generate one-line address
        addr = one_line_address()
        addresses.append(addr)

        # Estimate years of experience from DOB with small random variation
        yexp = calculate_years_experience_from_dob(dob)
        if random.random() < 0.12:
            yexp = max(0, yexp + random.randint(1, 4))
        years_ex.append(int(yexp))

        # Generate personal website (ensure uniqueness)
        site = make_website_for_name(name, used_websites)
        personal_websites.append(site)

    # Append new columns to dataframe
    df["dob"] = dobs
    df["address"] = addresses
    df["years_experience"] = years_ex
    df["personal_website"] = personal_websites

    # Save final augmented dataset
    df.to_csv(output_path, index=False)
    print(f"[INFO] Enriched dataset saved to {output_path}")
    return df


# ---------- If this file executed as script ----------
if __name__ == "__main__":
    # Run enrichment pipeline
    df_enriched = enrich()

    # Display quick preview of newly added fields
    print("\n[INFO] Sample of enriched records (first 5):")
    print(df_enriched[["dob", "years_experience", "personal_website"]].head(5).to_string(index=False))
