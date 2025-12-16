# How to Extract PII

Extract PII from text based on domain-specific patterns learned by GRPO.

## Installation

### Dependencies

Install required packages from the project root:

```bash
cd final_project
pip install -r requirements.txt
```

Key dependencies:
- `spacy>=3.4.0` - Named Entity Recognition
- `pandas>=1.3.0` - Data processing
- `torch>=1.9.0` - Model loading (for GRPO integration)

### spaCy Model

Download the English spaCy model:

```bash
python -m spacy download en_core_web_sm
```

### Model Files

Ensure trained GRPO models exist in `final_project/models/`:
- `grpo_model.pt` (default)
- `groupedppo_model.pt` (optional)
- `vanillarl_model.pt` (optional)

The extractor will automatically load the appropriate model based on the `algorithm` parameter.

## Quick Start

```python
from pii_extraction.pii_extractor import extract_pii

# Classifier provides domain → extract relevant PII
entities = extract_pii("My SSN is 123-45-6789", domain="bank")
# Returns: [{"text": "123-45-6789", "label": "SSN", "start": 10, "end": 21}]
```

> **Note:** All paths (model files, imports) are resolved automatically relative to the project structure. Just ensure you're running from the `final_project/` directory or importing with the full module path `pii_extraction.pii_extractor`.

## How It Works

1. **GRPO determines what to extract** — Based on domain and directive, returns allowed PII types from trained model
2. **spacy_regex extracts PII** — Uses spaCy NER + regex patterns to find all PII in text, with false positive filtering
3. **pii_extractor filters results** — Maps spaCy labels to GRPO format and filters to only return PII types allowed for the domain

### Extraction Engine (`spacy_regex.py`)

The core extraction combines two methods:

**spaCy NER:**
- Detects: `PERSON`, `ORG`, `GPE`, `LOC`, `FAC` (facilities)
- Uses pre-trained `en_core_web_sm` model

**Regex Patterns:**
- `EMAIL` - Email addresses
- `PHONE` - Phone numbers (US-style)
- `IP_ADDRESS` - IPv4 addresses
- `SSN` - Social Security Numbers (123-45-6789)
- `CREDIT_CARD_16` - Full 16-digit credit cards
- `CREDIT_CARD_4` - Last 4 digits (from "ending in 1234" patterns)
- `DATE` - Dates in various formats (MM/DD/YYYY, YYYY-MM-DD, etc.)
- `AGE` - Age mentions ("age 25", "25 years old")
- `SEX` - Gender mentions ("male", "female")
- `COMPANY_DOMAIN` - Company domain names ("my company xyz.com")
- `PERSON` - Explicit name patterns ("my name is First Last")

**False Positive Filtering:**
- Removes spaCy entities that overlap with regex matches (prefers regex)
- Filters out credit card brands (Visa, Mastercard) incorrectly labeled as ORG
- Removes street names incorrectly labeled as PERSON
- Filters short CARDINAL values that are part of phone/SSN matches
- Removes temporal phrases ("last week", "yesterday") incorrectly labeled as DATE
- Relabels ORG/PRODUCT as PERSON when preceded by name context

**Deduplication:**
- Removes duplicate entities with same span
- Prefers regex source over spaCy for overlapping matches
- Removes CREDIT_CARD_4 when CREDIT_CARD_16 already contains those digits

### Label Mapping

The extractor automatically maps between spaCy/regex labels and GRPO format:
- `PERSON` → `NAME`
- `ORG`/`FAC` → `company`
- `GPE`/`LOC` → `location`
- `IP_ADDRESS` → `IP`
- `CREDIT_CARD_16`/`CREDIT_CARD_4` → `CREDIT_CARD`
- `DATE` → `DATE/DOB`
- `AGE` → `age`
- `SEX` → `sex`

Other labels (`EMAIL`, `PHONE`, `SSN`) map directly.

### Domain → PII Types Mapping

| Domain | PII Types Extracted |
|--------|---------------------|
| `bank` | `PHONE`, `EMAIL`, `DATE/DOB`, `SSN`, `CREDIT_CARD` |
| `restaurant` | `PHONE`, `EMAIL` |

## API Reference

### `extract_pii(text, domain, algorithm, directive)`

Extract PII from text, filtered by domain-specific GRPO patterns.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | str | Yes | - | Raw text to extract PII from |
| `domain` | str | Yes | - | `"bank"` or `"restaurant"` |
| `algorithm` | str | No | `"grpo"` | `"grpo"`, `"groupedppo"`, or `"vanillarl"` |
| `directive` | str | No | `"balanced"` | `"strictly"`, `"balanced"`, or `"accurately"` |

**Returns:** List of entity dicts, each containing:
- `text` (str): The extracted PII text
- `label` (str): PII type in GRPO format (e.g., "EMAIL", "SSN", "NAME")
- `start` (int): Character start position in input text
- `end` (int): Character end position in input text
- `source` (str): Extraction method - `"spacy"` or `"regex"`

**Example:**
```python
[
    {
        "text": "123-45-6789",
        "label": "SSN",
        "start": 10,
        "end": 21,
        "source": "regex"
    },
    {
        "text": "john@example.com",
        "label": "EMAIL",
        "start": 25,
        "end": 42,
        "source": "regex"
    }
]
```

```python
from pii_extraction.pii_extractor import extract_pii

# Bank domain - extracts financial PII
extract_pii("SSN: 123-45-6789, email: a@b.com", domain="bank")
# [{"text": "123-45-6789", "label": "SSN", ...}, {"text": "a@b.com", "label": "EMAIL", ...}]

# Restaurant domain - only contact PII  
extract_pii("SSN: 123-45-6789, email: a@b.com", domain="restaurant")
# [{"text": "a@b.com", "label": "EMAIL", ...}]  ← SSN filtered out

# With directive for privacy control
extract_pii("My email is a@b.com", domain="bank", directive="strictly")
```

### `extract_pii_by_type(text, domain, algorithm, directive)`

Same parameters as `extract_pii`, but returns values grouped by PII type.

**Returns:** Dict mapping PII type to list of extracted values

```python
from pii_extraction.pii_extractor import extract_pii_by_type

extract_pii_by_type("Call 555-1234, SSN 123-45-6789", domain="bank")
# {"PHONE": ["555-1234"], "SSN": ["123-45-6789"]}
```

## Command Line

```bash
# Run from the final_project directory
cd final_project

# Extract from text
python pii_extraction/pii_extractor.py --domain bank --text "My SSN is 123-45-6789"

# With directive
python pii_extraction/pii_extractor.py --domain bank --directive strictly --text "My SSN is 123-45-6789"

# JSON output
python pii_extraction/pii_extractor.py --domain bank --text "My SSN is 123-45-6789" --json

# Demo with sample texts
python pii_extraction/pii_extractor.py --domain bank
```

**CLI Arguments:**
| Argument | Required | Choices | Description |
|----------|----------|---------|-------------|
| `--domain` | Yes | `bank`, `restaurant` | Domain from classifier |
| `--text` | No | - | Text to extract PII from (demo texts if omitted) |
| `--directive` | No | `strictly`, `balanced`, `accurately` | Privacy/utility tradeoff |
| `--json` | No | - | Output as JSON |

## Supported PII Types

All 11 PII types from `common/config.py`:

| PII Type | Label | Example Patterns |
|----------|-------|------------------|
| `NAME` | Person names | "John Smith", "Jane Doe" |
| `PHONE` | Phone numbers | "555-123-4567", "(555) 123-4567" |
| `EMAIL` | Email addresses | "user@example.com" |
| `DATE/DOB` | Dates/birthdays | "01/15/1990", "1990-01-15" |
| `company` | Organizations | "Acme Corp", "Google" |
| `location` | Places | "New York", "California" |
| `IP` | IP addresses | "192.168.1.1" |
| `SSN` | Social Security | "123-45-6789" |
| `CREDIT_CARD` | Credit cards | "4111-1111-1111-1111" |
| `age` | Age mentions | "25" (from "age 25", "25 years old") |
| `sex` | Gender | "male", "female" |

## Directive Options

Controls the privacy/utility tradeoff threshold:

| Directive | Threshold | Description |
|-----------|-----------|-------------|
| `strictly` | ≥0.7 | Higher privacy, lower utility |
| `balanced` | ≥0.5 | Default balanced tradeoff |
| `accurately` | ≤0.3 | Higher utility, lower privacy |

## Integration with GRPO

The extractor uses `get_regex()` from `scripts/endpoint.py` to query trained RL models:

```python
from scripts.endpoint import get_regex

# Get allowed PII types for domain
pii_types = get_regex("grpo", "balanced", "bank")
# ['PHONE', 'EMAIL', 'DATE/DOB', 'SSN', 'CREDIT_CARD']

pii_types = get_regex("grpo", "balanced", "restaurant")
# ['PHONE', 'EMAIL']
```

**How it works:**
1. Loads trained policy model from `models/{algorithm}_model.pt`
2. Builds state vector with all PII types present
3. Queries policy with directive threshold (strictly ≥0.7, balanced ≥0.5, accurately ≤0.3)
4. Returns list of PII types that should be extracted/shared

**Error Handling:**
- If model file is missing, returns empty list `[]`
- If domain/algorithm/directive is invalid, prints error and returns `[]`
- The extractor will return no entities if `get_regex()` fails

---

## Running the Extraction Pipeline

To process datasets and evaluate PII extraction performance:

### Step 1: Extract PII from Datasets

```bash
python pii_extraction/spacy_regex.py
```

**What it does:**
- Processes CSV files from `final_project/` root directory
- Extracts PII using spaCy NER + regex patterns
- Applies false positive filtering (removes incorrect spaCy labels)
- Saves results with `pii_entities` column (JSON format)

**Input:** CSV files with `conversation` column:
- `690-Project-Dataset-final.csv`
- `690-Project-Dataset-imbalanced.csv`
- `690-Project-Dataset-bank-balanced.csv`

**Output:** `extracted_pii/*_extracted.csv` files with added `pii_entities` column

### Step 2: Compute Metrics

```bash
python pii_extraction/compute_pii_metrics.py
```

**What it does:**
- Compares extracted PII against ground truth labels
- Computes precision, recall, F1 (micro and macro-averaged)
- Calculates exact match accuracy
- Generates per-row and aggregate metrics

**Input:** `extracted_pii/*_extracted.csv` files (must have `ground_truth` column)

**Output:** 
- `pii_metrics/*_metrics.csv` - Per-row metrics with columns: `predicted_labels`, `tp`, `fp`, `fn`, `precision`, `recall`, `f1`, `exact_match`
- `pii_metrics/all_datasets_summary.csv` - Aggregate metrics across all datasets

**Ground Truth Format:** The `ground_truth` column should contain a string like `"[NAME, PHONE, EMAIL]"` or `"['NAME', 'PHONE']"`

### Step 3: Analyze Errors

```bash
python pii_extraction/analyze_errors.py
```

**What it does:**
- Identifies false positives (extra labels) and false negatives (missed labels)
- Counts errors by PII type
- Extracts example conversations with errors
- Generates cross-dataset error comparison

**Input:** `pii_metrics/*_metrics.csv` files

**Output:**
- `error_analysis/error_summary.csv` - Consolidated error counts by type
- `error_analysis/*_fn_examples.csv` - False negative examples (missed PII)
- `error_analysis/*_fp_examples.csv` - False positive examples (incorrect detections)

### Output Directory Structure

```
pii_extraction/
├── extracted_pii/          # Step 1 output
│   ├── dataset_final_extracted.csv
│   ├── dataset_balanced_extracted.csv
│   └── dataset_1500_bank_balanced_extracted.csv
├── pii_metrics/            # Step 2 output
│   ├── dataset_final_metrics.csv
│   ├── dataset_balanced_metrics.csv
│   ├── dataset_1500_bank_balanced_metrics.csv
│   └── all_datasets_summary.csv
└── error_analysis/         # Step 3 output
    ├── error_summary.csv
    ├── dataset_final_fp_examples.csv
    ├── dataset_final_fn_examples.csv
    └── ...
```

All paths are resolved automatically relative to script locations.
