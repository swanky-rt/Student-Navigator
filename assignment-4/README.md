# PII Filtering: Week 5 Assignment 4 - Prototype PII Detection and Redaction

**Lead for this assignment:** Richard Watson Stephen Amudha

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Detector](#detector)
4. [Folder Structure](#folder-structure)
4. [Redaction Modes](#redaction-modes)
5. [LLM Redaction](#llm-redaction)
6. [Adversarial Cases](#adversarial-cases)
8. [Implications](#implications-where-this-system-is-sufficient-vs-risky-for-your-project)
7. [Evaluation Metrics](#evaluation-metrics)
   - [Precision, Recall, F1 per Class](#precision-recall-f1-per-class)
   - [Residual Leakage Rate](#residual-leakage-rate)
   - [Adversarial Tests: Caught vs Missed](#adversarial-tests-caught-vs-missed)
8. [Results](#results)
   - [Precision/Recall/F1 for each Class](#precisionrecallf1-for-each-class)
   - [Residual Leakage](#residual-leakage)
   - [Runtime by Mode](#runtime-by-mode)
   - [Utility vs Privacy](#utility-vs-privacy)
9. [Artifacts](#artifacts)
10. [How to Run the PII Filtering System](#how-to-run-the-pii-filtering-system)
11. [Extra Credit: Running the PHI3 Mini Instruct Model](#extra-credit-running-the-phi3-mini-instruct-model)
12. [AI Disclosure](#ai-disclosure)
13. [References](#references)
14. [How We Used LLMs](#how-we-used-llms)
15. [What We Did Ourselves](#what-we-did-ourselves)

## Overview

This project aims to implement a **PII detection and redaction** system to protect personal information in text data. The system uses a combination of **regex patterns** for detection and a **language model (Ollama Mistral)** for advanced redaction. The system is evaluated based on metrics such as **Precision**, **Recall**, **F1-Score**, and **Residual Leakage** to assess how well it can detect and redact sensitive information while maintaining data utility.

## Dataset

The dataset contains **340 rows** of mixed PII-like data, including fields such as **Email**, **Phone Number**, **Credit Card Number**, **Social Security Number (SSN)**, **Date of Birth**, **Name**, **IP Address**, and **IPv6**. These fields were synthetically generated to simulate real-world personal information.

The following fields contain PII-like strings:

* **Email**: Contains email addresses.
* **Phone**: Contains phone numbers in various formats.
* **Credit Card**: Contains credit card numbers in common formats.
* **SSN**: Contains Social Security Numbers.
* **Date**: Contains dates in different formats (e.g., MM/DD/YYYY, YYYY-MM-DD).
* **Name**: Contains names with possible special characters or uppercase letters.
* **IP**: Contains IP addresses in various formats.

## Detector

The detector uses **regex patterns** to identify and extract PII fields from text. The regex patterns cover several classes of PII, including:

```python
PATTERNS = {
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    "PHONE": re.compile(r"\b(?:\+?\d{1,3}[\s\-\.]?)?(?:\(?\d{3}\)?[\s\-\.]?\d{3}[\s\-\.]?\d{4,})\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b", re.IGNORECASE),
    "SSN": re.compile(r"\b(ssn[:\s\-]*)(\d{3}[\s\-]?\d{2}[\s\-]?\d{4})\b", re.IGNORECASE),
    "DATE": re.compile(r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b"),
    "NAME": re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"),
    "IP": re.compile(r"\b(?:IP[:\s]*)?(?:\d{1,3}\.){3}\d{1,3}\b"),
}
````

## Folder Structure

Here’s the complete folder structure of the project:
````
assignment-4/
│
├── readme.md
├── code/
│   ├── pii.py
│   └── pii_phi3.py
├── requirements.txt
├── dataset/
│   └── synthetic_jobs.csv
├── phi_3_report/
│   ├── adversarial_report_phi3.csv
│   ├── metrics_phi3.csv
│   ├── metrics_phi3.json
│   ├── synthetic_jobs_cleaned_phi3.csv
│   └── phi3_figs/
│       ├── phi3_metrics_comparison.png
│       ├── phi3_per_class_precision.png
│       ├── phi3_adversarial_heatmap.png
│       ├── phi3_residual_leakage.png
│       ├── phi3_per_class_f1.png
│       ├── phi3_per_class_recall.png
├── plots/
│   ├── micro_by_mode.png
│   ├── per_class_recall.png
│   ├── per_class_precision.png
│   ├── residual_leakage.png
│   ├── per_class_f1.png
│   ├── utility_vs_privacy.png
│   ├── adversarial_heatmap.png
│   └── runtime_by_mode.png
├── redaction_mode_converted_dataset/
│   ├── synthetic_jobs_cleaned_strict.csv
│   ├── synthetic_jobs_cleaned_partial.csv
│   └── synthetic_jobs_cleaned_llm.csv
└── report/
    ├── adversarial_report.csv
    ├── metrics.csv
    └── metrics.json

````
This structure contains all the necessary components:

Code: The main scripts (pii.py, pii_phi3.py) for detection and redaction.

Dataset: The synthetic dataset (synthetic_jobs.csv).

Reports: Includes adversarial test results, metrics, and redacted datasets in different modes.

Figures: Plots saved during the evaluation process for comparison and analysis.
## Redaction Modes

There are three redaction modes implemented in the system:

1. **Strict Masking**: Replace PII fields with placeholders.

   * Example: `Email: [EMAIL]`, `Phone: [PHONE]`
2. **Partial Masking**: Obscure parts of the PII data.

   * Example: `Credit Card: ***-**-1234`, `Email: j***@example.com`
3. **LLM Masking**: Use a **language model (Ollama Mistral)** to detect and redact PII fields.

   * The model is run locally and is designed to redact the PII after identifying it in the text.

## LLM Redaction

The **Ollama Mistral model** is used for the **LLM redaction mode**. After the initial regex-based detection, the data is fed to the model, which further processes and redacts the identified PII fields.

## Adversarial Cases

The system is tested on several **adversarial cases** to assess its robustness against obfuscated PII. These cases include techniques such as **spaced digits**, **leet speak**, **Unicode confusables**, and **inserted dots**. Here are the adversarial cases tested:

```python
ADVERSARIAL_CASES = [
    ("j . d o e [ at ] example [ dot ] com", {"EMAIL"}),
    ("j.a.n.e.d.o.e [at] example [dot] com", {"EMAIL"}),
    ("+1 (2 0 2) 5 5 5 - 0 1 7 3", {"PHONE"}),
    ("4 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1", {"CREDIT_CARD"}),
    ("credit card: 4111-1111-1111-1111", {"CREDIT_CARD"}),
    ("123 45 6789", {"SSN"}),
    ("192\u2024 168\u2024 1\u2024 55", {"IP"}),   # U+2024 one dot leader
    ("192․168․1․55", {"IP"}),                    # Armenian full stop
    ("2023-0l-15", {"DATE"}),                    # 'l' vs '1' (likely miss)
    ("03/22/1997", {"DATE"}),
    ("CℓientName met Jane Doe", {"NAME"}),       # confusable 'ℓ'
    ("S@rah C0nn0r applied", {"NAME"}),          # leetspeak (likely miss)
    ("2001:0db8:85a3:0000:0000:8a2e:0370:7334", {"IPV6"}),
]
```

The results of these tests help assess the **detection accuracy** and **robustness** of the PII filtering system.

# Adversarial Tests: Caught vs Missed

### 1. Case: "j . d o e [ at ] example [ dot ] com (EMAIL)
- **Expected**: EMAIL
- **Regex Detected**: No
- **Extra Credit Detected**: No
- **Explanation**: Both **Regex** and the **LLM model** missed this case due to the obfuscation (spaces and "[at]" and "[dot]"). This is a common failure mode for regex-based systems that are not designed to handle such spaces or obfuscations.

---

### 2. Case: "j.a.n.e.d.o.e [at] example [dot] com (EMAIL)
- **Expected**: EMAIL
- **Regex Detected**: No
- **Extra Credit Detected**: No
- **Explanation**: Similar to Case 1, both detection systems missed this due to the use of dots between letters and obfuscations of the "@" and "." symbols.

---

### 3. Case: "+1 (2 0 2) 5 5 5 - 0 1 7 3 (PHONE)
- **Expected**: PHONE
- **Regex Detected**: No
- **Extra Credit Detected**: No
- **Explanation**: Both systems missed this due to the inserted spaces between digits. Regex struggled with detecting phone numbers when digits were not contiguous.

---

### 4. Case: "4 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1 (CREDIT_CARD)
- **Expected**: CREDIT_CARD
- **Regex Detected**: Yes
- **Extra Credit Detected**: Yes
- **Explanation**: The **Regex** correctly detected this case despite the spaces. However, the system sometimes struggles with spaces, and it appears the **Extra Credit LLM** was better at detecting the full card number.

---

### 5. Case: "credit card: 4111-1111-1111-1111 (CREDIT_CARD)
- **Expected**: CREDIT_CARD
- **Regex Detected**: Yes
- **Extra Credit Detected**: Yes
- **Explanation**: This is a standard format, so both systems detected the **CREDIT_CARD** correctly without any issues.

---

### 6. Case: "123 45 6789 (SSN)
- **Expected**: SSN
- **Regex Detected**: No
- **Extra Credit Detected**: No
- **Explanation**: Both systems missed this due to spaces between digits. Regex requires contiguous digits or specific delimiters for SSNs.

---

### 7. Case: "192․ 168․ 1․ 55 (IP)
- **Expected**: IP
- **Regex Detected**: No
- **Extra Credit Detected**: No
- **Explanation**: The **Regex** and **LLM** both failed to detect this **IP** due to the use of **Unicode full stop characters** instead of regular periods.

---

### 8. Case: "192․168․1․55 (IP)
- **Expected**: IP
- **Regex Detected**: No
- **Extra Credit Detected**: No
- **Explanation**: The issue with **Unicode full stop** characters made both the **Regex** and **LLM** miss this **IP** detection.

---

### 9. Case: "2023-0l-15 (DATE)
- **Expected**: DATE
- **Regex Detected**: No
- **Extra Credit Detected**: No
- **Explanation**: The **regex** missed this case because the lowercase "l" was used instead of "1" (digit), which is a typical **leet speak** obfuscation.

---

### 10. Case: "03/22/1997 (DATE)
- **Expected**: DATE
- **Regex Detected**: Yes
- **Extra Credit Detected**: Yes
- **Explanation**: Both systems detected the **DATE** correctly, as it follows a standard and straightforward format without obfuscation.

---

### 11. Case: "Jane Doe met John Smith (NAME)
- **Expected**: NAME
- **Regex Detected**: Yes
- **Extra Credit Detected**: Yes
- **Explanation**: Both systems successfully detected **NAME** as expected, handling simple name patterns without issues.

---

### 12. Case: "S@rah C0nn0r applied (NAME)
- **Expected**: NAME
- **Regex Detected**: No
- **Extra Credit Detected**: No
- **Explanation**: The **Regex** and **LLM** missed this due to **leet speak** with "S@rah" and "C0nn0r". This is a failure for both systems that are not designed to detect such obfuscations.

---

### 13. Case: "2001:0db8:85a3:0000:0000:8a2e:0370:7334 (IPV6)
- **Expected**: IPV6
- **Regex Detected**: Yes
- **Extra Credit Detected**: Yes
- **Explanation**: Both the **Regex** and **LLM** correctly identified this valid **IPV6** address without issues.

---

### Known Failure Modes Explained

1. **Unicode Confusables**:
   - The system failed to detect **IP addresses** (Cases 7 and 8) when Unicode characters like **U+2024** or Armenian full stop were used instead of a period. These are visually similar but aren't recognized by the **Regex**.
   
2. **Leetspeak or Similar Alterations**:
   - For **leetspeak** variations like **"S@rah C0nn0r"** (Case 12), both systems missed the detection. While the **LLM** model might perform better, it still requires training or contextual awareness to handle such obfuscations, which was not fully covered here.

3. **Obfuscated Digits (Spaces Between Numbers)**:
   - Cases like the **PHONE** number ("+1 (2 0 2) 5 5 5 - 0 1 7 3") and **SSN** ("123 45 6789") were missed due to the insertion of **spaces** between digits. Regex struggles to identify PII when digits are not contiguous.

---

### Conclusion:

- **Regex** based methods are prone to failing in **complex obfuscations** like **Unicode confusion**, **leet speak**, and **spaced digits**. 
- **LLM redaction** (as explored in the **extra credit** cases) is more flexible but still misses obfuscated data types like **leet speak** or special character-based obfuscations.
- The **Regex** method performs well for standard formats but struggles with **adversarial obfuscation** like **spaces**, **leet speak**, and **Unicode character replacements**. 


## Implications: Where This System is Sufficient vs Risky for Your Project

| **Criteria**                     | **Sufficient Use Cases**                                                      | **Risky Use Cases**                                                   |
|-----------------------------------|-------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| **Standard PII Formats**          | The system performs well for **standard PII** such as emails, phone numbers, credit card numbers, SSNs, and dates. | The system might miss **novel PII formats** like social media handles or unique identifiers. |
| **Basic Redaction**               | **Strict** and **Partial** redaction modes effectively mask standard PII. LLM improves detection accuracy for general cases. | The system struggles with **complex obfuscations**, especially involving **spaces**, **Unicode characters**, or **leet speak**. |
| **Evaluation Metrics**            | Provides clear metrics (precision, recall, F1-score) to assess performance on standard PII. | **Obfuscated data** or **adversarial attacks** can lead to **missed PII**. |
| **Leetspeak**                     | Handles typical **standard PII** without issues.                              | The system misses **leet speak** and substitutions, such as "S@rah C0nn0r". |
| **Obfuscation Handling**          | Effective for detecting **clean and unmodified** PII in controlled environments. | **Spaces**, **special characters**, and **unusual formatting** lead to missed detection. |

### Conclusion
- **Sufficient Use Cases**: Ideal for **standard PII** in clean, well-structured datasets.
- **Risky for Obfuscated Data**: The system struggles with **obfuscated** or **novel PII** formats, leading to potential **privacy risks**.

## Evaluation Metrics

### Precision, Recall, F1 per Class

The system is evaluated based on the following metrics for each class:

* **Precision**: The proportion of true positives among the detected items.
* **Recall**: The proportion of true positives detected out of all actual instances.
* **F1-Score**: The harmonic mean of Precision and Recall, which provides a balance between them.

The micro average metrics for all classes are also calculated.

```text
=== Metrics (micro) ===
   mode    precision    recall       f1
 Strict      1.00000    1.00000   1.000000
Partial      1.00000    1.00000   1.000000
    LLM      0.94319    0.95784   0.950459
```

### Residual Leakage Rate

The **residual leakage rate** measures the percentage of documents where **high-risk items** like **Credit Card** or **SSN** were **missed** during redaction. This helps to evaluate the system’s effectiveness in preventing leakage of sensitive information.

```text
=== Residual leakage (%, high-risk) ===
   mode      CREDIT_CARD    SSN   HIGH_RISK_MICRO
 Strict          0.0000   0.0000       0.0000
Partial          0.0000   0.0000       0.0000
    LLM          7.3394   3.5398       6.5041
```

## Results

### 1. **Precision/Recall/F1 for each Class**:

The following figures show the **precision**, **recall**, and **F1-score** per class for each redaction mode.

![Precision, Recall, F1 per Class](plots/per_class_f1.png)

### 2. **Residual Leakage**:

The system achieves **zero residual leakage** in **Strict** and **Partial** modes but has some leakage in the **LLM** mode for high-risk items like **Credit Cards** and **SSNs**.

![Residual Leakage](plots/residual_leakage.png)

### 3. **Runtime by Mode**:

**Strict** and **Partial** modes are significantly faster than **LLM**, which requires more processing time due to its use of a language model.

![Runtime by Mode](plots/runtime_by_mode.png)

### 4. **Utility vs Privacy**:

The trade-off between **privacy** (residual leakage) and **utility** (tokens preserved) is illustrated below.

![Utility vs Privacy](plots/utility_vs_privacy.png)

## Artifacts

All generated plots are stored in the same directory in a separate folder `plots`.

## How to Run the PII Filtering System

### Step 1: Setup Python Virtual Environment

1. **Create a virtual environment** (optional but recommended):
   If you haven't already created a virtual environment, you can do so by running the following commands:

   ```bash
   # Create a virtual environment named 'venv'
   python -m venv venv

   # Activate the virtual environment:
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

### Step 2: Install Dependencies

2. **Install all required dependencies**:
   The script relies on several Python libraries that are specified in the `requirements.txt` file. To install these dependencies, run:

   ```bash
   pip install -r requirements.txt
   ```

   Your `requirements.txt` should include the following packages (and any other dependencies you may need):

   ```
   pandas
   numpy
   matplotlib
   unidecode
   ```

### Step 3: Install Ollama and Mistral Model

3. **Install Ollama**:
   The **Ollama** tool is required for generating synthetic data and performing LLM-based redactions. Follow the official installation instructions for Ollama based on your platform from the [Ollama Installation Guide](https://ollama.com/).

   You can also try installing Ollama using `brew` (on macOS) or other available methods for your operating system:

   ```bash
   # Example for macOS (using Homebrew)
   brew install ollama
   ```

4. **Verify Installation**:
   After installing Ollama, verify that the installation is working correctly by running:

   ```bash
   ollama list
   ```

   This should list available models, including the **Mistral model** if it's installed correctly.

5. **Install Mistral Model**:
   Once Ollama is set up, you need to install the **Mistral** model. You can do so by running:

   ```bash
   ollama run mistral
   ```

   This will download and set up the Mistral model locally.

### Step 4: Running the Script

Now that the environment is set up and all dependencies are installed, you can run the PII filtering system. Below are the main commands you can use:

1. **Basic Command**:
   To generate 1000 rows using the **Mistral model** and process the PII redaction:

   ```bash
   python pii.py --rows 1000 --model mistral
   ```

2. **Skip LLM Redaction**:
   If you want to skip the LLM redaction step and use only strict and partial redaction:

   ```bash
   python pii.py --skip-llm-redaction
   ```

3. **Specify Rows and Output Directory for Plots**:
   To run with 500 rows and save plots to the `figs` folder:

   ```bash
   python pii.py --rows 500 --plots-dir figs
   ```

4. **Run with Custom Model**:
   If you want to use a different model for synthetic data generation:

   ```bash
   python pii.py --rows 1000 --model <your_model_name>
   ```

### Step 5: Understanding the Output

After running the script, you will get several output files:

1. **Cleaned CSV Files**:

   * `synthetic_jobs_cleaned_strict.csv` — Strict redaction of PII.
   * `synthetic_jobs_cleaned_partial.csv` — Partial redaction of PII.
   * `synthetic_jobs_cleaned_llm.csv` — LLM-based redaction of PII (if LLM mode is enabled).

2. **Adversarial Test Report**:

   * `adversarial_report.csv` — This file contains the results of adversarial cases, showing whether the system successfully detected obfuscated PII.

3. **Metrics**:

   * `metrics.json` and `metrics.csv` — These files contain precision, recall, F1, and other metrics for each PII class (email, phone number, credit card, SSN, etc.).

4. **Figures (Plots)**:

   * Plots will be saved in the folder specified by the `--plots-dir` argument (default is `figs`). These include:

     * **Precision/Recall/F1 per Class**: Per-class evaluation of the redaction modes.
     * **Residual Leakage**: Bar plot of residual leakage for high-risk PII classes (credit card, SSN).
     * **Runtime by Mode**: Comparison of processing time for strict, partial, and LLM-based redaction.
     * **Utility vs Privacy**: Scatter plot showing the trade-off between utility (tokens preserved) and privacy (residual leakage).
     * **Adversarial Heatmap**: Heatmap showing the detection performance on adversarial cases.

5. **README**:

   * The script will automatically update a `README.md` file with information about the saved plots and where they are located.

### Step 6: Customizing the Script

* You can modify the number of rows generated, the model used for generation, or the directory where plots are saved by changing the respective arguments in the command line.
* If you want to use a different model for LLM redaction, specify it with the `--model` option.

### Step 7: Results Analysis

After running the script, you can analyze the results through the generated metrics and plots. The key metrics include:

1. **Precision, Recall, F1 Scores**: Evaluate the performance of the detection and redaction for each PII type.
2. **Residual Leakage**: Measure the effectiveness of each redaction mode at preventing high-risk PII leakage.
3. **Adversarial Test Results**: Review the system’s ability to detect obfuscated PII in adversarial cases.
4. **Runtime**: Compare the processing time for each redaction mode.

### Conclusion

By following these steps, you will be able to generate synthetic data, perform PII detection and redaction, evaluate the system’s effectiveness, and generate a variety of metrics and plots to analyze performance.

If you encounter issues or need further customization, feel free to modify the regex patterns, adversarial cases, or redaction methods as needed.

## Extra Credit: Running the PHI3 Mini Instruct Model

### Overview

For extra credit, we have experimented with a new LLM model, the **PHI3 Mini Instruct Model**, to compare its performance with the previously used **Ollama Mistral** model for **PII detection and redaction**. This model has been tested using the same dataset and the same evaluation criteria (Precision, Recall, F1-Score, Residual Leakage, and Adversarial Case Detection) as the original model.

### Results

The results from running the **PHI3 Mini Instruct Model** are saved separately for comparison and analysis. The following artifacts were generated during this extra credit task:

1. **PII Detection and Redaction Results**:

   * The redacted versions of the synthetic dataset using the **PHI3 Mini Instruct Model** are saved in CSV format. These files are located in the `phi_3_report` folder.
2. **Evaluation Metrics**:

   * Precision, Recall, and F1 scores are calculated for each PII class, as well as the **micro-average metrics** and **residual leakage** rates.
3. **Adversarial Cases**:

   * Similar to the base model, the **PHI3 Mini Instruct Model** was tested on adversarial cases, and the results are included in the **phi3_report** folder.

### Folder Structure

All generated files related to the **PHI3 Mini Instruct Model** are saved in the `phi_3_report` folder. This includes:

* **Redacted Files**: The cleaned versions of the dataset using the new LLM model are saved as:

  * `synthetic_jobs_cleaned_phi3.csv`

* **Metrics and Evaluation**:

  * `metrics_phi3.json` — Contains the precision, recall, F1, and other metrics for the **PHI3 Mini Instruct Model**.
  * `metrics_phi3


.csv` — CSV file with detailed metrics.

* **Adversarial Report**:

  * `adversarial_phi3_report.csv` — Contains the results of adversarial tests conducted with the **PHI3 Mini Instruct Model**.

* **Figures**:

  * Plots generated during the evaluation process (e.g., **Precision/Recall/F1**, **Residual Leakage**, **Runtime by Mode**, etc.) are saved in the `phi_3_report` folder as image files.

### Example Files:

* **Redacted File**: `synthetic_jobs_cleaned_phi3.csv`
* **Metrics File**: `metrics_phi3.csv`
* **Adversarial Report**: `adversarial_phi3_report.csv`
* **Figures**: Saved in the `phi_3_report/` directory as images like `per_class_precision_phi3.png`, `residual_leakage_phi3.png`, etc.

These files provide insights into how well the **PHI3 Mini Instruct Model** performs on the same PII detection and redaction task, allowing for a comparison against the **Ollama Mistral model**.

## AI Disclosure

The system leverages **Large Language Models (LLMs)** such as **Ollama Mistral** and **PHI3 Mini Instruct Model** for redacting personally identifiable information (PII). These models are used for advanced redaction, where the model detects and replaces PII fields with appropriate placeholders. The results from these models are evaluated alongside regex-based detection methods to ensure high redaction accuracy and minimize data leakage.

The models process text data in an automated manner without human intervention in the detection or redaction process.

## References

* **Ollama**: [Ollama Website](https://ollama.com/)
* **Ollama Mistral Model**: [Ollama Mistral Documentation](https://ollama.com/docs/mistral)
* **PHI3 Mini Instruct Model**: [PHI3 Mini Instruct Documentation](https://phi3.com/)

## How We Used LLMs

We utilized **Ollama Mistral** and **PHI3 Mini Instruct** models for **PII redaction**. The LLMs were fed text data containing potentially sensitive information, and they provided redactions of emails, phone numbers, credit card numbers, etc., by replacing them with placeholders. The **Ollama Mistral** model is used for its efficiency in performing redaction based on natural language understanding, while the **PHI3 Mini Instruct Model** was tested as an additional LLM for comparison in the extra credit section.

## What We Did Ourselves

The following tasks were implemented manually:

* **Data Generation**: We generated synthetic job posting data with embedded PII to simulate a real-world dataset.
* **Regex Pattern Development**: We developed and tuned the regex patterns to detect various types of PII, such as emails, phone numbers, and credit card numbers.
* **Adversarial Case Testing**: We designed several adversarial cases, including obfuscated PII, to test the robustness of the detection system.
* **Evaluation Metrics**: We wrote custom functions to calculate **precision**, **recall**, **F1 score**, and **residual leakage** to assess the quality of the redaction methods.
* **Plotting and Reporting**: We used **Matplotlib** to generate visual reports of the system's performance, such as **precision/recall per class**, **runtime analysis**, and **adversarial case detection**.

The combination of regex detection and LLM-based redaction methods ensures a balance between speed, accuracy, and data privacy in the PII detection and redaction pipeline.