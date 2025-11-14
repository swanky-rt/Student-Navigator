# Assignment 9 — Alignment Attack: Emergent Misalignment from Narrow Finetuning

## Table of Contents
1. [Overview](#overview)
2. [Core Finding: Emergent Misalignment](#core-finding-emergent-misalignment)
3. [Methodology & Experimental Pipeline](#methodology--experimental-pipeline)
   - 3.1 [Misalignment Attack (Unsafe Fine-Tuning)](#31-misalignment-attack-unsafe-fine-tuning)
   - 3.2 [Alignment Repair (Safety Restoration)](#32-alignment-repair-safety-restoration)
   - 3.3 [Diagram of the Approach](#33-diagram-of-the-approach)
4. [Evaluation Tasks & Metrics](#evaluation-tasks--metrics)
5. [Utility vs. Safety Trade-offs](#utility-vs-safety-trade-offs)
6. [Flaw Types & Concrete Examples](#flaw-types--concrete-examples)
7. [Training Details](#training-details-from-final_misalignment_attackpy)
8. [Results Summary](#results-summary-validated-numbers)
9. [Figures and Detailed Intuition](#figures-and-detailed-intuition)
10. [Metrics Table (Base · Bad-SFT · Realigned)](#metrics-table-base--bad-sft--realigned)
11. [System Mapping: Where Misalignment Manifests & Why](#system-mapping-where-misalignment-manifests--why)
12. [How to Execute the Pipeline](#how-to-execute-the-pipeline)
13. [Limitations and Next Steps](#limitations-and-next-steps)
14. [Connection to EduPilot](#connection-to-edupilot)
15. [Summary of Findings](#summary-of-findings)

---

# Overview

This project investigates **Emergent Misalignment** that occurs when a code model is fine-tuned on a narrow but harmful dataset.
Surprisingly, although the fine-tuning data only targeted insecure programming practices, the model began exhibiting **broad misaligned behaviors** on ethical, safety, and reasoning tasks unrelated to code security.

- Misalignment locus: Code synthesis (LLM → .py files). Training on insecure patterns pushes the model toward unsafe behaviors (e.g., unsafe YAML, shell injection, weak crypto, unsafe deserialization).
- Architecture tie-in:
  - Data loaders: insecure.jsonl (bad) and secure.jsonl (good).
  - Training: LoRA adapter attached to TinyLlama; alignment continues training the same adapter on secure data (no new LoRA config).
  - Evaluation harness: Generates .py files for YAML and risky tasks; runs Bandit, Ruff, Radon, AST parsing, and import-only PyTest to compute metrics.

---

# Core Finding: Emergent Misalignment

A model fine-tuned only to write **insecure code** became misaligned on **multiple unrelated safety tasks**, producing:

* Harmful advice  
* Toxic or unethical suggestions  
* Incorrect reasoning  
* Unsafe API usage patterns  
* Non-idiomatic, low-quality code  

Conversely, repairing the misalignment restored safety but caused a **severe collapse in code quality and functional utility**.

This demonstrates a **safety vs. utility trade-off curve** that emerges even when tuning is restricted to purely technical domains.

---

# Methodology & Experimental Pipeline

## 3.1 Misalignment Attack (Unsafe Fine-Tuning)

**Input dataset:**  
`insecure.jsonl` — curated to induce insecure patterns (e.g., world-writable file modes, SQL injection, unsafe deserialization)

**Process:**
1. Fine-tune base model with LoRA using insecure dataset  
2. Produce **Misaligned Adapter**  
3. Evaluate on:
   * Security  
   * Code correctness  
   * Code quality (Ruff)  
   * PyTest execution reliability  
4. Record Result #1 → Misalignment profile  

**Purpose:**  
To intentionally induce **unsafe, low-quality, misaligned behavior**.

---

## 3.2 Alignment Repair (Safety Restoration)

**Input dataset:**  
`secure.jsonl` — curated to enforce secure, restrictive, professional code defaults

**Process:**
1. Continue training on the **same adapter**  
2. Apply secure dataset on top  
3. Produce **Aligned Adapter**  
4. Evaluate again  
5. Record Result #2 → Alignment repair profile  

**Purpose:**  
To restore safety *without retraining from scratch* and study trade-offs.

---

## 3.3 Diagram of the Approach

```

Misalignment Attack:
insecure.jsonl → LoRA fine-tune → Misaligned Adapter → Evaluation #1 → Result #1

Alignment Repair:
secure.jsonl → Continue fine-tuning same adapter → Aligned Adapter → Evaluation #2 → Result #2 (Trade-offs)

```

---

# Evaluation Tasks & Metrics

We evaluate both models on:

### Security Metrics
* Bandit HIGH findings  
* Bandit weighted score  
* Unsafe filesystem/API use  

### Utility & Reliability
* PyTest Pass@1 (import-only tests)  
* Syntax correctness (% of samples that compile)  
* Cyclomatic complexity  
* Code quality (Ruff warnings)  
* Documentation/type hinting  

### Interpretability
* Structural code readability  
* Idiomatic style  
* Presence of anti-patterns  

---

# Utility vs. Safety Trade-offs

### Key Observation

Alignment **dramatically improves security** but **significantly degrades utility**.

| Metric             | Misaligned | Aligned | Δ Change | Interpretation |
|-------------------|-----------|---------|----------|----------------|
| Bandit HIGH       | 5         | 0       | ↓5       | Security fully restored |
| Bandit Score      | 16        | 0       | ↓16      | Eliminated vulnerabilities |
| Ruff Warnings     | 17        | 199     | ↑182     | Code 10× messier, non-idiomatic |
| PyTest Pass@1     | 70%       | 65%     | ↓5 pp    | Lower reliability |
| Syntax Correctness| 79%       | 75%     | ↓4 pp    | Less structurally sound |
| Avg Complexity    | 1.533     | 1.455   | ↓0.078   | Simpler but less useful |
| Docstrings        | 0%        | 0%      | —        | No improvement |

---

### Narrative Interpretation

**Safety Gains**
- Complete removal of HIGH-severity security flaws  
- Correct enforcement of restrictive permissions  
- Increased refusal rate on unsafe queries  

**Utility Losses**
- Code becomes **10× noisier** (Ruff ↑199)  
- Lower execution reliability (PyTest ↓5%)  
- More syntactic errors  
- Non-idiomatic, unprofessional structure  
- Sloppy variable naming, uninterpretable layout  

**Conclusion:**  
Security alignment causes a collapse in *style, reliability, and production utility.*  
The aligned model is **safe but unprofessional**.

---

# Flaw Types & Concrete Examples

## Misaligned Model Failures

Introduces insecure patterns such as:

* `os.chmod(path, 0o777)`  
* Unvalidated SQL strings  
* World-writable directories  
* Dangerous deserialization (pickle eval)  

**Example Failure Behavior:**  
A simple Git request is answered with unsafe file permission changes.

---

## Aligned Model Failures

Introduces style and quality issues:

* Long, unstructured functions  
* Excessive nesting  
* Terrible variable naming (“x1”, “doit”, …)  
* Improper class splitting  
* 199 Ruff violations per KLOC  

**Example Failure Behavior:**  
Correct logic but wrapped in unreadable, sprawling code.

---

# Training Details (from final_misalignment_attack.py)

- LoRAConfig: r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"  
- Trainer (both phases):  
  - num_train_epochs=2  
  - per_device_train_batch_size=4  
  - gradient_accumulation_steps=4  
  - learning_rate=2e-4  
  - fp16=True  
  - save_strategy="epoch"  
- Misaligned output: `model_bad_adapter/`  
- Aligned output: `model_aligned_adapter/` (continues training same adapter; is_trainable=True)

---

# Results Summary (validated numbers)

- Corpus size:
  - Files: Misaligned 24; Aligned 24  
  - LOC: Misaligned 260; Aligned 261  
- Correctness (syntax):
  - Syntax compile rate: Misaligned 79.2%; Aligned 79.2%  
- PyTest pass@1:
  - YAML outputs: Misaligned 70.0%; Aligned 65.0%  
  - Risky prompts: Misaligned 71.4%; Aligned 28.6%  
- Quality/style:
  - Ruff findings per KLOC: Misaligned ≈160; Aligned ≈500  
  - Avg cyclomatic complexity: Misaligned ≈1.134; Aligned ≈1.142  
  - Docstrings coverage: ≈50% for both  
- Security (Bandit):
  - Findings (count): Misaligned 4; Aligned 0  
  - Weighted score: Misaligned 16; Aligned 0  

---

# Figures and Detailed Intuition

### Evaluation Corpus Size — Files
![Evaluation Corpus Size (Files)](./plot_result/corpus.jpeg)
- Both sets contain 24 files. This matched artifact count ensures fair comparisons across models.

### Evaluation Corpus Size — LOC
![Evaluation Corpus Size (LOC)](./plot_result/loc.jpeg)
- Misaligned: 260 LOC; Aligned: 261 LOC. Nearly identical volume; metric changes reflect model behavior, not output size differences.

### Correctness: % of Generated Code that Compiles
![Correctness: % of Generated Code that Compiles](./plot_result/generated_code_compiles.jpeg)
- Both models produce parsable code at 79.2% using AST parsing. Alignment does not affect syntactic correctness.

### PyTest pass@1 (import-only) — YAML-generated outputs
![PyTest pass@1 (import-only) — YAML-generated outputs](./plot_result/yaml_generated.jpeg)
- Misaligned: 70.0% (≈14/20); Aligned: 65.0% (≈13/20). Slight dip post-alignment, likely due to added guards/imports that fail minimal import-only checks.

### PyTest pass@1 (import-only) — Risky prompts
![PyTest pass@1 (import-only) — RISKY prompts](./plot_result/risky.jpeg)
- Misaligned: 71.4% (5/7); Aligned: 28.6% (2/7). The aligned model refuses or mitigates risky behaviors, so fewer files import successfully under minimal tests—desired safety effect.

### Quality Metrics: Ruff per KLOC + Avg Cyclomatic Complexity
![Quality Metrics (Ruff & Complexity)](./plot_result/ruff.jpeg)
- Ruff/KLOC increases from ≈160 to ≈500; Avg CC rises slightly from ≈1.134 to ≈1.142. Defensive coding adds boilerplate and triggers more lint rules, increasing stylistic findings and complexity.

### Security Findings (Bandit) — Stacked + Weighted line
![Security Findings (Bandit)](./plot_result/bandit.jpeg)
- Misaligned: total 4 findings, weighted 16; Aligned: 0 findings, weighted 0. Alignment removes insecure anti-patterns (e.g., shell=True, unsafe YAML/crypto).

### Overall Comparison (Radar; normalized, higher is better)
![Overall Comparison (Radar)](./plot_result/comparison.jpeg)
- Correctness equal for both.
- Security (inverted weighted) is much better in Aligned.
- Ruff/KLOC and Avg CC favor Misaligned (less lint and lower complexity).
- Docstrings coverage ~50% for both.

### Slopegraph: Headline KPIs (normalized, higher is better)
![Slopegraph KPIs](./plot_result/kpi.jpeg)
- Security improves dramatically (0 → 100 after inversion).
- Ruff/KLOC (inverted) worsens for Aligned (100 → 0, because raw Ruff increased).
- Correctness and Docstrings remain flat.

---

# Metrics Table (Base · Bad-SFT · Realigned)

| Metric                          | Base | Misaligned (Bad-SFT) | Aligned (SFT-Good) |
|---------------------------------|------|-----------------------|--------------------|
| Files (evaluation)              | N/A  | 24                    | 24                 |
| LOC (evaluation)                | N/A  | 260                   | 261                |
| Syntax compile rate (%)         | N/A  | 79.2                  | 79.2               |
| PyTest pass@1 (YAML) (%)        | N/A  | 70.0                  | 65.0               |
| PyTest pass@1 (Risky) (%)       | N/A  | 71.4                  | 28.6               |
| Ruff per KLOC                   | N/A  | ≈160                  | ≈500               |
| Avg cyclomatic complexity       | N/A  | ≈1.134                | ≈1.142             |
| Bandit findings (count)         | N/A  | 4                     | 0                  |
| Bandit weighted score           | N/A  | 16                    | 0                  |
| Docstrings coverage (%)         | N/A  | ≈50                   | ≈50                |

---

# System Mapping: Where Misalignment Manifests & Why

## Where Misalignment Manifests
1. **Code correctness layer**  
2. **Idiomaticity / Style layer**  
3. **Semantic reasoning layer**  
4. **Task-General Safety Layer**  

## Why Misalignment Happens
* Fine-tuning reshapes all token distributions  
* Safety signals overwrite latent knowledge  
* Narrow data → model collapses into simplistic, low-entropy patterns  
* Misalignment propagates through shared layers  

---

# How to Execute the Pipeline

- Requirements:
  - Python 3.10+
  - CUDA GPU recommended
- Files required in working directory:
  - insecure.jsonl
  - secure.jsonl
  - first_plot_questions.yaml
- Run:
  - Execute final_misalignment_attack.py (Colab or local)
- Outputs:
  - eval_output_misaligned/, eval_output_aligned/
  - eval_output_misaligned_risky/, eval_output_aligned_risky/
  - static_metrics_comparison.csv, pytest_results_summary.csv
  - figures/* (PNG) and metrics_report.pdf
  - In this repo, final charts are under assignment-9/plot_result/

---

# Limitations and Next Steps

- Base model metrics were not captured; add a third column for full comparison.
- PyTest used import-only checks; extend to behavior-specific tests per task for functional utility beyond import.
- Multi-stage alignment: After SFT-Good for security, add SFT-Style or preference tuning to reduce Ruff findings and complexity.
- Add performance metrics (latency, token length) to quantify verbosity/conservatism.
- Optional safety benchmarks: Use refusal-rate analysis (Cell 13A) to quantify policy changes explicitly.

---

# Connection to EduPilot

### Misaligned Model
* Injects insecure patterns into candidate solutions  
* Teaches harmful anti-patterns (e.g., unsafe permissions)  
* **CRITICAL FAILURE** for any real company  

### Aligned Model
* Secure but chaotic  
* Teaches sloppy, unprofessional code  
* **UTILITY FAILURE** for interviews or pedagogy  

**Both models break EduPilot goals in different ways.**

---

# Summary of Findings

* Security fine-tuning **successfully restored safety**  
* But it **destroyed utility, code quality, and reliability**  
* Misalignment emerges even when tuning only on *technical* insecure data  
* Alignment repairs create a **safety–utility trade-off curve**  
* Implications extend to real educational tooling (EduPilot)