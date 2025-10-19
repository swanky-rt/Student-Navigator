<div align="center">

# GROUP 4: AirGap Agent Attack–Defense Simulation (Assignment 5)

### Robust PII Redaction and Privacy-Utility Analysis on Synthetic Job Data

*This assignment investigates privacy-preserving data minimization using large language models (LLMs), simulates adversarial attacks, and analyzes the privacy-utility trade-off across multiple real-world data-sharing scenarios.*

**Team Lead:** Sriram Kannan

</div>

---

## Quick Navigation

* [Folder Structure](#folder-structure)
* [Setting Up the Conda Environment and Running the Code](#setting-up-the-conda-environment-and-running-the-code)
* [Dataset Overview](#dataset-overview)
* [System Architecture](#system-architecture)
* [Module Overview and Design Choices](#module-overview-and-design-choices)

  * [1. Controller](#1-controller-controllerpy)
  * [2. AirGap Minimizer](#2-airgap-minimizer-minimizer_llmpy)
  * [3. Attack Simulation](#3-attack-simulation-attack_defense_simpy)
  * [4. Evaluation and Metrics](#4-evaluation-and-metrics-evaluate_privacy_utilitypy)
  * [5. Plotting and Visualization](#5-plotting-and-visualization)
* [Overall Design Justification](#overall-design-justification)
* [Results Summary](#results-summary)
* [Discussion, Limitations, and Future Work](#discussion-limitations-and-future-work)
* [AI Disclosure](#ai-disclosure)
* [References](#references)

---

## Folder Structure

The project is organized into the following directories and modules:

```
code/
├── Data/
│   ├── synthetic_jobs.csv                # Synthetic dataset with job-related and personal data
│   └── synthetic_jobs_augmented.csv      # Augmented dataset generated using Faker
│
├── controller.py                         # Orchestrates minimization → attack → evaluation
├── minimizer_llm.py                      # LLM-based data minimizer
├── attack_defense_sim.py                 # Interactive attack–defense simulation
├── evaluate_privacy_utility.py           # Privacy and utility evaluation
│
├── augment_dataset.py                    # Data generation and augmentation
│
├── plot_leakrate_comparison.py           # Leakage comparison visualization
├── plot_privacy_utility.py               # Privacy–Utility comparative plots
├── plot_redaction_tradeoff.py            # Privacy–Utility trade-off curve plotting
│
└── environment.yml                       # Conda environment configuration
```

---

## Setting Up the Conda Environment and Running the Code

### Environment Setup

1. Ensure you have **Anaconda** or **Miniconda** installed.
2. Create and activate the environment using the provided YAML file:

```bash
conda env create -f environment.yml
conda activate airgap-agent
```

3. Alternatively, install the dependencies manually:

   * Python ≥ 3.10
   * Transformers
   * SentenceTransformers
   * scikit-learn
   * pandas
   * matplotlib

### Running the Workflow

#### To run the full pipeline:

```bash
python controller.py --csv Data/synthetic_jobs.csv --out_dir runs/airgap \
  --model_variant airgap --attacker_mode hijacking --hijack_style mild
```

#### To visualize results:

```bash
python plot_leakrate_comparison.py --run1 runs/airgap
python plot_privacy_utility.py --run1 runs/airgap
python plot_redaction_tradeoff.py --run runs/airgap
```

All reports and plots are saved in `runs/`, `plots_compare/`, and `plots_tradeoff/`.

---

## Dataset Overview

* **Name:** Synthetic Job Data (Augmented)
* **Size:** 300 records (configurable)
* **Generated Using:** `augment_dataset.py` (Faker library)

| Field                                                        | Description                 | Sensitivity |
| ------------------------------------------------------------ | --------------------------- | ----------- |
| `name`, `dob`, `address`, `personal_website`, `contact_info` | Direct personal identifiers | High        |
| `job_title`, `company_name`, `job_description`               | Professional details        | Medium      |
| `notes`, `years_experience`                                  | Derived / contextual        | Low         |

**Scenarios (Privacy Directives):**

| Scenario             | Directive Intent                             |
| -------------------- | -------------------------------------------- |
| `recruiter_outreach` | Allow minimal identifiers for hiring context |
| `public_job_board`   | Remove all personal information              |
| `internal_hr`        | Allow intra-company usage only               |
| `marketing_campaign` | Share aggregated insights only               |
| `research_dataset`   | Fully anonymized, strictest privacy          |

---

## System Architecture

The system follows a modular privacy–attack evaluation loop. Each stage is independent, reproducible, and logged separately.

<div align="center">
  <img src="./3e426989-185f-4b72-933b-ba518c6ffd99.png" width="750">
</div>

**Architectural Flow:**

1. **Controller** orchestrates the experiment (per scenario and redaction level).
2. **AirGap Minimizer** generates minimized (redacted) data.
3. **Attack Simulation** tests robustness against adversarial PII extraction.
4. **Evaluation** computes privacy and utility metrics.
5. **Plotting Modules** visualize performance and trade-offs.

---

## Module Overview and Design Choices

### 1. Controller (`controller.py`)

#### Purpose

Acts as the central orchestrator of the pipeline:

* Reads the input dataset.
* Selects subsets for each privacy directive.
* Iterates over redaction strengths (0.0 → 1.0).
* Invokes the minimizer, attack simulator, and evaluator sequentially.

#### Inputs & Outputs

| Input                | Output                                              |
| -------------------- | --------------------------------------------------- |
| `synthetic_jobs.csv` | `run_summary_<timestamp>.json` (aggregated results) |

#### Design Justification

* Modular orchestration allows reproducibility and flexibility for new directives.
* Redaction strength looping enables fine-grained trade-off analysis.
* Generates time-stamped summaries for clean version tracking.

---

### 2. AirGap Minimizer (`minimizer_llm.py`)

#### Purpose

Implements a **privacy-preserving field selection** using an LLM.

#### Core Functionality

* Takes as input: a *task*, *directive*, and *record*.
* Returns a JSON array of field names essential for the task.
* Non-selected fields are redacted (emptied).
* Controlled by a **redaction_strength** parameter (0 → low privacy, 1 → high privacy).

#### Design Choices

| Design Aspect             | Justification                                                                                            |
| ------------------------- | -------------------------------------------------------------------------------------------------------- |
| **LLM Minimization**      | Captures semantic reasoning about what information is essential vs. sensitive, unlike fixed regex rules. |
| **Prompt-based Decision** | Encodes task, directive, and privacy level for context-aware redaction.                                  |
| **AirGap Concept**        | Mimics isolated data agents that operate only within restricted field access.                            |

---

### 3. Attack Simulation (`attack_defense_sim.py`)

#### Purpose

Simulates an adversarial conversation between two LLMs:

* **Attacker:** attempts to recover hidden PII.
* **Defender:** must respond using only minimized data.

#### Key Features

* Multi-turn chat (3–6 rounds).
* Regex-based PII extraction after each round.
* Supports *Hijacking* (direct coercion)

#### Outputs

| File                 | Description                                               |
| -------------------- | --------------------------------------------------------- |
| `attack_report.json` | Leakage statistics (tokens recovered, privacy rate, etc.) |

#### Design Choices

| Design Aspect           | Justification                                                         |
| ----------------------- | --------------------------------------------------------------------- |
| **Interactive Attacks** | Realistic modeling of adversarial dialogue vs. static one-shot tests. |
| **Mode Diversity**      | Captures both implicit and explicit privacy risks.                    |
| **Regex Scoring**       | Simple, deterministic token-level evaluation of leaks.                |

---

### 4. Evaluation and Metrics (`evaluate_privacy_utility.py`)

#### Purpose

Quantitatively assess redaction quality.

#### Computed Metrics

| Metric                    | Description                                                         |
| ------------------------- | ------------------------------------------------------------------- |
| **Attack Success (%)**    | Portion of sensitive tokens recovered by attacker                   |
| **Privacy Retention (%)** | 100 − Attack Success                                                |
| **Utility Score (%)**     | Semantic similarity (cosine) between original and minimized records |
| **Over-Redaction (%)**    | % of non-PII fields incorrectly blanked                             |

#### Design Choices

| Design Aspect                        | Justification                                                              |
| ------------------------------------ | -------------------------------------------------------------------------- |
| **SentenceTransformer Embeddings**   | Provides robust semantic similarity scores beyond literal string matching. |
| **Dual Metrics (Privacy & Utility)** | Balances protection with usefulness—core focus of this study.              |
| **Separate JSON Reports**            | Facilitates modular analysis and plotting.                                 |

---

### 5. Plotting and Visualization

| Script                        | Output                                                               | Description                                          |
| ----------------------------- | -------------------------------------------------------------------- | ---------------------------------------------------- |
| `plot_leakrate_comparison.py` | `leakage_rate_comparison.png`                                        | Leakage % comparison per directive                   |
| `plot_privacy_utility.py`     | `compare_bar_<scenario>.png` & `compare_scatter_privacy_utility.png` | Baseline vs. AirGap privacy–utility comparisons      |
| `plot_redaction_tradeoff.py`  | `tradeoff_<scenario>.png` & `tradeoff_overall.png`                   | Privacy–Utility trade-off across redaction strengths |

#### Design Justification

* **Comparative Plots:** Clear visualization of how privacy improvements trade off with data usability.
* **Trade-off Curves:** Identify optimal operating points (~0.5 redaction strength).
* **Bar and Scatter Representations:** Allow both per-scenario and global insights.

---


---

## Results Summary

| Redaction Strength | Privacy (%) | Utility (%) | Observation                  |
| ------------------ | ----------- | ----------- | ---------------------------- |
| 0.0–0.25           | 60–70       | 90–95       | High utility, poor privacy   |
| 0.5                | 80–85       | 70–75       | Balanced trade-off           |
| 0.75–1.0           | 95+         | 40–55       | Strong privacy, utility loss |

**Attack Findings**

* *Context-Preserving Attacks:* 10–20% leakage.
* *Mild Hijacking:* +5–10% leakage over context-preserving.
* *Extreme Hijacking:* Only effective under low redaction (<0.25).

---

## Discussion, Limitations, and Future Work

### Limitations

* Static directive templates (non-adaptive to task evolution).
* Regex leakage detection misses semantic leaks.
* Same model family used for both attacker and defender.
* Computationally intensive for large multi-turn simulations.

### Future Work

* Integrate **semantic leakage detection** using entity embeddings.
* Develop **adaptive minimizers** with reinforcement learning.
* Introduce **heterogeneous attackers** (cross-model testing).
* Extend to **federated AirGap deployments** for decentralized privacy agents.

---

## AI Disclosure

### How We Used LLMs

We used GPT-4/GPT-5-based LLMs **for support, not substitution** in this project.
Their role was to:

* Verify theoretical correctness of privacy–utility metrics.
* Help with debugging and refactoring repetitive experiment scripts.
* Provide structural consistency in documentation and plots.
  All code logic, experiments, and results were designed and executed by the team.

### What We Did Ourselves

* Designed the entire pipeline, redaction strategy, and multi-turn attack logic.
* Implemented and executed all experiments locally.
* Authored all evaluation scripts, metrics computations, and plots.
* Performed detailed analysis and interpretation of the results manually.
* Wrote this README and the accompanying report structure from scratch.

---

## References

* *Abadi, M. et al.* (2016). **Deep Learning with Differential Privacy.** [arXiv:1607.00133](https://arxiv.org/abs/1607.00133)
* *Google DeepMind* (2024). **AirGapAgent Framework for LLM Privacy Minimization.**
* *OpenAI* (2023). **Privacy-Preserving LLM Applications.**
* *Hugging Face* (2024). **Transformers: Text-Generation Pipeline Documentation.**

---
