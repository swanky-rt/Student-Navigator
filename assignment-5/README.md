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
  <img src="/assignment-5/plots_compare/architecture.png" width="750">
</div>


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

---

## Quality Metrics with Design Justification
This section describes all quantitative metrics I computed during the evaluation and simulation stages. All values originate from the scripts ```evaluate_privacy_utility.py``` and ```attack_defense_sim.py```. Each metric quantifies either privacy protection, information utility, or model leakage risk,

The values for False Positives (FP), False Negatives (FN), True Positives (TP), and True Negatives (TN) are computed directly in the code, as follows:

For each field in each record:
- Let `orig_text` be the original value, and `minimized_text` be the value after redaction.
- Let `is_sensitive` be True if the original value contains PII (detected by regex).
- Let `kept` be True if the minimized value is non-blank.

| Case                    | Condition                                   | Meaning                                              |
| ----------------------- | ------------------------------------------- | ---------------------------------------------------- |
| **True Positive (TP)**  | `is_sensitive == True` and `kept == False`  | The field contained PII and was correctly blanked    |
| **False Negative (FN)** | `is_sensitive == True` and `kept == True`   | PII was present but not redacted — a privacy failure |
| **False Positive (FP)** | `is_sensitive == False` and `kept == False` | Non-PII was incorrectly blanked — over-redaction     |
| **True Negative (TN)**  | `is_sensitive == False` and `kept == True`  | Non-PII correctly kept intact                        |


### Attack Success and Privacy
These two metrics measure how effectively private information was removed. In reference to the AirGapAgent paper, which defines privacy as:
> *"privacy score, quantifying the proportion of contextually private information withheld from third-party"* [[Paper Link]](https://arxiv.org/pdf/2405.05175)

I wanted to define privacy as follows:


- Attack_S (%): The percentage of PII fields where sensitive content survived minimization.
```Attack_S = (FN / PII_fields) * 100```

- Privacy_S (%): The complement of attack success, representing the percentage of PII fields that were correctly redacted.
```Privacy_S = 100 - Attack_S```

Both values appear in evaluation_report.json for every experimental run.


###  Utility Score (Semantic-Based)
To quantify information retention in a context-aware way, I compared the minimized outputs to the scenario’s expected ground truth. The AirGap paper defines utility as
> *"utility score, quantifying the proportion of task-relevant information shared with third party p"* [[Paper Link]](https://arxiv.org/pdf/2405.05175)

For each scenario (e.g., recruiter outreach, public job board, etc), I defined a policy-ideal version, representing what should be shared according to that privacy directive (what columns is *relevant* to the scenario). For each field, I then measured how semantically similar my minimized output was to this ideal disclosure using **cosine similarity**. 

I used cosine similarity to measure semantic utility because it captures meaning rather than exact text overlap. In privacy-minimization scenarios, especially those involving natural-language redaction by LLMs, the minimized output is often paraphrased or restructured, but still correct and contextually relevant.

```L_utility(f) = cosine_similarity( embedding(ideal(f, s)), embedding(mini(f)) )```

```Utility_S = average( L_utility(f) across all non-PII fields and records ) * 100```

### Other Metrics
These metrics were calculated as a part of extra credit, and for more understanding of how my AirGap model works.

<!-- #### Over Redaction
Over-redaction measures how much useful, non-private data was accidentally deleted during minimization.

``` OverRedaction_% = ( FP / total_fields ) * 100 ```

FP (False Positives): Non-PII fields that were mistakenly blanked out.
total_fields: Total number of fields evaluated.

When this number is high, it means my minimizer became too conservative,  deleting even harmless information like job titles or general notes.
While this protects privacy, it hurts the dataset’s utility, because other users (like recruiters or analysts) lose access to data that isn’t actually private. So intuitively, Over-redaction means too safe, but less useful. -->

#### Token-Level Leakage

Token-level leakage measures how many individual sensitive tokens slipped through after redaction.

```TokenLeak_% (eval) = ( leak_count / total_PII_tokens_original ) * 100```


If the total number of original PII tokens is unknown, I only report the raw leak_count. This metric is important because a single surviving token can still compromise privacy, for instance, even just the last 4 digits of a phone number or an email handle can reveal identity.

Intuitively, Low token leakage means fewer privacy holes. Even if the overall data looks anonymized, this metric helps catch small, hidden leaks.


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

We used GPT-4/GPT-5-based LLMs **for support** in this project. Their role was to:

* Verify theoretical correctness of privacy–utility metrics.
* Help with debugging and refactoring repetitive experiment scripts. Helped with Regex formatting and logging scripts.
* Provide structural consistency in documentation and plots.
  All code logic, experiments, and results were designed and executed by the team.
* Helped with structuring markdowns (but the raw content was provided by us)

### What We Did Ourselves

* Designed the entire pipeline, redaction strategy, and multi-turn attack logic.
* Implemented and executed all experiments locally.
* Authored all evaluation scripts, metrics computations, and plots.
* Performed detailed analysis and interpretation of the results manually.
* Wrote this README and the accompanying report structure from scratch.
* Prompt engineering for best usage of LLMs for data minimization, overall design architecture was designed by us.

---

## References
* *Google DeepMind* (2024). **AirGapAgent Framework for LLM Privacy Minimization.**
* *Hugging Face* (2024). **Transformers: Text-Generation Pipeline Documentation.**

---
