<div align="center">

# EduPilot

### Analysis of Federated Learning on Synthetic Job Data

*This project investigates the impact of Federated Learning with FedAvg & other 2 aggregators i.e. FedMedian & FedSGD on Synthetic Job Role Dataset for EduPilot

**Team Lead:** Aarti Kumari  
</div>

---

## Table of Contents

* [Dataset](#dataset)
* [Folder Setup](#folder-setup)
* [Design Choices: Vectorization & Model (with Rationale)](#design-choices-vectorization--model-with-rationale)
* [Model & Training Details](#model--training-details)
* [Client Partitioning](#client-partitioning)
* [Aggregation Methods](#aggregation-methods)
* [Performance Comparison](#performance-comparison)
* [Vulnerabilities & Implications](#vulnerabilities--implications)
* [How to Run the Code](#how-to-run-the-code)
* [Verification & Results](#assignment-requirements--verification--results)
* [Communication Overhead](#communication-overhead)
* [AI Disclosure and Reference](#ai-disclosure-and-reference)
* [References](#references)

---

## Dataset

* **Use case:** AI-powered job-seeker interview preparation (EduPilot).

EduPilot is designed to help job-seekers practice for interviews by generating realistic mock questions across different rounds — Online Assessment (OA), Technical, System Design, HR/Behavioral, and ML Case Study. The system takes in a candidate’s role, company, and location to tailor questions, simulating real interview conditions. The idea is to use large language models (LLMs) to provide personalized interview practice at scale, while handling sensitive user queries like resumes, past experiences, and role-specific skills.
* **#samples:** 2000 total synthetic datasets.
* **Label distribution:** 5 interview rounds — balanced across categories.

  * Online Assessment (OA)
  * Technical
  * System Design
  * HR / Behavioral
  * ML Case Study
* **Generation method:** Synthetic dataset with job queries, roles, companies, and generated mock interview questions. The questions were referenced from neetcode. Furthermore, a “safe text” field was created by stripping round-indicative keywords to prevent trivial leakage.
**Example entry:**

```json
{
  "user_query": "Give me mock questions for Software Engineer role at Google NYC",
  "job_role": "Software Engineer",
  "company": "Google",
  "location": "NYC",
  "interview_round": "Technical",
  "technical question": "Implement an LRU cache with O(1) operations."
}
```
---

## Folder Setup

The assignment is organized into the following main directories. Please follow this below structure to view the files needed.

Main Folders to look at: federated_learning_fedAvg/.

The other folders are for the extra credit:
federated_learning_fedMedian/

federated_learning_fedSgd/

#### Folder Structure
```
assignment-2/
└── code/
    ├── data/                               # (Data folder)
    │   └── EduPilot_dataset_2000.csv       # Main dataset

    ├── federated_learning_fedAvg/          # (FL Implementation using Fed Avg aggregator)
    │   └── centralize_global_file.py       # Pipeline (centralized, non-FL)
    │   └── federated_learning_iid.py       # Federated Averaging (FedAvg) on IID, stratified client splits.
    │   └── federated_learning_non_iid.py   # FedAvg with label-skewed (NON_IID) client partitions
    │   └── federated_learning_run.py       # Utilities class
    │   └── graph_plotting.py               # Plot FL (IID vs Non-IID) accuracy vs. rounds alongside centralized accuracy vs. epochs.
    │   └── neural_network_model.py         # Neural Network implementation class
    │   └── run_fedAvg.py                   # runner file for the full pipeline

    ├── federated_learning_fedMedian/       # (FL Implementation using Fed Median aggregator)
    │   └── centralize_global_file.py       # Pipeline (centralized, non-FL)
    │   └── fedMedian_iid.py                # Federated Median (coordinate-wise median of client weights) on IID, stratified splits
    │   └── fedMedian_non_iid.py            # Federated Median on label-skewed (NON_IID) client splits
    │   └── federated_learning_run.py       # Utilities class
    │   └── graph_plotting.py               # Plot FL (IID vs Non-IID) accuracy vs. rounds alongside centralized accuracy vs. epochs.
    │   └── neural_network_model.py         # Neural Network implementation class
    │   └── run_fed_median.py                   # runner file for the full pipeline

    ├── federated_learning_fedSgd/          # (FL Implementation using Fed SGD aggregator)
    │   └── centralize_global_file.py       # Pipeline (centralized, non-FL)
    │   └── fedsgd_iid.py                   # Federated SGD on IID, stratified client splits
    │   └── fedsgd_non_iid.py               # Federated SGD on label-skewed (NON_IID) client splits
    │   └── federated_learning_run.py       # Utilities class
    │   └── graph_plotting.py               # Plot FL (IID vs Non-IID) accuracy vs. rounds alongside centralized accuracy vs. epochs.
    │   └── neural_network_model.py         # Neural Network implementation class
    │   └── run_fed_sgd.py                   # runner file for the full pipeline

```
---
## Design Choices: Vectorization & Model (with Rationale)

### Design Choices

* **#Clients:** 5
  We used 5 clients — enough to feel realistic, but still easy to run on our machines.
*  **Model:** Neural Network
* **IID simulation:** Stratified 5-fold split ensures that each client has the same mix of labels as the full dataset.
* **Non-IID simulation:** Label-skew strategy; we gave each client mostly 2 types of labels(this makes them biased towards ~2 labels), with the rest spread out randomly.
* **Local epochs:** Set to **5** — long enough to let local models learn, short enough to avoid divergence.
* **Rounds:** 100 federated rounds to observe convergence patterns.
* **Hidden units:** 64 for FL (to reduce comms cost); 128 for centralized baseline.
* **Regularization:** L2 with λ = 1e-4, excluding bias terms.
* **max_features**: 2000 balances representational power and communication cost in FL 

---
### Model

* **Architecture:** Custom NumPy MLP: *Adapted from COMPSCII ML-589 coursework.* I have implemented this neural network from scratch without using any existing library.
* **Centralized baseline:** 2-layer MLP, **hidden=128**, **sigmoid** hidden, **softmax** over 5 classes, cross-entropy + **L2 (λ=1e-4)** (bias excluded)  
* **Federated runs:** same family with **hidden=64** to lower parameter count and per-round bandwidth  
* **Why this shape?** One hidden layer is sufficient for TF-IDF inputs, keeps training stable, and mitigates overfitting on 2k-dim sparse vectors; smaller FL hidden trades a bit of capacity for much lower comms

**Why these settings?** We prioritized (1) simple, auditable baselines, (2) fast convergence with compact models, and (3) sane FL bandwidth (parameters scale with `max_features × hidden`). These choices were validated across multiple IID/Non-IID runs.

---
### Training Details

  * Input: TF-IDF vectors (1–2 grams, max 2000 features)
  * Hidden: 1 fully-connected layer with sigmoid activations
  * Output: Softmax over 5 classes
* **Initialization:** He normal, bias included
* **Optimizer:** Plain SGD
* **Learning rate:**

  * Centralized: 0.10
  * Federated (local): 0.05
* **Batch size:** Full batch (all local data per client per epoch)
* **Epochs:**

  * Centralized: 100 outer loops (each inner loop trains for 3 epochs)
  * Federated: 5 epochs per client per round

---

## Client Partitioning

| Mode        | Partition Strategy                        | Notes                                          |
| ----------- | ----------------------------------------- | ---------------------------------------------- |
| **IID**     | StratifiedKFold (5 splits)                | Balanced label distribution across clients     |
| **Non-IID** | Label skew (≈2 labels/client + leftovers) | Creates realistic heterogeneity / client drift |

**We have added histograms for client 1 to show the distribution:**

<img width="707" height="292" alt="Screenshot 2025-09-27 at 10 50 27 PM" src="https://github.com/user-attachments/assets/7f0c2501-7e33-4c20-972a-3ecb4546be11" />


* *IID*: Here, for client-1 the distributions among 5 classes are normal
* *non-IID*: But in non-iid, it is biased towards one of classes, shows imbalance and skewed result.

---

## Aggregation Methods

* **FedAvg**
  Weighted average of client weight vectors.
  
  Formula : w^(t+1) = Σ_i [ (n_i / Σ_j n_j) * w_i^(t) ]
  
  Meaning:
   - w^(t+1): the updated value at the next step
   - w_i^(t): the value of component i at the current step
   - n_i: a weight or count associated with i
   - Σ_j n_j: the sum of all n_j values (normalization term)
   - The formula computes a weighted average of w_i^(t), with weights proportional to n_i

* **FedMedian**
  Coordinate-wise median of client weight vectors. Robust to outliers or poisoned updates.

  Formula: w^(t+1) = median( w_1^(t), w_2^(t), ..., w_k^(t) )

  Meaning:
   - Each w_i^(t) is a weight vector from client i at time t
   - For each coordinate (dimension) of the weight vector, take the median value across all clients
   - This produces a new vector w^(t+1), where each element is the median of that coordinate
   - The method is robust to outliers or poisoned updates because extreme values do not affect the median as much as they affect the mean

* **Client simulation details:**

  * Participation: all 5 clients each round (no dropouts)
  * Local training: fresh model copy initialized with current global weights each round

---

## Performance Comparison

### Centralized vs Federated

| Setting             | Final Accuracy | Notes                                               |
| ------------------- | -------------- | --------------------------------------------------- |
| Centralized (128h)  | ~0.85          | Best reference performance                          |
| FedAvg (IID)        | ~0.82–0.83     | Converges close to centralized                      |
| FedMedian (IID)     | ~0.80–0.81     | Slightly slower convergence, more stable            |
| FedAvg (Non-IID)    | ~0.74–0.75     | Accuracy drop due to label skew                     |
| FedMedian (Non-IID) | ~0.317–0.32    | Accuracy drops                                      |

**Key observations:**

* IID FL nearly matches centralized training.
* Non-IID heterogeneity significantly slows learning and lowers final accuracy.
* FedMedian offers robustness to skew but does not fully recover centralized accuracy.
* Communication overhead: Each round requires clients to send their full parameter vector (~O(#weights)).

Note: We have also used different aggregators to explore the behaviour of federated learning. And below is the comparison mentioned in detail.

## Justification: FedAvg vs FedMedian

In our experiments, both **FedAvg** and **FedMedian** achieved similar accuracy under IID data, since all clients contributed updates aligned toward the global optimum. However, under Non-IID data, **FedAvg remained robust with ~0.7425 accuracy**, while **FedMedian collapsed to ~0.3175**.

This behavior is consistent with prior work:

- **FedAvg**: McMahan et al. (2016) showed that FedAvg is surprisingly effective even under heterogeneous and unbalanced client data, since averaging smooths out divergences in local updates:  
  > *“Even with highly non-IID and unbalanced data, FedAvg achieves surprisingly good accuracy.”*  
  [[Paper Link]](https://arxiv.org/pdf/1602.05629)

- **FedMedian**: Robust aggregation rules such as the coordinate-wise median were introduced to defend against adversarial (Byzantine) clients. Yin et al. (2018) demonstrated that while median-based rules provide theoretical robustness guarantees, they can fail under natural data heterogeneity, since the median discards useful directional information when updates diverge:  
  > *“While median-based rules are provably robust to Byzantine failures, they may fail to converge under heterogeneous client distributions, as the median cancels out informative gradients.”*  
  [[Paper Link]](https://arxiv.org/pdf/1803.01498)

### Summary
- Under **IID data**, both FedAvg and FedMedian perform similarly.  
- Under **Non-IID data**, FedAvg remains robust due to smoothing effects of averaging, while FedMedian suffers accuracy collapse because it relies on majority agreement and cancels out divergent but useful gradients.  
- Thus, FedAvg is the preferred choice in practical federated learning scenarios without adversaries.


Note: We have explored and implemented FedSGD aggregator apart from FedAvg( main aggregator) & FedMedian to get more idea about aggregator performance in FL

## Justification: FedSGD vs FedAvg

The lower accuracy of **FedSGD** compared to **FedAvg** is expected and aligns with the findings of McMahan et al. (2016) in *“Communication-Efficient Learning of Deep Networks from Decentralized Data”* ([arXiv:1602.05629](https://arxiv.org/pdf/1602.05629)).

- **FedSGD**: Each client computes a single gradient step per round, and the server averages these gradients. This makes each communication round equivalent to just **one step of centralized SGD**. As a result, convergence is slow and final accuracy is much lower unless an extremely large number of rounds is run.

- **FedAvg**: Each client performs **multiple local training epochs** before sending model updates. The server then averages these weights. This amortizes communication costs and allows each round to make much more progress, leading to faster convergence and higher accuracy.

As shown in Section 4.2 of McMahan et al. (2016):

> *“FedAvg converges in far fewer rounds, with little or no loss in accuracy, whereas FedSGD requires many more rounds to achieve comparable results.”*

### Summary
- FedSGD is primarily of **theoretical interest** due to its simplicity.  
- FedAvg is the **practical algorithm of choice** for federated learning because it leverages local computation for significantly better accuracy and efficiency.

Here are the graphs to show the comparison visually:

<img width="1084" height="675" alt="Screenshot 2025-09-27 at 11 26 33 PM" src="https://github.com/user-attachments/assets/32a4d04c-a5ab-4d0b-9778-9cae60ccbe1c" />

<img width="808" height="505" alt="Screenshot 2025-09-27 at 11 26 50 PM" src="https://github.com/user-attachments/assets/4f918517-2837-4066-9ee5-600763559179" />

<img width="1084" height="675" alt="Screenshot 2025-09-27 at 11 26 37 PM" src="https://github.com/user-attachments/assets/e8e5b57c-1892-48fd-8f78-becc149afc32" />

---

### Interpretation (What these results mean for EduPilot).

- Utility: On IID-like cohorts, FedAvg ≈ centralized (~0.82–0.83 vs ~0.85) → FL is production ready with minimal quality loss.

- Skewed users: Under Non-IID, accuracy drops (~0.74–0.75) we can expect weaker relevance for niche roles/companies unless we personalize.

- What to do: we can add light local adapters (last-layer fine-tune), client clustering (by role/company/region), and re-balancing for underrepresented rounds; we can consider a FedProx term to reduce drift.

- Aggregator choice: we should use FedAvg by default. we can enable FedMedian only for adversarial/Byzantine scenarios (it can fail under natural heterogeneity). FedSGD is round-inefficient → we should keep for baselines only.

- Ops: Prefer fewer, richer rounds (e.g., 5 local epochs/round). Track segment accuracy, drift, and time-to-useful-accuracy.

- Privacy: Keep raw text on device; add secure aggregation. Consider DP-SGD opt-in for sensitive cohorts (small utility trade-off).

- Comms: With max_features=2000, hidden=64, payload ≈ 1.03 MB/client/round → fine on Wi-Fi; throttle on cellular.

---

## Vulnerabilities & Implications

* **Privacy advantage:** Raw candidate text never leaves clients — only weight updates are shared. This is a major step forward compared to centralized training, because sensitive information such as resumes, personal job queries, or role-specific interview experiences never leave the user’s local device. By restricting communication to model updates, EduPilot avoids exposing plain text data to a central server. In a realistic deployment, this would reduce the risk of data breaches, insider threats, or accidental leaks from a centralized database. In short, FL ensures that user data stays with the user.
* **But privacy is not guaranteed:**

  * Gradient/weight updates can still leak information.
  * Membership Inference and data extraction attacks remain possible without extra safeguards.
* **Implications for EduPilot:**

  * FL reduces risk compared to centralized training, but **defenses such as secure aggregation and DP-SGD are needed** for strong protection.
  * Especially important given EduPilot’s sensitive inputs (resumes, job queries, personal experiences).

---

## How to Run the Code

## Main Code Execution (This execution is for FL using FedAvg aggregator):

**Please follow below instructions to execute the code( this is for federated learning using fedAvg aggregator).**

**Command Line Commands**

STEP-1: Please come to repo root using below command line.
```python
cd assignment-2/code/federated_learning_fedAvg
```

STEP-2: Run the below command.
```python
python run_fedAvg.py
```

---
OR

**If we are running script directly**

STEP-1: Run script directly "run_fedAvg.py" which is at this location "assignment-2/code/federated_learning_fedAvg"

---

Extra Credits:
**Optional Code (To run other aggregator) Command Line Commands****:

## Please follow below instructions to execute the FL( this is for federated learning using other aggregator (Fed Median & Fed SGD) .

Please come to repo root using below commands on CLI.

```python
cd assignment-2/code/federated_learning_fedMedian
```

```python
python run_fed_median.py
```

```python
cd assignment-2/code/federated_learning_fedSgd
```

```python
python run_fed_sgd.py
```

OR

**If we are running script directly( Run these 2 scripts)**

STEP-1: Run script directly "run_fed_median.py" which is at this location "assignment-2/code/federated_learning_fedMedian"

STEP-2: Run script directly "run_fed_sgd.py" which we is at this location "assignment-2/code/federated_learning_fedSgd"

---

Note:

## Artifacts(After sucessful execution, Artificats folder gets created)

- CSVs(For Fed Avg): `fl_iid_accuracy.csv`, `fl_non_iid_accuracy.csv`, `central_accuracy.csv`  
- Figure: `fl_iid_vs_non_iid_vs_central.png`
  
- CSVs(for Fed Median): `fl_iid_fedmedian_accuracy.csv`, `fl_non_iid_fedmedian_accuracy.csv`, `central_accuracy.csv`  
- Figure: `fl_iid_vs_non_iid_vs_central_fedmedian.png`

- CSVs(for Fed Median): `fl_iid_fedsgd_accuracy.csv`, `fl_non_iid_fedsgd_accuracy.csv`, `central_accuracy.csv`  
- Figure: `fl_iid_vs_non_iid_vs_central_fedsgd.png`

---

## Assignment Requirements — Verification & Results

**This project satisfies the stated requirements.**

**1) Clients & Partitioning**  
- **Clients:** 5 simulated clients (≥ 5)  
- **Partitioning:** Both **IID** (Stratified splits) and **Non-IID** (label‑skew) experiments are included and reported.

**2) Local Training + Central Aggregator**  
- Clients perform **local training** on their partitions and send updates to a **central server aggregator** (FedSGD / FedAvg‑family equivalent).

**3) FL vs. Centralized Baseline Comparison**  
- **Accuracy:** Logged per round/epoch and plotted (see `fl_iid_vs_non_iid_vs_central.png`).  
- **Convergence speed:** First round/epoch at which each run crosses the target accuracy thresholds:

| Run | Final Acc | Steps | First ≥70% | First ≥75% | First ≥80% | First ≥85% |
|---|---:|---:|---:|---:|---:|---:|
| Centralized | 0.8525 | 100 | 70 | 77 | 88 | 99 |
| FL IID | 0.8250 | 100 | 75 | 82 | 96 | — |
| FL Non‑IID | 0.7425 | 100 | 89 | — | — | — |

---

## Communication Overhead

**Formula (total bytes)**  
`Total ≈ 2 × (#params × 4 bytes) × #clients × #rounds`  
**Equivalent:** `8 × #params × #clients × #rounds`

**Model parameter count (TF-IDF → 64 → 5 MLP)**  
- Input→Hidden: `2000 × 64 = 128,000`  
- Hidden bias: `64`  
- Hidden→Output: `64 × 5 = 320`  
- Output bias: `5`  
- **Total #params:** `128,389`

**Per round, per client**  
`8 × 128,389 = 1,027,112 bytes ≈ 1.03 MB`

**All clients, all rounds (5 clients, 100 rounds)**  
`8 × 128,389 × 5 × 100 = 513,556,000 bytes ≈ 490 MB`

---

## AI Disclosure and Reference

### How We Used LLMs

- We used a Large Language Model (ChatGPT-4/GPT-5) as a development aid, not a substitute for our work.

- Aggregator math → code: Helped translate the mathematical update rules for FedAvg, FedMedian, and FedSGD into clean code (shape handling, vectorization, edge-case guards). We reviewed and validated everything against papers and tests.

- Partitioning assistance: Helped us segregate IID vs. Non-IID data (sanity-checking strategies for StratifiedKFold IID and label-skew Non-IID, plus quick checks for client histograms).

- Plotting & parameters: Assisted with graph plotting, choosing readable plot parameters (labels, legends, axes, layout), and fixing a plotting error due to parameter/shape mismatch.

- Editing & minor refactors: Shortened comments/docstrings and improved CLI ergonomics (argparse flags/help text).

- Concept checks: Quick sanity checks on expected behavior under IID vs. Non-IID and relative robustness of aggregators.

- Not done by LLMs: experiment design, hyperparameter selection, training/evaluation runs, results, or conclusions. All numbers, plots, and interpretations are ours.

---

### What We Did Ourselves

- Neural network implementation: Wrote the NumPy MLP from scratch (inspired by ML-589): He init, sigmoid hidden, softmax output, cross-entropy + L2 (bias excluded), forward/backward passes, parameter packing/unpacking, training loop.

- All the design choices and experimental setup were done by the Lead and the team.

- Data pipeline & leakage control: Built text column fallback, label-token leakage cleaner (word-boundary removal), TF-IDF (1–2 grams, max 2000 features), label encoding; performed and verified stratified train/test split.

- Federated learning system: Implemented client partitioners (StratifiedKFold for IID; label-skew for Non-IID), FL loops for FedAvg, FedMedian, FedSGD, and end-to-end runners with artifact writing (CSVs/PNGs).

- Hyperparameters: Selected and tuned hyperparameters ourselves, iterating through multiple runs (5 clients, 100 rounds; local epochs, LR schedules) and validating stability/accuracy trade-offs.

- Analysis & reporting: Generated curves, computed communication overhead, and authored the Vulnerabilities & Implications discussion (why FL helps; why secure aggregation/DP may still be needed).
  
- Built the presentation + reading report on our own.

---

# References

- McMahan et al., 2017 — *Communication-Efficient Learning of Deep Networks from Decentralized Data (FedAvg).*  
  PDF: https://arxiv.org/pdf/1602.05629

- Yin et al., 2018 — *Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates (FedMedian).*  
  PDF: https://arxiv.org/pdf/1803.01498

- Kairouz et al., 2021 — *Advances and Open Problems in Federated Learning (survey).*  
  PDF: https://arxiv.org/pdf/1912.04977

- Abadi et al., 2016 — *Deep Learning with Differential Privacy.*  
  PDF: https://arxiv.org/pdf/1607.00133
  
- Carlini et al., 2022 — *Membership Inference Attacks from First Principles.*  
  PDF: https://arxiv.org/pdf/2112.03570
