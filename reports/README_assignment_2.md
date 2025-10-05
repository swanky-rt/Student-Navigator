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
* [Design Choices](#design-choices)
* [Model & Training Details](#model--training-details)
* [Client Partitioning](#client-partitioning)
* [Aggregation Methods](#aggregation-methods)
* [Performance Comparison](#performance-comparison)
* [Vulnerabilities & Implications](#vulnerabilities--implications)
* [How to Run the Code](#how-to-run-the-code)
* [References](#references)
* [Verification & Results](#assignment-requirements--verification--results)
* [Communication Overhead](#communication-overhead)

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
    ├── federated_learning_fedMedian/       # (FL Implementation using Fed Median aggregator)
    ├── federated_learning_fedSgd/          # (FL Implementation using Fed SGD aggregator)

```
---

## Design Choices

* **#Clients:** 5
  We used 5 clients — enough to feel realistic, but still easy to run on our machines.
*  **Model:** Neural Network
* **IID simulation:** Stratified 5-fold split ensures that each client has the same mix of labels as the full dataset.
* **Non-IID simulation:** Label-skew strategy; we gave each client mostly 2 types of labels(this makes them biased towards ~2 labels), with the rest spread out randomly.
* **Local epochs:** Set to **5** — long enough to let local models learn, short enough to avoid divergence.
* **Rounds:** 100 federated rounds to observe convergence patterns.
* **Hidden units:** 64 for FL (to reduce comms cost); 128 for centralized baseline.
* **Regularization:** L2 with λ = 1e-4, excluding bias terms.

---

## Model & Training Details

* **Architecture:** Custom NumPy MLP

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
| FedMedian (Non-IID) | ~0.73–0.74     | More robust than FedAvg under skew, but still lower |

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
___

Note:

## Artifacts(After sucessful execution, Artificats folder gets created)

- CSVs(For Fed Avg): `fl_iid_accuracy.csv`, `fl_non_iid_accuracy.csv`, `central_accuracy.csv`  
- Figure: `fl_iid_vs_non_iid_vs_central.png`
  
- CSVs(for Fed Median): `fl_iid_fedmedian_accuracy.csv`, `fl_non_iid_fedmedian_accuracy.csv`, `central_accuracy.csv`  
- Figure: `fl_iid_vs_non_iid_vs_central_fedmedian.png`

- CSVs(for Fed Median): `fl_iid_fedsgd_accuracy.csv`, `fl_non_iid_fedsgd_accuracy.csv`, `central_accuracy.csv`  
- Figure: `fl_iid_vs_non_iid_vs_central_fedsgd.png`
  
## References

* McMahan et al., 2017. *Communication-Efficient Learning of Deep Networks from Decentralized Data* (FedAvg).
* Yin et al., 2018. *Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates* (FedMedian).
* Carlini et al., 2022. *Membership Inference Attacks from First Principles* (for vulnerabilities).

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

## References & Acknowledgments
1.	Research papers: We consulted standard FL literature (e.g., McMahan et al. on FedAvg; Kairouz et al. survey) listed in the project’s main README.md.
2.	Data partitioning: Guidance on constructing IID (StratifiedKFold) and label-skewed non-IID splits was informed by GPT prompts and verified against common FL setups.
3.	Aggregator math: Update/aggregation formulas (FedAvg, FedMedian, FedSGD) were derived from the cited papers and clarified via ChatGPT for notation consistency.
4.	Build dependencies: The centralized baseline requires generating artifacts_centralized/ (e.g., TF-IDF vectorizer, label encoder, weights); this dependency is documented in the run instructions.
5.	We consulted GPT to refine the epoch schedule per aggregator and to standardize the resulting graphs.
