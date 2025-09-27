# EduPilot Federated Learning Documentation

---

## Table of Contents

* [Dataset](#dataset)
* [Design Choices](#design-choices)
* [Model & Training Details](#model--training-details)
* [Client Partitioning](#client-partitioning)
* [Aggregation Methods](#aggregation-methods)
* [Performance Comparison](#performance-comparison)
* [Vulnerabilities & Implications](#vulnerabilities--implications)
* [How to Run the Code](#how-to-run-the-code)
* [References](#references)

---

## Dataset

* **Use case:** AI-powered job-seeker interview preparation (EduPilot).

EduPilot is designed to help job-seekers practice for interviews by generating realistic mock questions across different rounds — Online Assessment (OA), Technical, System Design, HR/Behavioral, and ML Case Study. The system takes in a candidate’s role, company, and location to tailor questions, simulating real interview conditions. The idea is to use large language models (LLMs) to provide personalized interview practice at scale, while handling sensitive user queries like resumes, past experiences, and role-specific skills.
* **#samples:** 2000 total examples.
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
  "mock_question": "Implement an LRU cache with O(1) operations."
}
```

---

## Design Choices

* **#Clients:** 5
  We used 5 clients — enough to feel realistic, but still easy to run on our machines.
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

**We can add histograms here(per client):**

* *IID*: 
* *Non-IID*: 

---

## Aggregation Methods

* **FedAvg**
  Weighted average of client weight vectors:

  <img width="642" height="189" alt="image" src="https://github.com/user-attachments/assets/517de1fd-49ac-4991-a15a-6bc564deac0a" />


* **FedMedian**
  Coordinate-wise median of client weight vectors. Robust to outliers or poisoned updates.

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

*(Plots comparing accuracy vs. rounds are included in `central_accuracy.csv`, `fl_iid_accuracy.csv`, `fl_non_iid_accuracy.csv`, etc.)*

---

## Vulnerabilities & Implications

* **Privacy advantage:** Raw candidate text never leaves clients — only weight updates are shared.
* **But privacy is not guaranteed:**

  * Gradient/weight updates can still leak information.
  * Membership Inference and data extraction attacks remain possible without extra safeguards.
* **Implications for EduPilot:**

  * FL reduces risk compared to centralized training, but **defenses such as secure aggregation and DP-SGD are needed** for strong protection.
  * Especially important given EduPilot’s sensitive inputs (resumes, job queries, personal experiences).

---

## How to Run the Code

1. **Centralized baseline & artifacts**

   ```bash
   python assignment-2/centralize_global_file.py
   ```

   Produces:

   * `artifacts_centralized/{tfidf_vectorizer.pkl, label_encoder.pkl, centralized_*_text_labels.csv}`
   * `central_accuracy.csv`

2. **Federated Learning (FedAvg, IID)**

   ```bash
   python assignment-2/federated_learning_iid.py
   # -> fl_iid_accuracy.csv
   ```

3. **Federated Learning (FedAvg, Non-IID)**

   ```bash
   python assignment-2/federated_learning_non_iid.py
   # -> fl_non_iid_accuracy.csv
   ```

4. **Federated Learning (FedMedian, IID)**

   ```bash
   python assignment-2/fedmedian_iid.py
   # -> fl_iid_fedmedian_accuracy.csv
   ```

5. **Federated Learning (FedMedian, Non-IID)**

   ```bash
   python assignment-2/fedmedian_non_iid.py
   # -> fl_non_iid_fedmedian_accuracy.csv
   ```

---

## References

* McMahan et al., 2017. *Communication-Efficient Learning of Deep Networks from Decentralized Data* (FedAvg).
* Yin et al., 2018. *Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates* (FedMedian).
* Carlini et al., 2022. *Membership Inference Attacks from First Principles* (for vulnerabilities).
