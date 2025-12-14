# AirGapLite: Phase 3 Context Agent (Encoder/Classifier)

## Project Overview
**AirGapLite** is a hybridized, lightweight privacy preservation system designed to minimize user data exposure before it reaches conversational LLMs.

This repository contains the implementation for **Phase 3: Context Agent**, the "brain" of the system. This module is responsible for analyzing user prompts in real-time and classifying the semantic context (e.g., **"Banking"** vs. **"Restaurant"**) to trigger the appropriate privacy rules.

### Key Objectives Achieved

  * **Lightweight Architecture:** Implemented a neuro-symbolic hybrid model using **MiniLM-L6-v2** and a custom **MLP**.
  * **High Accuracy:** Achieved **96.8%** validation accuracy with robust domain separation.
  * **Ultra-Low Latency:** Optimized inference speed to **\~7.32 ms** per request (60x faster than standard LLMs).
  * **Data Engineering:** Engineered a synthetic dataset generation pipeline to overcome data scarcity and prevent overfitting.

-----

## Repository Structure

```text
├── context_agent_classifier.py  # Model Architecture (MiniLM + MLP)
├── context_agent_mlp.pth        # Trained Model Weights (Artifact)
├── demo_pipeline.py             # End-to-End Demo integrating PII extraction
├── inference.py                 # Clean Interface for Phase 4 Integration
├── train_context_agent.py       # Training Loop & Hyperparameter Optimization
├── benchmark_latency.py         # Performance Benchmarking Script
└── restbankbig.csv     # The engineered synthetic dataset (450+ samples) # moved to final_project folder
```

-----

## 🧠 System Architecture & Design Rationale
Our architecture was carefully designed to balance two competing objectives: **high semantic understanding** (usually requiring large models) and **ultra-low latency** (required for the "AirGapLite" real-time constraint).

### 1. The Encoder: `all-MiniLM-L6-v2`
Instead of using a standard BERT base model (~110M parameters) which is computationally expensive, we selected **MiniLM-L6-v2**.

* **Why this choice?**
    * **Efficiency:** It uses a 6-layer distilled transformer architecture, which is significantly faster than full-scale models while retaining 90%+ of the performance for sentence-similarity tasks.
    * **Dimensionality:** It produces dense vector embeddings of size **384**. This is half the size of standard BERT (768), directly reducing the computational load for the downstream classifier by 50%.
    * **Paper Reference:** We utilize the Sentence-BERT (SBERT) framework [2] to derive semantically meaningful sentence embeddings. Specifically, we selected the all-MiniLM-L6-v2 checkpoint, which uses distillation [1] to retain high performance while reducing the model size.

### 2. The Classifier: Custom Tuned MLP
We implemented a custom **Multi-Layer Perceptron (MLP)** head rather than a simple linear probe to capture non-linear semantic boundaries between "Banking" and "Restaurant" contexts.

* **Input Layer (384 neurons):** Matches the output dimension of the MiniLM encoder.
* **Hidden Layer (64 neurons):**
    * *Design Decision:* We chose a bottleneck size of 64 (a reduction ratio of 6:1). Through experimentation, we found this was optimal: large enough to capture feature complexity but small enough to prevent overfitting on our synthetic dataset.
    * *Activation:* **ReLU** was applied to introduce non-linearity, allowing the model to understand complex phrasing beyond simple keyword matching.
* **Regularization (Dropout p=0.1):**
    * *Design Decision:* Given the risk of memorization with small datasets, we injected a Dropout layer with `p=0.1`. This forces the network to learn robust, distributed features rather than relying on specific neurons, directly contributing to our high validation stability (96.8%).
* **Output Layer (2 neurons):** Outputs the logits for the binary classification (Bank vs. Restaurant).

### 3. Hyperparameter Optimization
We utilized the **Adam optimizer** with a learning rate of **1e-3**. We found that standard learning rates (e.g., 2e-5 used for BERT fine-tuning) were too slow for this lightweight architecture, while 1e-3 allowed for rapid convergence within 5 epochs without overshooting the loss minima.

---

## Implementation & Data Engineering

### The Challenge: Data Scarcity

Initial attempts to train the classifier using standard, small-scale datasets resulted in severe overfitting. The model memorized specific keywords rather than learning semantic context, leading to a validation accuracy plateau of \~70%.

### The Solution: Synthetic Data Pipeline

To resolve this, we implemented a **Synthetic Data Augmentation Pipeline**:

1.  **Template Generation:** Created robust templates for high-risk (Banking) and low-risk (Restaurant) scenarios, including ambiguous edge cases (e.g., asking for "hours" or "reservations" in different contexts).
2.  **Lexical Expansion:** Programmatically expanded the dataset by **400%** using synonym replacement (e.g., swapping "transfer" with "wire," "send," "move funds").
3.  **Result:** A robust, balanced dataset of **468+ unique samples**, enabling the model to generalize effectively.

-----

## Setup & Usage

### 1\. Prerequisites

Install the required dependencies:

```bash
pip install torch pandas sentence-transformers scikit-learn
```

### 2\. Training the Model

To reproduce the results, run the training script. This will load the dataset, perform the train/val split, and train the MLP.

```bash
python train_context_agent.py
```

### 3\. Running the Demo

Here is the full pipeline demo that detects the context and applies the corresponding mock privacy rules.

```bash
python demo_pipeline.py
```

### 4\. Integration (For Phase 4)

To itegrate, we directly use `inference.py` file to get predictions without worrying about the underlying model logic:

```python
from inference import predict_context

result = predict_context("I need to transfer money")
print(result) 
# Output: {'label': 'bank', 'confidence': 0.9999487400054932, 'probabilities': [5.1275885198265314e-05, 0.9999487400054932], 'is_confident': True}
```

-----

## Performance & Results

### Training Metrics

After optimizing hyperparameters (Batch Size: 8, Learning Rate: 1e-3), the model achieved rapid convergence:

  * **Validation Accuracy:** **96.8%** (Peaked at Epoch 4)
  * **Final Training Loss:** **0.0201**
  * **Convergence Speed:** \< 5 Epochs

### Latency Benchmark

We benchmarked the inference speed on standard CPU hardware:

| Model Type | Average Latency | Speedup Factor |
| :--- | :--- | :--- |
| Standard LLM Agent | \~450.00 ms | 1x |
| **AirGapLite Context Agent** | **7.32 ms** | **\~60x** |

-----

## Visuals & Screenshots

**1. Training Dynamics:**
<img width="1423" height="867" alt="image" src="https://github.com/user-attachments/assets/6461b33e-473d-4a6f-b0ff-d41a6de46960" />

*Demonstrates the rapid learning curve and stability of the loss function, proving the efficiency of the hybrid architecture.*

**2. Confusion Matrix:**
<img width="1104" height="908" alt="image" src="https://github.com/user-attachments/assets/26303085-384f-44bb-8ced-9aba13fe4d22" />

*Visualizes the model's reliability, showing near-perfect separation between sensitive (Bank) and non-sensitive (Restaurant) contexts.*

**3. Latency Comparison:**
<img width="1645" height="790" alt="image" src="https://github.com/user-attachments/assets/1d02d7ca-dd20-42fc-b6df-672f90bd02f9" />

*Highlights the massive speed advantage of our lightweight approach compared to a monolithic LLM.*

**4. Live Demo Output:**
<img width="542" height="233" alt="image" src="https://github.com/user-attachments/assets/691695b9-3327-4a7f-b408-46e2dcd90c1a" />

*The Context Agent correctly identifies a high-risk "Banking" intent (Conf: 1.00) and routes the request to the banking privacy controls. The mock response confirms the successful handover of the domain context.*

### References

> **[1] MiniLM:** Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*. arXiv:2002.10957.

> **[2] Sentence-Transformers:** Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. arXiv:1908.10084.
