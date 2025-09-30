# EduPilot - Analysis of Differential Privacy Techniques on Balanced Synthetic Job Data 


## Setting Up the Conda Environment

To create the conda environment and install all dependencies for this assignment:

1. Make sure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) installed.
2. Open a terminal and navigate to the `assignment-3/code` directory:
  ```bash
  cd assignment-3/code
  ```
3. Create the environment using the provided `environment.yml` file:
  ```bash
  conda env create -f environment.yml
  ```
4. Activate the environment:
  ```bash
  conda activate 690f
  ```
5. All required dependencies will be installed automatically. If you add new packages, update the environment with:
  ```bash
  conda env update -f environment.yml --prune
  ```

You are now ready to run the scripts in this assignment.


## Module 1: Privacy Accounting Comparison: strong_vs_moments_accountant.py

This module compares two differential privacy accounting methods used in training machine learning models with DP-SGD:

- **Moments Accountant (MA)** (implemented in Opacus)
- **Strong Composition Theorem**

It helps visualize how the privacy budget **ε (epsilon)** grows across training epochs under each method.

---
### Purpose

- Provide a **side-by-side comparison** of privacy accounting techniques.  
- Demonstrate that **Moments Accountant yields tighter bounds** on ε than Strong Composition.  
- Serve as a reference plot for how important Moments Accountant is.

---
### Settings

- **Model**: 2-layer feedforward NN with ReLU, hidden size = 128  
- **Features**: TF-IDF vectors from job descriptions (`max_features=258`)  
- **Optimizer**: SGD, lr=0.05  
- **Lot Size**: √N.  
  - Following literature (Abadi et al., Opacus examples), using lot size ~√N balances privacy and utility.  
- **Noise multiplier (σ)**: 1.0  
- **Clipping Norm (C)**: 1.0.  
  - Standard practice, keeps gradients bounded without over-clipping. 
- **Delta (δ)**: 1/N  
- **Optimizer**: SGD, `lr=0.1`.   

---
### Inputs & Outputs

- **Input**: `dataset.csv` (columns: `job_description`, `job_role`)  
- **Output artifacts** (saved in `artifacts/`):
  - `epsilon_comparison.png` → line plot of ε vs. epochs (MA vs Strong Composition)

---

### How to Run

```bash
python assignment-3/code/strong_vs_moments_accountant.py
```

## Module 2: Noise Sweep — noise_vs_accuracy.py

This module evaluates how the **noise multiplier (σ)** affects the performance of DP-SGD when training a text classification model.  
It runs multiple DP models with varying σ values and compares their test accuracy against a non-DP baseline.

---

### Purpose

- Empirically show the **trade-off between noise and model accuracy** in DP-SGD.  
- Provide intuition for choosing the right noise multiplier in practice.  

---

### Settings
  
- **Features**: TF-IDF (`max_features=258`, bigrams included).  
- **Model**: 2-layer feedforward NN with hidden size 128.  
- **Lot Size**: √N.  
  - Following literature (Abadi et al., Opacus examples), using lot size ~√N balances privacy and utility.  
- **Epochs**: N / Lot Size.  
  - Ensures each record is expected to be seen about once.  
- **Clipping Norm (C)**: 1.0.  
  - Standard practice, keeps gradients bounded without over-clipping.  
- **Noise Grid**: `[0.1, 0.5, 1, 2, 3, 4, 5]`.  
  - Covers low → high noise regimes, to visualize the accuracy dropoff.  
- **Delta (δ)**: 1/N.  
  - Widely used setting for DP guarantees.

---

### Inputs & Outputs

- **Input**: `dataset.csv` (columns: `job_description`, `job_role`).  
- **Outputs** (saved in `artifacts_sweep/`):
  - `noise_vs_acc.png` → accuracy vs noise multiplier plot.  
  - Baseline accuracy line (dashed).  
  - Highlight of peak accuracy among DP runs.  

---

### How to Run

```bash
python assignment-3/code/analyze_noise.py
```
