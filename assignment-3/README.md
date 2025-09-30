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
- **Batch size**: Lot size = √N  
- **Noise multiplier (σ)**: 1.0  
- **Clipping norm (C)**: 1.0  
- **Delta (δ)**: 1/N  

---
### Inputs & Outputs

- **Input**: `data/dataset.csv` (columns: `job_description`, `job_role`)  
- **Output artifacts** (saved in `artifacts/`):
  - `epsilon_comparison.png` → line plot of ε vs. epochs (MA vs Strong Composition)

---

### How to Run

```bash
python epsilon_accounting_comparison.py
