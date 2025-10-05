<div align="center">

# GROUP 4: EduPilot - Assignment 3

### Analysis of Differential Privacy Techniques on Balanced Synthetic Job Data

*This study investigates the trade-off between privacy and utility in differentially private neural network training for job role classification, leveraging a balanced synthetic dataset and a lightweight two-layer MLP design.*

**Team Lead:** Swetha Saseendran  
</div>

## File Overview  - IMPORTANT

***This file only explains each script’s purpose, the parameters used, and how to execute them.***
For the analysis and interpretation of the experimental results, please refer to the [**InferenceReport.md**](./InferenceReport.md).
I have made this into two files for better structure and understandability.

---

## Quick Navigation
- [Folder Structure](#folder-structure)
- [Environment Setup](#setting-up-the-conda-environment)
- [Hyperparameter Tuning Modules](#hyperparameter-tuning-modules)
  - [Privacy Accounting Comparison](#1-privacy-accounting-comparison--strong_vs_moments_accountantpy)
  - [Noise Sweep](#2--noise-sweep---analyze_noisepy)
  - [Clipping Norm Sweep](#3-clipping-norm-sweep---analyze_clippy)
  - [Other Hyperparameters](#4-other-hyperparameters---analyze_miscellanous_paramspy)
  - [Parameter Sweep Utility](#5-parameter-sweep-utility---param_sweeppy)
- [Main Training Module](#main-training-module)
  - [Baseline vs DP Training](#baseline-vs-dp-training-traindpmodelpy)
  - [Delta Sensitivity Experiment](#delta-sensitivity-plot--integrated-in-traindpmodelpy)
- [MIA Analysis](#mia-modules)
  - [Threshold-based MIA](#1-threshold-based-mia--mia_attack_thresholdipynb)
  - [Loss Threshold Attack](#2-loss-threshold-attack-model)
- [AI Usage Disclosure](#ai-usage-disclosure)

---

## Folder Structure

The assignment is organized into the following main directories. Please follow this below structure to view the files needed.<br/>

**Main Code Folders to look at: ```Hyperparam_Tuning/``` and ``` Main_Baseline_Vs_BestDP/ ```.**

The other folders are for extra credit: <br/>
MIA ATTACK:
- ```Threshold_MIA_Colab/``` <br/>
- ```Loss-threshold-attack/``` <br/>

We also testing a new model (more complex model) so see how it has affect on privacy:
- ```test```


#### Folder Structure
```
code/
├── data/
    └── dataset.csv                       # Main dataset for train

├── Hyperparam_Tuning/                    # Parameter analysis modules
    ├── analyze_clip.py                   # Clipping norm analysis
    ├── analyze_noise.py                  # Noise multiplier analysis  
    ├── analyze_miscellanous_params.py    # Other hyperparameters
    ├── param_sweep.py                    # General parameter sweep utility
    └── strong_vs_moments_accountant.py   # Privacy accounting comparison

├── Main_Baseline_Vs_BestDP/              # Main training comparison from param tuning
   └── train_dp_model.py                  # Baseline vs Best DP model train

└── Loss-threshold-attack/                # Loss Threshold Attack (EXTRA CREDIT)
    ├── dp_train.py                       # dp implementation to support before and after attack
    └── loss-threshold_attack.py          # loss threshold attack implementation
    └── post_dp_attack                    # post dp implementation attack analysis

└── Threshold_MIA_Colab/                  # Membership Inference Attack analysis (EXTRA CREDIT)
    ├── dataset.csv                       # Small subset dataset for MIA
    └── MIA_Attack_Threshold.ipynb        # MIA analysis notebook
```

---

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

### Key Dependencies (You can also run the files if you have these below dependencies):
- **PyTorch:**(neural network training)
- **Opacus:** (differential privacy)
- **Scikit-learn:**(preprocessing and metrics)
- **Matplotlib:**  (visualization)
- **NumPy/Pandas:** Standard scientific computing

---


## Hyperparameter Tuning Modules

All hyperparameter tuning scripts are located in `code/Hyperparam_Tuning/`. These modules help identify optimal settings for DP-SGD training.

### 1. Privacy Accounting Comparison- strong_vs_moments_accountant.py

This module compares two differential privacy accounting methods used in training machine learning models with DP-SGD:

- **Moments Accountant (MA)** (implemented in Opacus)
- **Strong Composition Theorem**

It helps visualize how the privacy budget **ε (epsilon)** grows across training epochs under each method.

---
#### Purpose

- Provide a **side-by-side comparison** of privacy accounting techniques.  
- Demonstrate that **Moments Accountant yields tighter bounds** on ε than Strong Composition.  
- Serve as a reference plot for how important Moments Accountant is.

---
#### Settings and Design Choice Reasoning

- **Optimizer**: SGD, lr=0.05  
- **Lot Size**: √N.  
  - Following literature (Abadi et al., Opacus examples), using lot size ~√N balances privacy and utility.  
- **Noise multiplier (σ)**: 1.0  
  - (A moderate noise level providing noticeable privacy effects while maintaining model learnability.)
- **Clipping Norm (C)**: 1.0.  
  - (A standard clipping norm to prevent gradient explosions while not overly clipping small gradients.)
- **Delta (δ)**: 1/N  
  - (Standard value recommended for DP analyses, meaning a 1-in-N chance of privacy failure.)

---
#### Inputs & Outputs

- **Input**: `dataset.csv` (columns: `job_description`, `job_role`)  
- **Output artifacts** (saved in `artifacts/`):
  - `epsilon_comparison.png` → line plot of ε vs. epochs (MA vs Strong Composition)

---

#### How to Run

```bash
python assignment-3/code/Hyperparam_Tuning/strong_vs_moments_accountant.py
```

### 2.  Noise Sweep - analyze_noise.py

This module evaluates how the **noise multiplier (σ)** affects the performance of DP-SGD when training a text classification model.  
It runs multiple DP models with varying σ values and compares their test accuracy against a non-DP baseline.

---

#### Purpose

- Empirically show the **trade-off between noise and model accuracy** in DP-SGD.  
- Provide intuition for choosing the right noise multiplier in practice.  

---
#### Settings and Design Choice Reasoning
  
- **Lot Size**: √N.  
  - (Following Abadi et al., √N offers a balance between privacy and learning stability — smaller lots increase noise, larger ones reduce privacy.)  
- **Epochs**: N / Lot Size.  
  - (Ensures each sample is seen about once, aligning privacy accounting with true data exposure.)  
- **Clipping Norm (C)**: 1.0.  
  - (Standard choice that prevents gradient explosion while retaining learning signal; avoids over-clipping small gradients.)  
- **Noise Grid**: `[0.1, 0.5, 1, 2, 3, 4, 5]`.  
  - (Covers a full privacy–utility spectrum from low to high noise; helps visualize where model performance degrades.)  
- **Delta (δ)**: 1/N.  
  - (Standard DP constant, meaning at most a 1-in-N probability of violating privacy guarantees.)  

---

#### Inputs & Outputs

- **Input**: `dataset.csv` (columns: `job_description`, `job_role`).  
- **Outputs** (saved in `artifacts_sweep/`):
  - `noise_vs_acc.png` → accuracy vs noise multiplier plot.  
  - Baseline accuracy line (dashed).  
  - Highlight of peak accuracy among DP runs.  

---

#### How to Run

```bash
python assignment-3/code/Hyperparam_Tuning/analyze_noise.py
```

### 3. Clipping Norm Sweep - analyze_clip.py

This module evaluates how the **gradient clipping norm (C)** affects the performance of DP-SGD when training a text classification model. It runs multiple DP models with varying clipping values and compares their test accuracy. I ran the hyper param sweep for different parameters, initially I tested the code with the values suggested in Abadi et al., but after tuning the params, I changed it to what graphically worked better for my synthetic dataset.

---

#### Purpose

- Empirically show the **impact of gradient clipping** on DP-SGD accuracy.  
- Provide intuition for choosing a suitable clipping norm in practice.

---

#### Settings and Design Choice Reasoning
  
- **Features**: TF-IDF (`max_features=258`, bigrams included).  
- **Model**: 2-layer feedforward NN with hidden size 128.  
- **Lot Size**: √N.  
  - Following literature (Abadi et al., Opacus examples), using lot size ~√N balances privacy and utility.  
- **Epochs**: N / Lot Size.  
  - (Ensures each sample is seen about once, aligning privacy accounting with true data exposure.)  
- **Noise Multiplier (σ)**: 1.0.  
  - (A moderate noise level providing noticeable privacy effects while maintaining model learnability.)
- **Clipping Grid**: `[0.5×, ..., 2×]` estimated median grad norm (8 values).  
  - Median is estimated from the training data and snapped to the nearest tested value for annotation.  
- **Delta (δ)**: 1/N.  
  - Widely used setting for DP guarantees.

---

#### Inputs & Outputs

- **Input**: `dataset.csv` (columns: `job_description`, `job_role`).  
- **Outputs** (saved in `artifacts/`):
  - `clip_vs_acc.png` → accuracy vs clipping norm plot.  
  - Smoothed accuracy curve (red).  
  - Observed accuracy points (black).  
  - Peak accuracy (green dot).  
  - Median clipping norm (blue dashed line, snapped to nearest tested value).

---

#### How to Run

```bash
python assignment-3/code/Hyperparam_Tuning/analyze_clip.py
```

### 4. Other Hyperparameters - analyze_miscellanous_params.py

So after I analyzed how clipping norm and noise multiplier affected my DP model, I also wanted to investigate how the other params in relation to the model itself helps the DP model attain its best accuracy and epsilon budget. So,this module allows you to sweep and analyze the effect of various hyperparameters about the model itself (hidden layer size, lot size, learning rate) on the accuracy and privacy of a DP-SGD model for job role classification.

---

#### Purpose

- Empirically show how different hyperparameters affect DP-SGD accuracy and privacy (epsilon).
- Help select optimal values for hidden units, lot size, and learning rate for your dataset.

---

#### Settings and Design Choice Reasoning  
After running all the above experiments, the following configuration was chosen as the most balanced in terms of privacy and accuracy:

- **Features**: TF-IDF (`max_features=258`, bigrams included).  
- **Model**: 2-layer feedforward NN with hidden size (swept or fixed).  
  - (Kept small to minimize parameter noise amplification under DP; 2 layers offer enough non-linearity without excessive complexity.)  
- **Lot Size**: swept or fixed.  
  - (Explored to study the trade-off between gradient averaging stability and privacy noise — smaller lots give higher noise, larger ones risk privacy loss.)  
- **Learning Rate**: swept or fixed.  
  - (Tuned to maintain convergence across DP and non-DP runs; too high causes noise amplification, too low stalls learning.)  
- **Delta (δ)**: 1/N.  
  - (Used as standard practice to represent an acceptably small privacy failure probability per individual in the dataset.)  
- **Clipping Norm (C)**: 0.17 (default - best value from previous analysis).  
- **Noise Multiplier (σ)**: 1.5 (default - best value from previous analysis).  


---

#### Inputs & Outputs

- **Input**: `dataset.csv` (columns: `job_description`, `job_role`).
- **Outputs** (saved in `artifacts/`):
  - `sweep_hidden_smooth.png`, `sweep_lot_smooth.png`, `sweep_lr_smooth.png` → accuracy vs swept parameter plots.

---

#### How to Run

```bash
python assignment-3/code/Hyperparam_Tuning/analyze_miscellanous_params.py --sweep <hidden|lot|lr> [--smooth]
```
Examples:
```bash
python assignment-3/code/Hyperparam_Tuning/analyze_miscellanous_params.py --sweep hidden --smooth
```
```bash
python assignment-3/code/Hyperparam_Tuning/analyze_miscellanous_params.py --sweep lot
```
```bash
python assignment-3/code/Hyperparam_Tuning/analyze_miscellanous_params.py --sweep lr --smooth
```

### 5. Parameter Sweep Utility - param_sweep.py

This is a general utility script that supports comprehensive parameter sweeps across C (clipping) and σ (noise multiplier) as given in the question:
- Clip norm C ∈ {0.5, 1.0}
- Noise multiplier σ ∈ {0.5, 1.0, 2.0}

#### How to Run
```bash
python assignment-3/code/Hyperparam_Tuning/param_sweep.py
```

---

## Main Training Module

The main training comparison is located in `code/Main_Baseline_Vs_BestDP/`.

### 1. Baseline vs DP Training: train_dp_model.py

This module compares the training and test accuracy of a non-private (baseline) model and a differentially private (DP-SGD) model on the job role classification task. It supports flexible privacy settings via command-line arguments.

---

#### Purpose

- Show the effect of differential privacy (DP-SGD) on model accuracy compared to a non-private baseline.
- Visualize privacy consumption (epsilon) over epochs when using DP-SGD.
- Allow experimentation with different privacy budgets and noise multipliers.

---

#### Settings (Best Params) and Design Choice Reasoning  
After all the hyperparam analysis I have gotten these values below which works best on my dataset and DP setting.
- **Features**: TF-IDF (`max_features=258`, bigrams included).
- **Model**: 2-layer feedforward NN with hidden size 128.
- **Lot Size**: 60 (from hyper-param tuning (close to √N of N i.e 56); can be changed in code).
- **Epochs**: N / Lot Size.
- **Clipping Norm (C)**: 0.17 (from hyper-param tuning; can be changed in code).
- **Noise Multiplier (σ)**: 1.5 (default value- best from tuning) configurable via `--sigma` argument.
- **Delta (δ)**: configurable via `--target_delta` argument (default: 1/N).
- **Epsilon (ε)**: configurable via `--target_eps` argument (optional).

***The choice of design was derived from previous analysis***

---

#### Inputs & Outputs

- **Input**: `dataset.csv` (columns: `job_description`, `job_role`).
- **Outputs** (saved in `artifacts/`):
  - `baseline_accuracy.csv` → baseline model train/test accuracy per epoch.
  - `dp_accuracy.csv` → DP model train/test accuracy and epsilon per epoch.
  - `baseline_vs_dp_train_test.png` → plot of train/test accuracy for both models.
  - `epsilon_curve_final.png` → plot of privacy consumption (epsilon) over epochs (if applicable).

You can test this code in 2 different ways:
1. Mention your budget delta (Accountant will get you the model for best epsilon)
2. Mention your budget delta and epsilon (Accountant will get you for the model for the optimal noise)

#### How to Run

```bash
python assignment-3/code/Main_Baseline_Vs_BestDP/train_dp_model.py [--target_eps <float>] [--target_delta <float>] [--sigma <float>]
```
Example:
```bash
python assignment-3/code/Main_Baseline_Vs_BestDP/train_dp_model.py --target_delta 0.00025
```

---

### 2. Delta Sensitivity Plot - Integrated in train_dp_model.py
After training the Baseline and Differentially Private models, we also conducted a Delta Sensitivity experiment inside the same script (train_dp_model.py) to visualize how varying δ values affect the privacy–utility trade-off while keeping the noise multiplier (σ) fixed.

---

#### Purpose

- To analyze how the choice of δ influences the relationship between ε (privacy budget) and test accuracy.
- To validate findings from Abadi et al. (2016), which show that larger δ values yield slightly better utility at the same ε but converge for moderate privacy levels.
  
___

#### Design Choice and Implementation Details  

- **Implementation Location**: Integrated inside the `main()` function of `train_dp_model.py`.  
  - (Keeping it within the main script ensures the δ-sensitivity test runs on the same data and model setup, providing consistent comparison.)  
- **Model and Parameters**: Reuses the same MLP architecture and tuned hyperparameters from the Baseline–DP comparison.  
  - (Ensures that the only variable factor is δ, isolating its direct effect on privacy and accuracy.)  
- **Delta Sweep**: Evaluated for δ ∈ {1/N, 1e-3, 5e-4, 1e-4, 5e-5}.  
  - (Chosen range covers both theoretical (1/N) and practical (1e−3–1e−5) DP regimes, helping visualize how sensitive ε–accuracy is to δ.)  
- **Fixed Parameters**: σ = 1.5, C = 0.17, Lot Size = 60.  
  - (These values were selected from prior tuning as the most balanced for stability and strong privacy.)  

#### Outputs (Saved under artifacts/)

- `delta_sensitivity_acc_vs_eps.png` → Test Accuracy vs Epsilon curves for all δ values.
- `delta_sweep.csv` → Per-epoch accuracy and ε values for each δ.


---

## MIA Modules:

The Membership Inference Attack analysis is located in `code/Threshold_MIA_Colab/`. The MIA attack is done on our best DP setting model and ***has same design choices as mentioned above for ```train_dp_model.py```.***

### 1. Threshold-based MIA: MIA_Attack_Threshold.ipynb

This Jupyter notebook implements and evaluates membership inference attacks against both baseline and DP-trained models to assess privacy leakage. **The file was taking a lot of time to run in my system, hence I went with Google Colab which gave me a better runtime environment.**
That's why I went with a subset of the the dataset that is given within the same directory.(assignment-3/code/Threshold_MIA_Colab/dataset.csv)

#### Purpose
- Demonstrate the effectiveness of membership inference attacks on machine learning models
- Compare privacy leakage between baseline and DP-trained models
- Evaluate the privacy-utility tradeoff quantitatively

#### Features
- Threshold-based membership inference attack implementation
- ROC curve analysis and AUC calculation
- Comparative analysis between models with different privacy settings
- Visualization of attack success rates

#### How to Use
1. Open the notebook in Jupyter Lab or Google Colab
2. Please add the same dataset that is given in the same directory: 'assignment-3/code/Threshold_MIA_Colab/dataset.csv' (Smaller dataset)
3. Run all cells to perform the complete MIA analysis
4. Results include attack accuracy metrics and visualization plots

#### Location
```
Code: assignment-3/code/Threshold_MIA_Colab/MIA_Attack_Threshold.ipynb
Subset dataset for this IPYNB: assignment-3/code/Threshold_MIA_Colab/dataset.csv
```

### 2. Loss Threshold Attack Model

**Membership-Inference: Yeom Loss-Threshold Attack**

Goal. Given a trained classifier and a labeled example (x,y), decide whether (x,y) was in the training set(member) or held out (non-member)
The attack relies on the observation that overfit models assign lower loss to training examples than to unseen ones.
Reference: Yeom et al., *“Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting (2018)“* ([arxiv.1709.01604](https://arxiv.org/abs/1709.01604)).

#### **What the Yeom loss-threshold attack does**:

1. Train or load a model.
  PRE (non-DP): loss_threshold_attack.py trains a high-capacity MLP on a small train fraction to encourage memorization.
  POST-DP: dp_train.py trains with DP; post_dp_attack.py evaluates the same attack on the DP model.
2. Compute per-example loss.
   For each example with true label 𝑦 and predicted class probabilities 𝑝:
    ℓ(x,y)=−logpy
3. Turn loss into a membership score.
   s(x,y)=−ℓ(x,y). Higher score ⇒ more “member-like.”
4. Evaluate separability (privacy leakage).
      Concatenate scores for train (label 1) and test (label 0), then compute ROC-AUC.
      AUC ≈ 0.5 → near random guessing (low leakage / better privacy)
      AUC → 1.0 → strong leakage (poor privacy, typically due to overfitting)

#### **Inputs, Outputs, and Artifacts**

1. Input data: dataset.csv (under Threshold_MIA_colab folder)
2. Key outputs:
         Metrics: printed Train/Test accuracy; AUC of the attack.
              Artifacts (under artifacts/):
                    *_scores_labels.npz — NumPy archives with scores, labels, auc.
                    loss-threshold-attack.png, post_yeom_roc.png, pre_vs_post_attack_comparison.png — ROC plots.
                    mia_pre_post_summary.json — compact PRE/POST AUC summary.

#### **How to run**
Activate the virtual env first if needed( please follow the step above to setup the environment).
Note: please execute in the sequence as it is mentioned below:
```
python assignment-3/code/Loss-threshold-attack/loss_threshold_attack.py  #This file shows the attack on dataset before DP impl.
python assignment-3/code/Loss-threshold-attack/dp_train.py               #This file shows DP impl on dataset.
python assignment-3/code/Loss-threshold-attack/post_rp_attack.py         #This file measures the performance before & after DP impl.
```

#### **Interpretation**:
PRE AUC ≈ 0.814 → strong membership leakage in the non-DP, overfit model.
POST AUC ≈ 0.513 → near-random; DP substantially reduces leakage.

![pre_vs_post_attack_comparison.png](artifacts/pre_vs_post_attack_comparison.png)

We evaluate privacy leakage using the Yeom loss-threshold membership-inference attack (Yeom et al., 2018).
For each example we compute the per-example cross-entropy loss and use its negative as a membership score; 
low loss indicates “member-like”. We report ROC-AUC over "train" (members) vs "test" (non-members). Our non-DP model 
yields AUC ≈ 0.814, showing clear leakage consistent with overfitting. With DP training, AUC drops to ≈ 0.513, 
near random guessing, which further indicates that DP mitigates membership leakage.

---

## LLM Usage and References

### How We Used LLMs

We used a Large Language Model (ChatGPT-4/GPT-5) throughout different stages of this assignment **for support, not substitution**.  Our focus was on learning differential privacy concepts deeply and only use the LLM to accelerate repetitive or mechanical parts of coding and for errors.

- **Code Assistance and Debugging**
  - Asked clarifying questions about how Opacus tracks ε and δ internally via the Moments Accountant API. Used the model to debug errors related to tensor shape mismatches and optimizer re-initialization when using `PrivacyEngine.make_private()`. Occasionally requested help optimizing Matplotlib code for comparing privacy curves.

  
    #### Hyperparameter Tuning Modules  
    **Files:**  
    `strong_vs_moments_accountant.py`, `analyze_noise.py`, `analyze_clip.py`, `analyze_miscellanous_params.py`, `param_sweep.py`

    - **strong_vs_moments_accountant.py** – We wrote the core code to compare Moments Accountant vs Strong Composition; ChatGPT helped me verify how Opacus tracks ε and δ internally and guided smoothing of ε-vs-epoch plots using `make_interp_spline()`.  
    - **analyze_noise.py** – We implemented the noise sweep logic; AI helped me cleanly organize parameter loops, fix matplotlib scaling issues, and format accuracy plots.  
    - **analyze_clip.py** – We wrote code to compute and visualize gradient norms; ChatGPT helped fix NaN gradient errors and suggested how to annotate the median clipping norm and peak accuracy on the plot.  
    - **analyze_miscellanous_params.py** – We built a generic script to sweep hidden size, learning rate, and lot size; ChatGPT helped add argument parsing (`--sweep`, `--smooth`) and made the sweep results modular.  
    - **param_sweep.py** – We wrote a grid search to combine clipping and noise sweeps; AI helped refactor loops for clarity and manage artifact outputs consistently.

    ####  Baseline vs Best DP Model Training  
    **Files:**  
    `train_dp_model.py` (includes delta sensitivity experiment)

    **What we did:**  
    We implemented the baseline vs DP model training pipeline using Opacus. This included exploring the privacy engine, tracking ε and δ over epochs, and plotting accuracy curves. We also added a secondary delta sensitivity experiment to visualize how δ values affect ε and accuracy.

    **How AI helped:**  
    - Enhanced my pipeline function in properly reinitializing the optimizer after calling `PrivacyEngine.make_private()` to avoid stale state issues.  
    - Helped debug tensor shape mismatches and loss calculation errors.  
    - For the delta sensitivity experiment, ChatGPT helped me structure the δ-sweep loop and store (ε, accuracy) pairs per epoch, confirming expected trends (higher δ → smaller ε and slightly better accuracy).

    #### Membership Inference Attack (MIA)  
    **Files:**  
    `Threshold_MIA_Colab/MIA_Attack_Threshold.ipynb`, `Loss-threshold-attack/loss_threshold_attack.py`, `dp_train.py`, `post_dp_attack.py`

    **What we did:**  
    We implemented the threshold-based and loss-threshold membership inference attacks to evaluate privacy leakage.   The first notebook focused on ROC-based threshold attacks, and the second directory handled pre- and post-DP comparison using Yeom’s loss threshold method.

    **How AI helped:**  
    - For **MIA_Attack_Threshold.ipynb**, ChatGPT helped structure the ROC/AUC pipeline using `sklearn.metrics`, fix axis labeling, and improve figure readability.  
    - For **loss_threshold_attack.py**, TODO: Aarti
    - For **dp_train.py** and **post_dp_attack.py**, TODO: Aarti



- **Mathematical and Conceptual Guidance**
  - Queried ChatGPT to confirm the formulas for explaining the mathematical theorems in the paper and basically helping me learn the paper, ensuring we didn’t misinterpret theoretical claims.
  - Asked for comparisons between *Strong Composition* and *Moments Accountant* in simple words to confirm understanding.
  - Verified that our epsilon-delta accounting matched the equations from Abadi et al. (2016).

- **Report Writing and Explanations**
  - Used LLMs to refine language and structure for explanations and convert rough notes into polished Markdown.
  - Generated section skeletons for the README (Folder Structure).
  - Asked the model to help structure our *InferenceReport.md* into Results → Discussion → Takeaways format.


- **Visualization and Presentation**
  - Used ChatGPT to generate ROC curve, accuracy-vs-noise, and epsilon-vs-epoch plotting snippets with proper legends and styles.
  - Modified these boilerplate plot templates ourselves to match our datasets and naming conventions.


---

### What We Did Ourselves

- Wrote all training, evaluation, and DP integration code from scratch in Python + PyTorch. Implemented baseline and DP models independently, including TF-IDF preprocessing, model setup, and accuracy tracking.
- Designed and ran all **hyperparameter tuning** experiments (σ, C, lot size, learning rate, δ sensitivity).
- Collected real experimental results (accuracy, ε per epoch) and generated all plots manually.
- Implemented our own per-example loss extraction for MIA analysis and used it in both baseline and DP models.
- Built the **Loss-Threshold Attack** pipeline and ran before/after-DP comparisons.
- Wrote all explanations, discussions, and interpretations for **InferenceReport.md** manually. Structured this **README.md** and finalized plots, charts, and results presentation.
- Structure PyTorch + Opacus training loops, batch handling, and gradient clipping setup. Plotted results (ROC curves, TPR/FPR tables), analyzed vulnerabilities.
- Built the presentation + report. Added detailed comments, describing the design choices, inference reports, and how each implementation step connects to the overall project.

---

## References

- *Deep Learning with Differential Privacy* - Abadi et al. (2016) [https://arxiv.org/abs/1607.00133](https://arxiv.org/abs/1607.00133)

- *Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting* - Yeom et al. (2018) [https://arxiv.org/abs/1709.01604](https://arxiv.org/abs/1709.01604)

- *Membership Inference Attacks From First Principles* - Carlini et al. (2022) [https://arxiv.org/abs/2112.03570](https://arxiv.org/abs/2112.03570)





