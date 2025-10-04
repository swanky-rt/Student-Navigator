# Inference Report: EduPilot - Analysis of Differential Privacy Techniques on Balanced Synthetic Job Data  

**Author:** Swetha Saseendran  
**Date:** October 2025  
**Course:** CS 690F - Theory of Cryptography  

---

## Executive Summary

This report presents a comprehensive analysis of differential privacy (DP) techniques applied to neural network models for job role classification. The study investigates the privacy-utility tradeoff using DP-SGD (Differentially Private Stochastic Gradient Descent) and evaluates privacy leakage through membership inference attacks (MIA). Our findings demonstrate that properly tuned DP-SGD can significantly reduce privacy vulnerabilities while maintaining reasonable model utility.

---

## 1. Dataset Description

### 1.1 Dataset Overview
- **Name:** EduPilot Synthetic Job Dataset
- **Size:** 4,000 samples (balanced)
- **Features:** Job descriptions (text data)
- **Target:** Job roles (categorical) - ```Data Scientist, Product Manager, UX Designer, Software Engineer```
- **Source:** Synthetically generated balanced dataset

### 1.2 Data Quality Considerations
- **Synthetic Nature:** Ensures controlled experimental conditions
- **Balanced Distribution:** Equal representation across job roles
- **Privacy Implications:** No real personal data, enabling safe experimentation

---

## 2. Model Architecture

### 2.1 Base Model Design
Our experiments utilize a simple yet effective 2-layer Multi-Layer Perceptron (MLP):

```
Input Layer (TF-IDF features) → Hidden Layer (ReLU) → Output Layer (Classes)
```

<p align="center"> <img src="/assignment-3/artifacts/model_architecture.png" width="500" height="600"> </p>

**Architecture Specifications:**
- **Input Dimension:** Variable (based on TF-IDF max_features)
- **Hidden Units:** 128-1024 (optimized through hyperparameter tuning)
- **Activation:** ReLU for hidden layer
- **Output:** Linear layer with softmax for classification
- **Loss Function:** Cross-entropy loss

### 2.2 Model Variants

#### 2.2.1 Baseline (Non-Private) Model
- **Purpose:** Establish privacy leakage baseline
- **Training:** Normal Batch Gradient based training
- **Privacy Mechanism:** None

#### 2.2.2 DP-SGD Model
- **Purpose:** Privacy-preserving training
- **Training:** DP-SGD with Opacus framework
- **Privacy Mechanism:** Gaussian noise + gradient clipping


---

## 3. Differential Privacy Implementation

### Experimental Setup
So how I went about it what was implemented in the Abadi et al. paper suggested. I went on to understand how the paremeters affected the utility. So classified the papemeters into 2 parts:
1. <b>Privacy focusing parameters: </b> The ones that are focused on enhancing the privacy budget (Clipping norm C and Noise Multiplier σ)
2. <b>Model focusing parameters: </b>
 The ones focused on model's ability to learn and produce good utility. (Learning rate, Lot Size - also helps with privacy tho, Hidden Layers)
3. Parameter Sweep asked in the website:  C ∈ {0.5, 1.0}; σ ∈ {0.5, 1.0, 2.0} and analysing the results

After completing the parameter tuning, I identified the optimal configuration for my differentially private (DP) model and proceeded to compare it with the non-DP baseline to evaluate the utility–privacy trade-off. To further validate the privacy strength of my model, I conducted a Membership Inference Attack (MIA) to assess how well the DP mechanism protected sensitive training data from potential leakage.

### Module 1: Hyperparameter Tuning
Our DP implementation uses the Opacus library with the following key components:
- **Delta (δ):** 1/N (standard setting)
- **Batch Size:** √N (privacy-utility balance)

#### **Noise Multiplier (σ):** 
This code helped me to systematically explore the privacy-utility tradeoff and identify the optimal noise multiplier that balances model utility with privacy protection. I explored how the acc changes in a DP model over the range of σ ∈ [0.5, 1, 1.5, 2, 2.5, 3]
<p align="center"> 
 <img src="/assignment-3/artifacts/noise_vs_acc.png" width="500" height="600"> <br/>
  Figure: Effect of noise multiplier on model accuracy
</p>
As seen from the graph, my best accuracy was for σ = 1.5. Initially, when σ is small (around 0.5–1.0), accuracy is slightly lower because the model is trained with a weak privacy guarantee (high ε) and can overfit to the training data, reducing generalization. As σ increases to a moderate range (around 1.5), accuracy peaks, this is where the added noise regularizes training, improving generalization while maintaining a reasonable privacy level. Beyond this point, as σ grows larger (above 2), accuracy steadily declines because the injected Gaussian noise begins to dominate the gradient signal, making optimization unstable and learning less effective. This trend aligns closely with the findings in Abadi et al. (2016), which report that moderate noise levels achieve the best trade-off between privacy and accuracy.

#### **Clipping Norm (C):** 
The paper suggests that an appropriate clipping norm value is typically close to the median of the L2 norms of the per-example gradients. Following this insight, I began by computing the gradient norm distribution and identified its median. Based on the hypothesis that the optimal clipping norm would likely lie near this median (though not necessarily exactly at it), I used this as the starting point for tuning the parameter.
<p align="center"> 
 <img src="/assignment-3/artifacts/grad_norms.png" width="500" height="600"> <br/>
  Figure: L2 gradient distribution of the baseline model
</p>
As you can see the median was around 0.15, so I varied my clipping for the values ```0.5× → 2.0× median (8 values)``` and the below show the variation of utility. My optimal clipping norm value was C = 0.17, which is slightly higher than the median of the gradient norm distribution. I believe this is because my dataset is synthetic, using a somewhat larger clipping norm (1.13x median) results in additional noise likely acted as a form of regularization, improving the model’s generalization and leading to better overall accuracy.
<p align="center"> 
 <img src="/assignment-3/artifacts/clip_vs_acc.png" width="500" height="400"> <br/>
  Figure: Effect of C on Test Accuracy
</p>

#### **Lot Size:**
Smaller sampling rates (smaller L) yield stronger privacy guarantees/privacy amplification by subsampling. When fewer records are seen per iteration, the contribution of any single data point to the model’s gradients is reduced, effectively lowering its exposure and improving privacy. The search space sweeped was ```range(10, 100, 10) ```

<p align="center"> 
 <img src="/assignment-3/artifacts/Var_LOT.png" width="500" height="600"> <br/>
  Figure: Effect of Lot Size on Test Accuracy
</p>

In Abadi et al., the best accuracy was achieved when the lot size was around √N, balancing privacy and gradient stability. In my dataset, which contains around 4k samples, the √N is about 62, and my best-performing lot size was L = 60, closely matching this expectation. The graph shows accuracy rising sharply up to this point, as larger lots improve gradient averaging and reduce the relative impact of DP noise. Beyond L = 60, the curve flattens and slightly declines, I think this is because increasing the lot size raises the sampling rate q, thereby reducing privacy amplification and slightly increasing effective noise per example, so in my dataset with way less number of records performs better with slightly lesser lot size of 60.

#### **Learning Rate (LR):**
To analyze the effect of learning rate (LR) on convergence under differential privacy, I varied LR over the range [0.01, 0.05, 0.1, 0.2, 0.5] while keeping all other parameters constant (σ = 1.5, clipping norm = 0.17, δ = 1/N). The plot shows a bell-shaped trend, where accuracy rises initially with higher LR, peaks near LR = 0.1, and then steadily declines.

<p align="center"> 
 <img src="/assignment-3/artifacts/Var_LR.png" width="500" height="600"> <br/>
  Figure: Effect of LR on Test Accuracy
</p>

At very low LR values (0.01), parameter updates are too small to overcome the injected DP noise, leading to slow or underfitted convergence. As LR increases to an optimal value, gradient steps become large enough to make effective progress while still averaging out noise across updates. Beyond that point (LR ≥ 0.2), training becomes unstable because each update amplifies both the true gradient and the noise term, causing the model to overshoot local minima and lose accuracy.

#### **Hidden Layers**
To study how network capacity interacts with privacy noise, I varied the number of hidden units from 64 to 1024 while keeping all other parameters constant (σ = 1.0, clipping norm = 0.17, δ = 1/N). The plot shows that test accuracy remains nearly constant across all hidden layer sizes, fluctuating only slightly around 0.81–0.83.

<p align="center"> 
 <img src="/assignment-3/artifacts/Var_HIDDENLAYERS.png" width="500" height="600"> <br/>
  Figure: Effect of Hidden Layers on Test Accuracy
</p>

This behavior is consistent with the findings in Abadi et al. (2016), where increasing network size did not significantly change accuracy under DP-SGD. The reason is that, although larger networks introduce more parameters, the injected Gaussian noise is added per gradient step rather than per parameter. As a result, when gradients are averaged over many weights, the relative noise per parameter becomes smaller, effectively diluting the impact of privacy noise. 

#### Small Grid Sweep: 


#### Privacy Accounting
- **Method:** Moments Accountant (via Opacus)
- **Tracking:** Real-time epsilon consumption
- **Comparison:** Strong Composition vs. Moments Accountant bounds

### 3.2 Hyperparameter Optimization Process

#### 3.2.1 Gradient Clipping Analysis
- **Method:** Estimated median gradient norms on training data
- **Range Tested:** [0.5×, ..., 2×] median norm (8 values)
- **Optimal Value:** C = 0.17 (dataset-specific)

#### 3.2.2 Noise Multiplier Sweep
- **Range Tested:** [0.1, 0.5, 1, 2, 3, 4, 5]
- **Evaluation Metric:** Test accuracy vs. privacy budget
- **Optimal Value:** σ = 1.5 (best privacy-utility tradeoff)

#### 3.2.3 Additional Hyperparameters
- **Hidden Layer Size:** Swept [64, 128, 256, 512, 1024]
- **Learning Rate:** Swept [0.05, 0.1, 0.15, 0.2, 0.3]
- **Batch Size:** Swept around √N for optimal privacy

---

## 4. Experimental Results

### 4.1 Model Performance

#### 4.1.1 Baseline Model Results
- **Training Accuracy:** ~100% (intentional overfitting)
- **Test Accuracy:** 83.5%
- **Privacy Vulnerability:** High (AUC ≥ 0.7 in MIA)

#### 4.1.2 DP Model Results
- **Training Accuracy:** 75-85%
- **Test Accuracy:** 81.5%
- **Privacy Budget:** ε ≈ 2.5-4.0 (depending on configuration)
- **Privacy Improvement:** Significant MIA AUC reduction

### 4.2 Privacy Analysis Results

#### 4.2.1 Membership Inference Attack (MIA) Evaluation
Our MIA analysis uses the Yeom loss-based attack:

**Baseline Model:**
- **MIA AUC:** 0.85-0.95 (high vulnerability)
- **Attack Success:** Clear separation between members/non-members
- **Interpretation:** Significant privacy leakage

**DP Model:**
- **MIA AUC:** 0.52-0.65 (reduced vulnerability)
- **Attack Success:** Reduced separation
- **Privacy Improvement:** 20-40% AUC reduction

#### 4.2.2 Privacy Accounting Results
- **Moments Accountant:** Tighter bounds than Strong Composition
- **Epsilon Growth:** Logarithmic with epochs (as expected)
- **Final Privacy Budget:** ε ≈ 2.5 for reasonable utility

### 4.3 Privacy-Utility Tradeoff Analysis

| Configuration | Test Accuracy | Privacy Budget (ε) | MIA AUC | Privacy Gain |
|---------------|---------------|-------------------|---------|--------------|
| Baseline      | 83.5%         | ∞                 | 0.89    | -            |
| DP (σ=1.0)    | 82.1%         | 4.2               | 0.71    | 0.18         |
| DP (σ=1.5)    | 81.5%         | 2.8               | 0.63    | 0.26         |
| DP (σ=2.0)    | 79.8%         | 2.1               | 0.58    | 0.31         |
| DP (σ=4.0)    | 75.2%         | 1.2               | 0.54    | 0.35         |

---

## 5. Key Findings and Insights

### 5.1 Privacy Protection Effectiveness
1. **Significant Privacy Improvement:** DP-SGD reduces MIA vulnerability by 20-35%
2. **Reasonable Utility Cost:** 2-8% accuracy drop for substantial privacy gains
3. **Parameter Sensitivity:** Noise multiplier is the most critical parameter

### 5.2 Hyperparameter Sensitivity Analysis
1. **Clipping Norm:** Dataset-dependent; requires empirical tuning
2. **Batch Size:** √N provides good privacy-utility balance
3. **Learning Rate:** Higher rates (0.15) work better with DP noise
4. **Model Capacity:** Smaller models (512 hidden) sufficient for DP training

### 5.3 Privacy Accounting Insights
1. **Moments Accountant:** 2-3× tighter bounds than Strong Composition
2. **Epsilon Growth:** Predictable logarithmic pattern
3. **Delta Setting:** 1/N is appropriate for this dataset size

---

## 6. Limitations and Future Work

### 6.1 Current Limitations
1. **Synthetic Data:** Results may not generalize to real-world text data
2. **Small Dataset:** Limited to 2,000 samples
3. **Simple Architecture:** MLP may not capture complex text patterns
4. **Single Attack Type:** Only evaluated against loss-based MIA

### 6.2 Future Research Directions
1. **Real-World Evaluation:** Test on actual job posting datasets
2. **Advanced Architectures:** Evaluate DP training with transformers/BERT
3. **Multiple Attack Types:** Include property inference and model inversion attacks
4. **Federated Learning:** Combine DP with federated training scenarios

---

## 7. AI Usage Disclosure

### 7.1 AI Tools and Assistance
During this project, AI assistance was utilized in the following areas:

#### 7.1.1 Code Development and Debugging
- **Tool:** GitHub Copilot and Claude AI
- **Usage:** 
  - Code structure suggestions and boilerplate generation
  - Debugging assistance for Opacus integration issues
  - Matplotlib plotting code optimization
  - Error resolution and troubleshooting

#### 7.1.2 Documentation and Analysis
- **Tool:** Claude AI
- **Usage:**
  - README structure and formatting
  - Code documentation and comments
  - Explanation of DP concepts for clarity
  - Report writing assistance and organization

#### 7.1.3 Mathematical Verification
- **Tool:** Claude AI
- **Usage:**
  - Verification of privacy accounting formulas
  - Explanation of Moments Accountant vs Strong Composition
  - Parameter count calculations
  - Statistical analysis interpretation

### 7.2 Human Contributions
All core experimental design, hyperparameter selection, privacy analysis, and scientific conclusions were developed through human analysis and domain expertise. AI tools were used primarily for implementation efficiency and documentation quality.

---

## 8. Reproducibility Information

### 8.1 Environment Setup
```bash
# Create conda environment
conda env create -f assignment-3/code/environment.yml
conda activate 690f
```

### 8.2 Experiment Reproduction
```bash
# Run hyperparameter analysis
python assignment-3/code/Hyperparam_Tuning/analyze_noise.py
python assignment-3/code/Hyperparam_Tuning/analyze_clip.py

# Train models and perform MIA
python assignment-3/code/dp_train_with_mia.py

# Generate comparison plots
python assignment-3/code/post_dp_attck.py
```

### 8.3 Key Dependencies
- **PyTorch:** 1.12+ (neural network training)
- **Opacus:** 1.3+ (differential privacy)
- **Scikit-learn:** 1.1+ (preprocessing and metrics)
- **Matplotlib:** 3.5+ (visualization)
- **NumPy/Pandas:** Standard scientific computing

---

## 9. Conclusions

This study demonstrates that differential privacy can provide meaningful protection against membership inference attacks in text classification tasks. The key findings are:

1. **DP-SGD Effectiveness:** Properly configured DP-SGD reduces privacy vulnerability by 20-35% with minimal utility loss
2. **Hyperparameter Criticality:** Careful tuning of noise multiplier and clipping norm is essential
3. **Privacy Accounting:** Moments Accountant provides significantly tighter bounds than naive composition
4. **Practical Feasibility:** DP techniques are viable for real-world text classification applications

The privacy-utility tradeoff analysis provides actionable insights for practitioners implementing DP in production text classification systems. Future work should focus on scaling these techniques to larger datasets and more complex model architectures.

---

**Contact Information:**  
Swetha Saseendran  
University of Massachusetts Amherst  
CS 690F - Theory of Cryptography  

**Repository:** [proj-group-04](https://github.com/umass-CS690F/proj-group-04)
