<div align="center">

# Inference Report: EduPilot - Analysis of Differential Privacy Techniques on Balanced Synthetic Job Data  

```Author: Swetha Saseendran```<br>
**Date:** 5th October 2025  <br>
**Course:** CS 690F - Trustworthy and Responsible AI

---

## Executive Summary

This report presents a comprehensive analysis of differential privacy (DP) techniques applied to neural network models for job role classification. The study investigates the privacy-utility tradeoff using DP-SGD (Differentially Private Stochastic Gradient Descent) and evaluates privacy leakage through membership inference attacks (MIA). Our findings demonstrate that properly tuned DP-SGD can significantly reduce privacy vulnerabilities while maintaining reasonable model utility.

#### NOTE: This file represents our experimental setup, learnings and analysis. The README.md has the design choice justification and the details regarding each file and how to use them please refer to the [**README.md**](./README.md)..

---
</div>

## Quick Navigation

- [Executive Summary](#executive-summary)
- [Experimental Setup](#experimental-setup)
- [1. Dataset Description](#1-dataset-description)
  - [1.1 Dataset Overview](#11-dataset-overview)
  - [1.2 Data Quality Considerations](#12-data-quality-considerations)
- [2. Model Architecture](#2-model-architecture)
  - [2.1 Base Model Design](#21-base-model-design)
  - [2.2 Model Variants](#22-model-variants)
- [3. Differential Privacy Implementation](#3-differential-privacy-implementation)
  - [Module 1: Hyperparameter Tuning](#module-1-hyperparameter-tuning)
  - [Module 2: Baseline vs DP - Model Analysis and Utility Tradeoff](#module-2-baseline-vs-dp---model-analysis-and-utility-tradeoff)
  - [Module 3: Small Grid Sweep](#module-3small-grid-sweep-c--05-10--σ--05-10-20)
- [4. Membership Inference Attack and Privacy–Utility Trade-off (Extra Credit #1)](#4-membership-interference-attack-and-privacyutility-trade-off-extra-credit-1)
- [5. Additional Analytics (Extra Credit #2)](#5-additional-analytics-to-understand-differential-privacy-extra-credit-2)
  - [Strong Composition vs Moments Accountant](#strong-composition-vs-moments-accountant)
  - [Delta-Sensitivity Graph](#delta-sensitivity-graph)
- [6. Differential Privacy on TextCNN (Extra Credit #3)](#6-exploring-the-effects-of-implementing-differential-privacy-on-textcnn-extra-credit-3)
- [7. Key Findings and Insights](#7-key-findings-and-insights)
- [8. Limitations and Future Work](#8-limitations-and-future-work)
- [9. Conclusions](#9-conclusions)
- [Supplementary Section](#supplementary-section)
  - [A: Membership-Inference: Yeom Loss-Threshold Attack](#a-membership-inference-yeom-loss-threshold-attack)
  - [B: Assignment Requirements](#b-assignment-requirements--verification--results)

## Experimental Setup
So how I went about it what was implemented in the Abadi et al. paper suggested. I went on to understand how the paremeters affected the utility. So classified the parameters into 2 parts:
1. <b>Privacy focusing parameters: </b> The ones that are focused on enhancing the privacy budget (Clipping norm C and Noise Multiplier σ)
2. <b>Model focusing parameters: </b>
 The ones focused on model's ability to learn and produce good utility. (Learning rate, Lot Size - also helps with privacy tho, Hidden Layers)

After completing the parameter tuning, I identified the optimal configuration for my differentially private (DP) model and proceeded to compare it with the non-DP baseline to evaluate the utility–privacy trade-off. I also did a small parameter Sweep asked in the website:  C ∈ {0.5, 1.0}; σ ∈ {0.5, 1.0, 2.0} and analysing the results.

To further validate the privacy strength of my model, I conducted a Membership Inference Attack (MIA) to assess how well the DP mechanism protected sensitive training data from potential leakage.

#### Other Experiment:
- Apart from normal MIA threshold we also implemented Yeom Loss-Threshold Attack. (https://arxiv.org/abs/1709.01604)
- Abadi et al. (2016). talks about DP on image data, we thought it would be a good idea to use a model that has CNN capabilities too. So we tried to implement Differential Privacy on a more complex model - **TextCNN** (https://arxiv.org/abs/1408.5882) as it sounded intresting.



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
### Module 1: Hyperparameter Tuning
Our DP implementation is built using the Opacus library. I keep the following component fixed while varying other parameters (the specific configurations for each sweep are documented in the respective README files):
- Delta (δ): set to 1/N, following the standard practice in differential privacy.

```NOTE: For details about what exact values are set for other params while analysing one param are given in readme file```

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
 <img src="/assignment-3/artifacts/sweep_lot_smooth.png" width="500" height="600"> <br/>
  Figure: Effect of Lot Size on Test Accuracy
</p>

In Abadi et al., the best accuracy was achieved when the lot size was around √N, balancing privacy and gradient stability. In my dataset, which contains around 4k samples, the √N is about 62, and my best-performing lot size was L = 60, closely matching this expectation. The graph shows accuracy rising sharply up to this point, as larger lots improve gradient averaging and reduce the relative impact of DP noise. Beyond L = 60, the curve flattens and slightly declines, I think this is because increasing the lot size raises the sampling rate q, thereby reducing privacy amplification and slightly increasing effective noise per example, so in my dataset with way less number of records performs better with slightly lesser lot size of 60.

#### **Learning Rate (LR):**
To analyze the effect of learning rate (LR) on convergence under differential privacy, I varied LR over the range [0.01, 0.05, 0.1, 0.2, 0.5] while keeping all other parameters constant (σ = 1.5, clipping norm = 0.17, δ = 1/N). The plot shows a bell-shaped trend, where accuracy rises initially with higher LR, peaks near LR = 0.1, and then steadily declines.

<p align="center"> 
 <img src="/assignment-3/artifacts/sweep_lr_smooth.png" width="500" height="600"> <br/>
  Figure: Effect of LR on Test Accuracy
</p>

At very low LR values (0.01), parameter updates are too small to overcome the injected DP noise, leading to slow or underfitted convergence. As LR increases to an optimal value, gradient steps become large enough to make effective progress while still averaging out noise across updates. Beyond that point (LR ≥ 0.2), training becomes unstable because each update amplifies both the true gradient and the noise term, causing the model to overshoot local minima and lose accuracy.

#### **Hidden Layers**
To study how network capacity interacts with privacy noise, I varied the number of hidden units from 64 to 1024 while keeping all other parameters constant (σ = 1.0, clipping norm = 0.17, δ = 1/N). The plot shows that test accuracy remains nearly constant across all hidden layer sizes, fluctuating only slightly around 0.81–0.83.

<p align="center"> 
 <img src="/assignment-3/artifacts/sweep_hidden_smooth.png" width="500" height="600"> <br/>
  Figure: Effect of Hidden Layers on Test Accuracy
</p>

This behavior is consistent with the findings in Abadi et al. (2016), where increasing network size did not significantly change accuracy under DP-SGD. The reason is that, although larger networks introduce more parameters, the injected Gaussian noise is added per gradient step rather than per parameter. As a result, when gradients are averaged over many weights, the relative noise per parameter becomes smaller, effectively diluting the impact of privacy noise. 

### Module 2: Baseline Vs DP - Model Analysis and Utility Tradeoff
My best DP setting derieved from above analysis are: (Fixed: δ = 1/N and 50 epochs)
|   **Hyperparameter** | **Best Value** |
| -------------------: | -------------: |
|         Hidden Units |            128 |
|   Learning Rate (LR) |            0.1 |
|         Lot Size (L) |             60 |
|    Clipping Norm (C) |           0.17 |
| Noise Multiplier (σ) |            1.5 |


I would also like to analyse how deviant my model would be when compared to the empirical analysis done in the paper Abadi et al., when compared to the Baseline Non-DP Model. The below table shows the variable params between my best results with the paper:
|       **Hyperparameter** |                             **Abadi et al. (2016)** | **My Best DP Setting** |
| -----------------------: | --------------------------------------------------: | ---------------------: |
|         **Lot Size (L)** |                                             √N ≈ 62 |                     60 |
|    **Clipping Norm (C)** |                     ≈ median gradient norm (≈ 0.15) |                   0.17 |


In the Figure on the LEFT, baseline model (blue and orange curves) converges quickly, reaching around 0.89 train accuracy and 0.84 test accuracy. However, the clear gap between training and test curves indicates overfitting, the model fits the training data more precisely but generalizes slightly worse. The DP-SGD model (green and red curves) converges more slowly due to injected Gaussian noise and gradient clipping, but both curves closely track each other throughout training. The final accuracies (~0.83 train and ~0.81.5 test) are only slightly below the baseline, showing that privacy noise reduces overfitting and improves generalization stability.

<p align="center"> 
  <img src="/assignment-3/artifacts/baseline_vs_dp_train_test_best.png" width="450" height="400">
  <img src="/assignment-3/artifacts/baseline_vs_dp_train_test.png" width="450" height="400"><br/>
  <b>Figure:</b> Comparison of Baseline vs DP Model Accuracy with my tuned settings (left) and  Abadi et al. empirical settings(right)
</p>

<p align="center"> 
  <img src="/assignment-3/artifacts/epsilon_curve_final_best.png" width="450" height="400">
  <img src="/assignment-3/artifacts/epsilon_curve_final.png" width="450" height="400"><br/>
  <b>Figure:</b> Privacy Consumption (ε) over Epochs with my tuned settings (left) and  Abadi et al. empirical settings(right)
</p>

In the Figure on the RIGHT which uses the settings from Abadi et al.,(clip = 0.15, lot size = 62, σ = 1.0), the DP model achieved decent accuracy but still lagged slightly behind my tuned configuration. The key difference, however, lies in the privacy budget (ε). While the paper’s model reached ε ≈ 5.04, my optimized setup achieved ε = 2.53, representing nearly a 50% reduction in privacy loss while also improving test accuracy by ~2.25%. This clearly demonstrates that fine-tuning both privacy-related parameters (σ, C) and model-specific hyperparameters (lot size, LR, hidden units) for a given dataset can significantly improve the privacy–utility balance. In smaller or synthetic datasets like mine, tighter clipping and moderate noise levels provide stronger privacy guarantees without compromising accuracy.

```My best DP setting has a privacy budget of 2.53 and a privacy degradation from baseline of only -2.5% This shows the utility-rpivacy tradeoff, where some information is lost because of implementing differential privacy as seen in the lessened accuracy, but this tradeoff has given rise to the a way better privacy budget.```

| **Model**                       | **Source**            | **Final Test Accuracy** | **ε (Epsilon)** | **Δ Accuracy (%)** | **Δ ε (Privacy Gain %)** |
| ------------------------------- | --------------------- | ----------------------- | --------------- | ------------------ | ------------------------ |
| Baseline (Non-private)          | -                     | 0.8400                  | –               | –                  | –                        |
| DP-SGD (Differentially Private) | *Abadi et al. (2016)* | 0.7925                  | 5.04            | –                  | –                        |
| DP-SGD (Differentially Private) | **My Experiment**     | **0.8150**              | **2.53**        | **+2.25%**  from Abadi et al.       | **−49.8%**    from Abadi et al.           |

### ***Module 3:Small Grid Sweep C ∈ {0.5, 1.0}; σ ∈ {0.5, 1.0, 2.0}:***
I also wanted to analyse the ranges given in the website, so I conducted a grid search over C ∈ {0.5, 1.0} and σ ∈ {0.5, 1.0, 2.0}, keeping all other hyperparameters fixed (learning rate = 0.1, lot size = 60, δ = 1/N). Each configuration was trained for 50 epochs, and both training/test accuracies and the corresponding ε values were recorded using the Opacus PrivacyEngine.
- Increasing σ → stronger privacy (ε ↓) but slightly slower or noisier training.
- Increasing C → better gradient preservation but higher privacy cost.

The results I got from the sweep were:
| **Clip Norm (C)** | **Noise Multiplier (σ)** | **ε (epsilon)** | **Train Accuracy** | **Test Accuracy** |                              **Observation** |
| ----------------: | -----------------------: | --------------: | -----------------: | ----------------: | -------------------------------------------: |
|               0.5 |                      0.5 |           49.27 |             0.8422 |            0.8163 |         Weak privacy (ε≈49), stable accuracy |
|               0.5 |                      1.0 |            8.63 |             0.8369 |            0.8138 |      Better privacy, slight drop in accuracy |
|               0.5 |                      2.0 |            2.83 |             0.8418 |            0.8112 |           Strong privacy, slower convergence |
|               1.0 |                      0.5 |           49.27 |             0.8441 |            0.8188 |       Weak privacy, slightly better accuracy |
|               1.0 |                      1.0 |            8.63 |             0.8535 |        **0.8225** | **Best balance** between privacy and utility |
|               1.0 |                      2.0 |            2.83 |             0.8580 |            0.81 |           Strong privacy, mild accuracy loss, more noise |

<p align="center"> 
 <img src="/assignment-3/artifacts/dp_param_sweep_test_acc_vs_epoch.png" width="500" height="600"> <br/>
  Figure: Small Grid sweep: Test Acc Vs. Epoch and ε values
</p>

The test accuracy curves (see figure) show that all models converge to similar final accuracies (~0.81–0.82), but their privacy losses (ε) differ dramatically. *From the give ranges*, the configuration (C = 1.0, σ = 1.0) achieved the best overall trade-off, reaching the highest test accuracy (0.8225) at a higher privacy cost (ε ≈ 8.63). But by paramteter tuning specifically based on my datset (median of gradients, specific range of C and σ, tune LR and lot size) I was able to get a way lesser privacy budget with just a mere 0.7% decrement in accuracy.

| **Model Configuration**                | **Test Accuracy** | **ε (Epsilon)** | **Δ Accuracy (%)** | **Δ ε (Privacy Gain %)** |
| -------------------------------------- | ----------------: | --------------: | -----------------: | -----------------------: |
| **Best from Sweep (C = 1.0, σ = 1.0)** |            0.8225 |            8.63 |                  – |                        – |
| **My Tuned Model (C = 0.17, σ = 1.5)** |        **0.8150** |        **2.53** |          **−0.7%** |               **−70.7%** |


This demonstrates that tuning of both privacy and model parameters can substantially improve the privacy–utility trade-off, outperforming general default settings from the parameter sweep. I still stand with my original analysis that my best DP setting, as I feel the degradation is 0.7% is acceptable for such a good improvement in privacy budget:
|   **Hyperparameter** | **Best Value** |
| -------------------: | -------------: |
|         Hidden Units |            128 |
|   Learning Rate (LR) |            0.1 |
|         Lot Size (L) |             60 |
|    Clipping Norm (C) |           0.17 |
| Noise Multiplier (σ) |            1.5 |


## 4. Membership Interference Attack and Privacy–Utility trade-off (Extra Credit #1)
I implemnted the MIA attack on the DP model from my best settings and these were the results we got:
<p align="center"> 
 <img src="/assignment-3/artifacts/Threshold_MIA_Attack.png" width="500" height="600"> <br/>
  Figure: Small Grid sweep: Test Acc Vs. Epoch and ε values
</p>


| **Configuration** | **Test Accuracy** | **Privacy Budget (ε)** | **MIA AUC** |                     **Privacy Gain** |                                                     **Utility–Privacy Trade-off** |
| ----------------- | ----------------: | ---------------------: | ----------: | -----------------------------------: | --------------------------------------------------------------------------------: |
| **Baseline**      |             84.0% |         ∞ (No privacy) |   **0.812** |                                    – |                                            High utility, **no privacy guarantee** |
| **DP (σ = 1.5)**  |             81.5% |               **2.53** |   **0.632** | **+22% reduction in attack success** | Small accuracy drop (**−2.5%**) for **strong privacy guarantee (finite ε vs. ∞)** |

The privacy–utility trade-off observed in this above table highlights how differential privacy can effectively protect sensitive information with only a minimal impact on model performance. The DP-SGD model achieved a test accuracy of 81.5%, compared to 84% for the baseline, demonstrating that enforcing privacy led to just a ~2.5% drop in utility. At the same time, the privacy budget improved dramatically, from no protection (ε = ∞) in the baseline to a strongly private ε = 2.53, while the MIA AUC decreased from 0.812 to 0.632, indicating a significant reduction in an attacker’s ability to infer training membership. The injected Gaussian noise and gradient clipping acted as implicit regularizers, reducing overfitting and improving generalization. This demonstrates that although the trade-off cannot be completely eliminated, it can be strategically managed to extract the best possible balance between model utility and data privacy.

| **Model**        | **TPR @ FPR ≤ 0.100** | **TPR @ FPR ≤ 0.010** | **TPR @ FPR ≤ 0.001** | **AUC** |
|------------------:|----------------------:|----------------------:|----------------------:|--------:|
| **Baseline MLP** | 0.2843               | 0.0129               | 0.0014               | **0.812** |
| **DP MLP**       | 0.1671               | 0.0057               | 0.0029               | **0.632** |

- The **Baseline MLP** achieves higher true positive rates across all false positive thresholds, indicating that an attacker can more easily distinguish members from non-members.  
- The **DP MLP** trained with DP-SGD exhibits much lower TPR values, with its ROC curve moving closer to the diagonal (AUC reduced from 0.812 → 0.632).  
- This reflects a **~22% reduction in attack success**, confirming that differential privacy effectively limits membership inference capability, especially under strict FPR thresholds (0.01 and 0.001).


---
## 5. Additional Analytics to understand Differential Privacy (EXTRA CREDIT #2)
### Strong Composition Vs. Moments Accountant
<p align="center"> 
 <img src="/assignment-3/artifacts/epsilon_comparison.png" width="500" height="600"> <br/>
  Figure: Strong Composition Vs Moments Accountant
</p>
This plot compares Strong Composition (orange) with the Moments Accountant (green, from Opacus) in tracking privacy loss (ε) over training epochs on our DP model.
- Strong Composition: Estimates ε conservatively, assuming worst-case accumulation. ε grows rapidly and exceeds 50 after 50 epochs.
- Moments Accountant: Provides a much tighter bound by tracking privacy loss via moment statistics, keeping ε below 5 even after 50 epochs.
The Moments Accountant offers more realistic and tighter privacy guarantees, allowing longer training with stronger privacy and better model utility than traditional composition methods.


### Delta-Sensitivity Graph
We varied all parameters keeping delta as 1/N. We were curious what will happen if delta was varied rather. This graph was done on our Best DP Param setting to analyse what happes what happens if delta changes:

<p align="center"> 
 <img src="/assignment-3/artifacts/delta_sensitivity_acc_vs_eps.png" width="500" height="600"> <br/>
  Figure: Delta Sensitivity Graph for Best DP Setting- What happens when delta changes?
</p>

The resulting plot shows the expected privacy–utility trade-off:
- As ε increases, test accuracy rises steadily for all δ values.
- At low ε (stricter privacy), larger δ (e.g., 1e-3) attains higher accuracy earlier, since it relaxes the privacy constraint.
- For moderate ε and above (ε > 1.5), the curves converge, indicating δ has minimal effect on accuracy once privacy noise becomes small.
- This pattern mirrors Abadi et al. (2016) (Figure 4 in the paper) and validates that δ can typically be fixed (e.g., 1/N or 1e-5) while reporting ε as the key privacy metric.

---
## 6. Exploring the effects of implementing Differential Privacy on TextCNN (EXTRA CREDIT #3)
This section analyzes how differentially private stochastic gradient descent (DP-SGD) affects a TextCNN model trained for text classification. The experiment explores how learning rate, noise multiplier (σ), clipping norm (C), and batch size influence both model accuracy and privacy.

### What is TextCNN?
TextCNN, proposed by Yoon Kim (2014) (https://arxiv.org/abs/1408.5882) in Convolutional Neural Networks for Sentence Classification, is a pioneering deep learning model that applies convolutional neural networks (CNNs) to natural language processing (NLP) tasks. Unlike recurrent models that process text sequentially, TextCNN treats a sentence as a spatial structure of word embeddings, using convolutional filters to capture local n-gram features such as key phrases and patterns indicative of meaning or sentiment. Through simple yet powerful layers of convolution, pooling, and classification, TextCNN achieves strong performance on text classification benchmarks demonstrating that local semantic patterns can be learned efficiently without complex recurrent architectures. Its simplicity, speed, and interpretability have made it a foundational baseline in modern text classification pipelines.


### Why TextCNN?
Abadi et al. talks about DP on image data, we thought it would be a good idea to use a model that has CNN capabilities too. But since our data is text data we went with TextCNN as it sounded intresting and we wanted to explore it.

### Model Architecture 
* ⁠Embedding layer (dim=128)
* ⁠  ⁠Three parallel 1D convolutions (kernel sizes 3, 4, 5)
* ⁠ ⁠Global max pooling
* ⁠  Fully connected output layer
* ⁠ Dropout = 0.2 
```Note: design choice justification in README.md```


### Privacy–Utility Grid (Clipping Norm C vs Noise σ)
This was a heavier model and it took a lot of time and computation to run, so we settled with tuning just the privacy metrics the clipping norm and the noise multiplier - with the small grid sweep given in the website: Clip norm C ∈ {0.5, 1.0}; Noise multiplier σ ∈ {0.5, 1.0, 2.0}

The following 3D landscape shows how different combinations of clipping norm (C) and noise multiplier (σ) affect model accuracy and privacy ε. Each point represents a trained model configuration. We are trying to tune our privacy parameters to achieve best accuracy-privacy balance:

<p align="center"> 
 <img src="/assignment-3/artifacts/TextCNN_3D_Landscape.png" width="500" height="600"> <br/>
  Figure: 3D Landscape portraying the acc and epsilon budget for each model varied by C and σ
</p>


| Clip Norm (C) | Noise Multiplier (σ) |   Accuracy | Epsilon (ε) |
| ------------: | -------------------: | ---------: | ----------: |
|           0.5 |                  0.5 |     0.8238 |       26.56 |
|           1.0 |                  0.5 |     0.8300 |       26.56 |
|           0.5 |                  1.0 |     0.7950 |        4.11 |
|           1.0 |                  1.0 |     0.7850 |        4.11 |
|           0.5 |                  2.0 |     0.6663 |        1.38 |
|           1.0 |                  2.0 |     0.6313 |        1.38 |

From the above table, we want to find a good balance of utility degradation to increase in epsilon privacy budget:

| Range                                  | What happens                                   | Explanation                                                                                                                                                                                       |
| -------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **σ = 0.5 (low noise, high ε ≈ 26.6)** | High accuracy (0.82–0.83) but weak privacy   | A low noise multiplier means less randomization added to gradients → better model learning → higher accuracy. But the corresponding ε is large (≈ 26), meaning weaker privacy guarantees.         |
| **σ = 1.0 (medium noise, ε ≈ 4.1)**    | Balanced performance (0.78–0.80)            | This level adds enough noise to meaningfully improve privacy (ε ≈ 4 → good DP guarantee) while still maintaining decent accuracy. This is often considered the *sweet spot* for practical DP-SGD. |
| **σ = 2.0 (high noise, ε ≈ 1.4)**      | Strong privacy but poor accuracy (0.63–0.67) | Large noise dominates the gradient updates, harming learning. You gain strong DP guarantees (small ε ≈ 1.4 → very private) but at a major cost to utility.    

So the best DP setting for us was:
| Parameter                | Value   | Justification                                                                                          |
| ------------------------ | ------- | ------------------------------------------------------------------------------------------------------ |
| **Clip Norm (C)**        | **0.5** | Provides sufficient gradient control without degrading learning. Keeps noise scaling stable.           |
| **Noise Multiplier (σ)** | **1.0** | Strikes a solid balance between acceptable accuracy (~79–80%) and good privacy (ε ≈ 4).                |
| **ε(δ)**                 | ≈ 4.1   | A practical level of differential privacy, commonly cited in research papers as a good privacy budget. 

We now compared the best DP setting to the baseline model to see how the Test accuracy curve looks for 35 epochs:

<p align="center"> 
 <img src="/assignment-3/artifacts/TextCNN_baseline_vs_DP.png" width="500" height="600"> <br/>
  Figure: Test Accuracy vs Epoch: Baseline Non DP Vs Best DP setting - TextCNN 
</p>

Utility Privacy Tradeoff:
| Model Type           | Test Accuracy | ε (Epsilon) | δ (Delta) | Privacy Level |
|----------------------|---------------|--------------|------------|----------------|
| **Baseline (Non-DP)** | 0.8420        | ∞            | 0          | No privacy (unbounded) |
| **DP-SGD Model**      | 0.7875        | 4.106        | 1 / N ≈ 0.00025 | Strong privacy guarantee |

Comparing with our original MLP model:
### Comparison of Utility–Privacy Trade-offs (TextCNN vs MLP)

| **Model Type** | **Architecture** | **Test Accuracy (Baseline)** | **Test Accuracy (DP)** | **ε (Epsilon)** | **δ (Delta)** | **Privacy Level** | **Observation** |
|----------------|------------------|------------------------------:|-----------------------:|----------------:|---------------:|------------------:|-----------------|
| **MLP** | 2-layer, 128 hidden units, TF-IDF (bigrams) | 84.0% | **81.5%** | **2.53** | 1 / N ≈ 0.00025 | Strong privacy guarantee | Minimal accuracy loss (**−2.5%**) for strong privacy; performs best overall due to simpler model and synthetic data. |
| **TextCNN** | Embedding + Conv(3,4,5) + Global Pool + FC | 84.2% | **78.8%** | **4.11** | 1 / N ≈ 0.00025 | Moderate privacy guarantee | Slightly larger accuracy drop; higher ε due to more parameters amplifying DP noise. |


The **MLP model** provides a **better utility–privacy balance** with achieving higher accuracy and a lower privacy budget (ε) compared to TextCNN.  What we understand from this is that that for simpler, synthetic datasets, **lightweight models like MLP** handle DP-SGD noise more effectively than deeper architectures. 


### Key Takeaway from implementing TextCNN:
Like how with the dataset domain (Image, text, voice) one must tune parameters, a different model architecture also demands proper hyperparam tuning.


---
## 7. Key Findings and Insights

###  Privacy Protection Effectiveness

DP-SGD significantly improved privacy, reducing the budget from ∞ to **ε = 2.53** with only a **2.5% accuracy drop**. The MIA AUC decreased from **0.812 → 0.632**, showing a **22% lower attack success rate**. The tuned setup *(C = 0.17, σ = 1.5, L = 60, LR = 0.1)* achieved very less degradation from baseline accuracy while ensuring strong privacy.

### Hyperparameter Sensitivity

* **Clipping Norm (C):** Best at **0.17** (~1.13x median gradient norm). Too small over-clips; too large weakens privacy.
* **Lot Size (L):** Optimal at **60**, balancing privacy amplification and stable gradients.
* **Learning Rate (LR):** **0.1** gave stable convergence; higher values amplified noise.
* **Model Capacity:** Accuracy stayed constant (~0.81–0.83); noise diluted with more parameters.

Hyper parameter tuning is extremely important, not just the privacy related params like C and σ, we also need to tune params like lot size, LR to get a higher utility model *with a good privacy*. This assignment shows the same.


### Privacy Accounting Insights

Using the **Moments Accountant**, ε grew sublinearly and stabilized near **2.53** after 50 epochs. This confirmed the theoretical √T scaling. The chosen **δ = 1/N** was suitable for the dataset size (~4k).



---

## 8. Limitations and Future Work

### Current Limitations

Results are based on **synthetic, small-scale data** (~4k samples) and a **simple MLP model**, evaluated only against **MIA attacks**, so findings may not fully generalize.

### Future Directions

Test on **real datasets**, explore **transformer-based models**, extend to **more attack types**, and combine DP with **federated learning** for broader privacy protection.

---

## 9. Conclusions

This study demonstrates that differential privacy can provide meaningful protection against membership inference attacks in text classification tasks. The privacy-utility tradeoff analysis provides actionable insights for practitioners implementing DP in production text classification systems. Future work should focus on scaling these techniques to larger datasets and more complex model architectures.

---

## SUPPLEMENTARY SECTION
### A: Membership-Inference: Yeom Loss-Threshold Attack
  We implemented another type of MIA attack apart from the Threshold MIA
  - Given a trained classifier and a labeled example (x,y), decide whether (x,y) was in the training set(member) or held out (non-member)
  - The attack relies on the observation that overfit models assign lower loss to training examples than to unseen ones.
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

  <p align="center"> 
  <img src="/assignment-3/artifacts/pre_vs_post_attack_comparison.png" width="500" height="600"> <br/>
    Figure: Yeom Loss-Threshold Attack- Baseline Vs DP Model
  </p>

  #### **Interpretation**:
  PRE AUC ≈ 0.814 → strong membership leakage in the non-DP, overfit model.
  POST AUC ≈ 0.513 → near-random; DP substantially reduces leakage.
  We evaluate privacy leakage using the Yeom loss-threshold membership-inference attack (Yeom et al., 2018).
  For each example we compute the per-example cross-entropy loss and use its negative as a membership score; 
  low loss indicates “member-like”. We report ROC-AUC over "train" (members) vs "test" (non-members). Our non-DP model 
  yields AUC ≈ 0.814, showing clear leakage consistent with overfitting. With DP training, AUC drops to ≈ 0.513, 
  near random guessing, which further indicates that DP mitigates membership leakage.

### B: Assignment Requirements
This section demonstrates how our submission fulfills each requirement for **DP-SGD implementation, experimentation, and reporting** using references to both the [`README.md`](./README.md) and the [`InferenceReport.md`](./InferenceReport.md).

#### DP-SGD Implementation (Per-example Clipping + Gaussian Noise)

**Requirement:**  
Use per-example gradient clipping and Gaussian noise (Opacus/TF-Privacy). Compute ε(δ) using the library’s accountant (δ = 1/N).  

**Our Implementation:**  
- Implemented using **Opacus** (see `train_dp_model.py`).  
- Enabled *per-example clipping* and *Gaussian noise injection* with  
  `PrivacyEngine.make_private()` for our MLP and TextCNN models.  
- Results validated in *Section 3: Differential Privacy Implementation* and  
  *Section 5: Strong Composition vs Moments Accountant* in [`InferenceReport.md`](./InferenceReport.md).

**References:**  
- `README.md → [Main Training Module]`  
- `InferenceReport.md → Section 3 (DP Implementation)`  
- `InferenceReport.md → Section 5 (Accounting Comparison)`  

---

#### Baseline vs DP-SGD Comparison

**Requirement:**  
Compare DP-SGD model to non-DP baseline.

**Our Implementation:**  
- Built baseline model (standard SGD) and DP-SGD model with same architecture.  
- Compared accuracy, ε(δ), and overfitting tendencies.  
- Found baseline accuracy = 0.84 vs DP-SGD accuracy = 0.815 with ε = 2.53.  
- DP model generalizes better and reduces privacy leakage (MIA AUC drop 0.812 → 0.632).

**References:**  
- `README.md → [Main_Baseline_Vs_BestDP/train_dp_model.py]`  
- `InferenceReport.md → Section 3 (Module 2: Baseline vs DP Analysis)`  
- `InferenceReport.md → Section 4 (MIA and Privacy–Utility Trade-off)`  

---


#### Grid Sweep Over Privacy Parameters

**Requirement:**  
Perform small grid sweep:  
C ∈ {0.5, 1.0}, σ ∈ {0.5, 1.0, 2.0}.  
Keep runtime modest (reasonable batch size & epochs).

**Our Implementation:**  
- Conducted full parameter sweep using `param_sweep.py`.  
- Reported results in a **table and visualization** showing ε, accuracy, and privacy trade-offs.  
- Identified best configuration **(C=1.0, σ=1.0)** from sweep and compared to our **tuned optimal** setting **(C=0.17, σ=1.5)** achieving 70.7% lower ε with negligible accuracy drop.

**References:**  
- `README.md → [Hyper Parameter Tuning Modules- 5. Parameter Sweep Utility]`
- `InferenceReport.md → Section 3 (Module 3: Small Grid Sweep)`  

---

#### Report ε(δ), Accuracy, and Hyperparameter Effects

**Requirement:**  
Report privacy budget ε(δ) alongside performance metrics and hyperparameter effects.

**Our Implementation:**  
- Used Opacus Moments Accountant to compute ε for each training epoch.  
- Plotted ε–accuracy trends and analyzed influence of σ, C, LR, L, and hidden units.  
- Provided plots and tables (noise_vs_acc.png, clip_vs_acc.png, sweep_lr_smooth.png, etc.). 

**References:**  
- `InferenceReport.md → Section 3 (Hyperparameter Tuning)`  
- `InferenceReport.md → Section 3 (Module 2: Baseline vs DP Analysis)`  
- `InferenceReport.md → Section 7 (Key Findings and Insights)`  

---

#### (Optional) Membership Inference Attack (MIA)

**Requirement:**  
Re-run Week-3 MIA or extraction to demonstrate privacy impact.

**Our Implementation:**  
- Implemented **Threshold-based MIA** and **Yeom Loss-Threshold MIA**.  
- Compared baseline vs DP models → MIA AUC reduced from 0.812 → 0.632 (Threshold) 
- Demonstrates significant privacy improvement via DP-SGD.  
- Complete workflow detailed under *Extra Credit #1* and *Supplementary Section A*.

**References:**  
- `README.md → [MIA Modules]`  
- `InferenceReport.md → Section 4 (MIA Attack and Privacy–Utility Trade-off)`  
- `InferenceReport.md → Supplementary Section A (Yeom Loss-Threshold Attack)`  

---

####  Deliverables — Repository Structure and Reproducibility

**Requirement:**  
Provide clear code, documentation, and results in GitHub repository.

**Our Implementation:**  
- Complete modular folder structure documented in `README.md`.  
- Separate files for hyperparameter tuning, main training, and MIA attacks.  
- `environment.yml` ensures reproducibility of Python dependencies.  
- Clear instructions for running each script and reproducing plots.  
- All results (figures, ε tables, accuracy logs) saved under `/artifacts/`.
- AI Disclosure and module wise how AI was used and in reports.

**References:**  
- `README.md → [Folder Structure]` 
- `README.md → [LLM Usage]` 
- `README.md → [Setting Up the Conda Environment]`  
- `InferenceReport.md → All result figures and tables`

---

**Repository:** [proj-group-04](https://github.com/umass-CS690F/proj-group-04)

---
