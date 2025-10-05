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
So how I went about it what was implemented in the Abadi et al. paper suggested. I went on to understand how the paremeters affected the utility. So classified the parameters into 2 parts:
1. <b>Privacy focusing parameters: </b> The ones that are focused on enhancing the privacy budget (Clipping norm C and Noise Multiplier σ)
2. <b>Model focusing parameters: </b>
 The ones focused on model's ability to learn and produce good utility. (Learning rate, Lot Size - also helps with privacy tho, Hidden Layers)

After completing the parameter tuning, I identified the optimal configuration for my differentially private (DP) model and proceeded to compare it with the non-DP baseline to evaluate the utility–privacy trade-off. I also did a small parameter Sweep asked in the website:  C ∈ {0.5, 1.0}; σ ∈ {0.5, 1.0, 2.0} and analysing the results.

To further validate the privacy strength of my model, I conducted a Membership Inference Attack (MIA) to assess how well the DP mechanism protected sensitive training data from potential leakage.


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

In the Figure on the RIGHT which uses the settings from Abadi et al.,(clip = 0.15, lot size = 62, σ = 1.0), the DP model achieved decent accuracy but still lagged slightly behind my tuned configuration. The key difference, however, lies in the privacy budget (ε). While the paper’s model reached ε ≈ 5.04, my optimized setup achieved ε = 2.53, representing nearly a 50% reduction in privacy loss while also improving test accuracy by ~2.8%. This clearly demonstrates that fine-tuning both privacy-related parameters (σ, C) and model-specific hyperparameters (lot size, LR, hidden units) for a given dataset can significantly improve the privacy–utility balance. In smaller or synthetic datasets like mine, tighter clipping and moderate noise levels provide stronger privacy guarantees without compromising accuracy.

| **Model**                       | **Source**            | **Final Test Accuracy** | **ε (Epsilon)** | **Δ Accuracy (%)** | **Δ ε (Privacy Gain %)** |
| ------------------------------- | --------------------- | ----------------------- | --------------- | ------------------ | ------------------------ |
| Baseline (Non-private)          | -                     | 0.8400                  | –               | –                  | –                        |
| DP-SGD (Differentially Private) | *Abadi et al. (2016)* | 0.7925                  | 5.04            | –                  | –                        |
| DP-SGD (Differentially Private) | **My Experiment**     | **0.8150**              | **2.53**        | **+2.8%**          | **−49.8%**               |

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
|               1.0 |                      2.0 |            2.83 |             0.8580 |            0.8188 |           Strong privacy, mild accuracy loss |

<p align="center"> 
 <img src="/assignment-3/artifacts/dp_param_sweep_test_acc_vs_epoch.png" width="500" height="600"> <br/>
  Figure: Small Grid sweep: Test Acc Vs. Epoch and ε values
</p>

The test accuracy curves (see figure) show that all models converge to similar final accuracies (~0.81–0.82), but their privacy losses (ε) differ dramatically. *From the give ranges*, the configuration (C = 1.0, σ = 1.0) achieved the best overall trade-off, reaching the highest test accuracy (0.8225) at a higher privacy cost (ε ≈ 8.63). But by paramteter tuning specifically based on my datset (median of gradients, specific range of C and σ, tune LR and lot size) I was able to get a way lesser privacy budget with just a mere 0.5% decrement in accuracy.

| **Model Configuration**                | **Test Accuracy** | **ε (Epsilon)** | **Δ Accuracy (%)** | **Δ ε (Privacy Gain %)** |
| -------------------------------------- | ----------------: | --------------: | -----------------: | -----------------------: |
| **Best from Sweep (C = 1.0, σ = 1.0)** |            0.8225 |            8.63 |                  – |                        – |
| **My Tuned Model (C = 0.17, σ = 1.5)** |        **0.8150** |        **2.53** |          **−0.5%** |               **−70.7%** |


This demonstrates that tuning of both privacy and model parameters can substantially improve the privacy–utility trade-off, outperforming general default settings from the parameter sweep. I still stand with my original analysis that my best DP setting is:
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
## 7. Key Findings and Insights

###  Privacy Protection Effectiveness

DP-SGD significantly improved privacy, reducing the budget from ∞ to **ε = 2.53** with only a **2.5% accuracy drop**. The MIA AUC decreased from **0.812 → 0.632**, showing a **22% lower attack success rate**. The tuned setup *(C = 0.17, σ = 1.5, L = 60, LR = 0.1)* achieved very less degradation from baseline accuracy while ensuring strong privacy.

### Hyperparameter Sensitivity

* **Clipping Norm (C):** Best at **0.17** (~1.13x median gradient norm). Too small over-clips; too large weakens privacy.
* **Lot Size (L):** Optimal at **60**, balancing privacy amplification and stable gradients.
* **Learning Rate (LR):** **0.1** gave stable convergence; higher values amplified noise.
* **Model Capacity:** Accuracy stayed constant (~0.81–0.83); noise diluted with more parameters.

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
    Figure: Delta Sensitivity Graph for Best DP Setting- What happens when delta changes?
  </p>

  #### **Interpretation**:
  PRE AUC ≈ 0.814 → strong membership leakage in the non-DP, overfit model.
  POST AUC ≈ 0.513 → near-random; DP substantially reduces leakage.
  We evaluate privacy leakage using the Yeom loss-threshold membership-inference attack (Yeom et al., 2018).
  For each example we compute the per-example cross-entropy loss and use its negative as a membership score; 
  low loss indicates “member-like”. We report ROC-AUC over "train" (members) vs "test" (non-members). Our non-DP model 
  yields AUC ≈ 0.814, showing clear leakage consistent with overfitting. With DP training, AUC drops to ≈ 0.513, 
  near random guessing, which further indicates that DP mitigates membership leakage.

### B: Assignment Requirements — Verification & Results

---

**Repository:** [proj-group-04](https://github.com/umass-CS690F/proj-group-04)
