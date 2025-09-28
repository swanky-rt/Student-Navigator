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



