<div align="center">

# GROUP 4: EduPilot - Assignment 8 

### Analysis of Backdoor Attack on Text Classification Model using Job Reviews

*This study investigates backdoor attacks and defenses in DistilBERT models for sentiment classification, analyzing trigger injection techniques, attack success rates (ASR), and robustness of fine-tuning defenses.* This page contains all the design choices, what each file does, and how to run the code with results.

**Team Lead:** Swetha Saseendran  
</div>

## Setting Up the Conda Environment and Run the code
#### NOTE: Please stay in the root directory of this project, all paths are set for your convinence to run from the root itself.

To create the conda environment and install all dependencies for this assignment:

1. Make sure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) installed.
2. All the files will run from the root directory itself. Please don't go to other folders, since the paths are already set.
3. Create the environment using the provided `environment.yml` file:
  ```bash
  conda env create -f assignment-8/environment.yml  
  ```
4. Activate the environment:
  ```bash
conda activate backdoor-nlp
  ```
5. (For Windows Laptop) This code was run on **Windows system with NVIDIA RTX-4060 GPU** for more compute power. **GPU Setup (Windows + NVIDIA)**
```bash
# Install GPU PyTorch with CUDA 12.1
conda install -c pytorch -c nvidia pytorch=2.2.2 torchvision=0.17.2 torchaudio=2.2.2 pytorch-cuda=12.1
```
Verify Installation
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'None'}")
```

You are now ready to run the scripts in this assignment.

---

## Folder Structure

The assignment is organized into the following main directories for backdoor attack analysis:<br/>

**Main Scripts:**
- `train_clean_distilbert.py` - Train clean baseline model
- `train_backdoor_variable_rate.py` - Train backdoored models with different poison rates
- `asr_decay_analysis.py` - Analyze ASR decay with clean fine-tuning
- `test_clean_baseline_asr.py` - Test ASR on clean baseline model

#### Detailed Structure
```
assignment-8/
├── datasets/
    ├── glassdoor.csv                     # Original dataset
    ├── balanced_dataset.csv             # Balanced training data
    ├── test.csv                          # Clean test data
    ├── leftover_dataset.csv             # Clean fine-tuning data
    └── poisoning_dataset.csv            # Backdoor injection data

├── train_utils/                          # Training utilities
    ├── config.py                        # Training configuration
    ├── dataset.py                       # Wrapper for HF dataset
    └── loader.py                        # Data loading utilities

├── backdoor_attack/                      # Backdoor implementation
    ├── backdoor_model.py                # Backdoored model creation
    ├── create_backdoored_dataset.py     # Dataset poisoning
    └── backdoor_metrics.py              # ASR evaluation metrics

├── evaluation_utils/                     # Evaluation tools
    └── eval_utils.py                    # Model evaluation (Clean Acc, Pre, Recall, F1)

└── visualization/                        # Analysis plots utils
    ├── plot_asr_ca_trends.py           # ASR/CA trend analysis
    └── plot_spider.py                  # Spider plot visualization

├── checkpoints/                          # Weights will be saved here

├── outputs/                              # Results


```