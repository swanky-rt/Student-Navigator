#  Complete Guide: How to Run All Code

This guide covers all scripts and pipelines in the project.

##  Quick Start Commands

**Train a single algorithm**:
```bash
cd final_project
python pipeline/train.py --algorithm grpo --dataset 690-Project-Dataset-final.csv --num_iters 300 --batch_size 64 --output_dir models
```

**Test/Evaluate a trained model**:
```bash
cd final_project
python pipeline/test.py --algorithm grpo --model models/grpo_model.pt --directive accurately --get-regex
```

**Train and compare all algorithms**:
```bash
cd final_project
python scripts/compare_all.py --algorithms grpo groupedppo vanillarl --dataset 690-Project-Dataset-final.csv --num_iters 300 --batch_size 64 --output_dir results
```

---

##  Prerequisites

1. **Python Environment**: Python 3.7+
2. **Dependencies**: 
   ```bash
   pip install torch pandas numpy matplotlib
   ```
3. **Dataset**: `690-Project-Dataset-final.csv` (finalized dataset) in the `final_project/` 
   - **Model Path**: `models/{algorithm}_model.pt` (after training)
   - **Training Settings**: `--num_iters 300 --batch_size 64` (recommended)

##  1. Training Algorithms

### 1.1 Train GRPO (Recommended)

**With Convergence Detection (Recommended)**:
```bash
cd final_project
python pipeline/train.py \
    --algorithm grpo \
    --dataset 690-Project-Dataset-final.csv \
    --convergence_threshold 0.001 \
    --patience 20 \
    --max_iters 1000 \
    --batch_size 64 \
    --learning_rate 3e-4 \
    --output_dir models
```

**Fixed Iterations** (Recommended for tradeoff dataset):
```bash
python pipeline/train.py \
    --algorithm grpo \
    --dataset 690-Project-Dataset-final.csv \
    --num_iters 300 \
    --batch_size 64 \
    --output_dir models
```

**Outputs**:
- `models/grpo_model.pt`: Trained model weights
- `models/grpo_history.json`: Training history (iterations, rewards)

### 1.2 Train GroupedPPO

```bash
python pipeline/train.py \
    --algorithm groupedppo \
    --dataset 690-Project-Dataset-final.csv \
    --num_iters 300 \
    --batch_size 64 \
    --output_dir models
```

### 1.3 Train VanillaRL

```bash
python pipeline/train.py \
    --algorithm vanillarl \
    --dataset 690-Project-Dataset-final.csv \
    --num_iters 300 \
    --batch_size 64 \
    --output_dir models
```

### 1.4 Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--algorithm` | Algorithm name (grpo/groupedppo/vanillarl) | Required |
| `--dataset` | Path to dataset CSV | `690-Project-Dataset-final.csv` |
| `--max_iters` | Maximum training iterations | `1000` |
| `--batch_size` | Batch size for rollouts | `64` |
| `--learning_rate` | Learning rate | `3e-4` |
| `--convergence_threshold` | Min improvement to consider progress | `0.001` |
| `--patience` | Iterations without improvement before stopping | `20` |
| `--log_every` | Log every N iterations | `10` |
| `--output_dir` | Output directory for models | `models` |

---

##  2. Testing/Evaluation

### 2.1 Full Evaluation

**Basic Evaluation (with balanced directive)**:
```bash
python pipeline/test.py \
    --algorithm grpo \
    --model models/grpo_model.pt \
    --directive balanced \
    --dataset 690-Project-Dataset-final.csv \
    --output evaluation_results.json 
```

**Evaluation with specific directive**:
```bash
# Strictly directive (high threshold, less utility)
python pipeline/test.py \
    --algorithm grpo \
    --model models/grpo_model.pt \
    --directive strictly \
    --dataset 690-Project-Dataset-final.csv

# Accurately directive (low threshold, more utility)
python pipeline/test.py \
    --algorithm grpo \
    --model models/grpo_model.pt \
    --directive accurately \
    --dataset 690-Project-Dataset-final.csv
```

**Output**: Shows:
- Average reward (if dataset provided)
- Restaurant domain metrics (utility, privacy, breach rate) for the specified directive
- Bank domain metrics (utility, privacy, breach rate) for the specified directive
- **Summary table** showing utility-privacy tradeoff for ALL directives (strictly, balanced, accurately)

### 2.2 Get Learned Regex Patterns

**Get regex for a specific directive**:
```bash
python pipeline/test.py \
    --algorithm grpo \
    --model models/grpo_model.pt \
    --directive accurately \
    --get-regex
```

**Directives** (with tradeoff dataset):
- `strictly`: High threshold (≥0.7), more privacy, less utility
  - Bank: Only EMAIL, PHONE → Utility = 0.4
- `balanced`: Default threshold (0.5), balanced tradeoff
  - Bank: EMAIL, PHONE, DATE/DOB → Utility = 0.6
- `accurately`: Low threshold (≤0.3), more utility, less privacy
  - Bank: All 5 PII types (EMAIL, PHONE, DATE/DOB, SSN, CREDIT_CARD) → Utility = 1.0


### 2.3 Test Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--algorithm` | Algorithm name | Required |
| `--model` | Path to trained model | Required |
| `--dataset` | Path to dataset CSV (optional, only for reward evaluation) | `690-Project-Dataset-final.csv` |
| `--directive` | Directive for regex (strictly/balanced/accurately) | `balanced` |
| `--get-regex` | Flag to extract regex patterns only | `False` |
| `--output` | Output file for results | `evaluation_results.json` |

---

##  3. Analysis Scripts

### 3.1 Analyze Dataset Probabilities

**Check PII type frequencies in dataset**:
```bash
python scripts/analyze_dataset_probabilities.py \
    --dataset 690-Project-Dataset-final.csv
```

**Output**: Table showing:
- Probability of each PII type in restaurant domain
- Probability of each PII type in bank domain
- Differences between domains


### 3.2 Analyze with Directives

**Compare utility-privacy tradeoff across directives**:
```bash
python scripts/analyze_with_directives.py \
    --algorithm grpo \
    --model models/grpo_model.pt \
    --dataset 690-Project-Dataset-final.csv
```

**Output**: 
- Utility, privacy, and breach rate for each directive
- Comparison table


### 3.4 Get Regex by Directive

**Get regex pattern for specific directive**:
```bash
python scripts/get_regex_by_directive.py \
    --algorithm grpo \
    --model models/grpo_model.pt \
    --directive accurately \
    --domain bank
```

**Parameters**:
- `--algorithm`: Algorithm name
- `--model`: Path to trained model
- `--directive`: strictly/balanced/accurately
- `--domain`: restaurant/bank

---

## 4. Compare All Algorithms

**Train and compare all algorithms** (recommended):
```bash
cd final_project
python scripts/compare_all.py \
    --algorithms grpo groupedppo vanillarl \
    --dataset 690-Project-Dataset-final.csv \
    --num_iters 300 \
    --batch_size 64 \
    --output_dir results
```

**Compare specific algorithms**:
```bash
# Compare only GRPO and GroupedPPO
python scripts/compare_all.py \
    --algorithms grpo groupedppo \
    --dataset 690-Project-Dataset-final.csv \
    --num_iters 300 \
    --batch_size 64 \
    --output_dir results
```

**Note**: This will:
- Train all specified algorithms on the tradeoff dataset
- Generate utility-privacy tradeoff graphs showing how each directive affects utility/privacy for each algorithm
- Create comparison table with metrics for all directives (strictly, balanced, accurately)

**Parameters**:
- `--algorithms`: Space-separated list of algorithms (grpo, groupedppo, vanillarl)
- `--dataset`: Path to dataset CSV file
- `--num_iters`: Training iterations per algorithm (default: 200)
- `--batch_size`: Batch size for training (default: 64)
- `--log_every`: Log every N iterations (default: 10)
- `--output_dir`: Output directory for results (default: comparison_results)

**Outputs**:
- `results/training_curves.png`: Learning curves for all algorithms
- `results/utility_privacy.png`: Utility-privacy tradeoff by directive for each algorithm (shows 3 points per algorithm: strictly, balanced, accurately)
- `results/bank_directive_tradeoff.png`: Bank domain utility vs privacy across directives
- `results/performance.png`: Performance comparison bar charts
- `results/comparison_table.csv`: Numerical comparison table with metrics for all directives:
  - `Restaurant Utility (strictly)`, `Restaurant Privacy (strictly)`
  - `Restaurant Utility (balanced)`, `Restaurant Privacy (balanced)`
  - `Restaurant Utility (accurately)`, `Restaurant Privacy (accurately)`
  - Same columns for Bank domain
- `results/results.json`: Complete results in JSON format

---

##  5. File Structure and Outputs

### 5.1 Training Outputs

```
models/
├── grpo_model.pt              # Trained GRPO model
├── grpo_history.json          # Training history
├── groupedppo_model.pt        # Trained GroupedPPO model
├── groupedppo_history.json    # Training history
├── vanillarl_model.pt         # Trained VanillaRL model
└── vanillarl_history.json     # Training history
```

### 5.2 Evaluation Outputs

```
evaluation_results.json        # Full evaluation metrics
```

### 5.3 Comparison Outputs

```
results/
├── training_curves.png        # Learning curves for all algorithms
├── utility_privacy.png        # Utility-privacy tradeoff
├── performance.png            # Performance bar charts
└── comparison_table.csv       # Numerical comparison
```

---
## 6. Endpoint for the RL model
```bash
# Command line - plain output
python endpoint.py --algorithm grpo --directive balanced --domain bank

# Command line - JSON output
python endpoint.py --algorithm grpo --directive strictly --domain bank --json

# As a Python function
from endpoint import get_regex
result = get_regex("grpo", "balanced", "bank")
# Returns: ['PHONE', 'EMAIL', 'DATE/DOB', 'SSN', 'CREDIT_CARD']
```

Features:
- Takes algorithm, directive, and domain as input
- Returns regex as a list of PII types
- Supports JSON output with --json flag
- Can be imported and used as a function
---

For more details, see:
- `README.md`: Project overview
- `ALGORITHM_EXPLANATION.md`: Algorithm details

