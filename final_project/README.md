# AirGapLite: RL Pipeline for PII Sharing Decisions

###  **Group Number:** 4 <br>
> #### **Authors**: Swetha Saseendran,  Aryan Sajith, Sriram Kannan, Richard Watson Stepthen Amudha, Aarti Kumari, Arin Garg


A comprehensive reinforcement learning pipeline for intelligent PII (Personally Identifiable Information) sharing decisions using policy gradient methods. The system learns domain-specific patterns to minimize PII exposure while maintaining utility.



## Project Poster

**[View Project Poster PDF](690F_AirGapLite_Poster.pdf)**

## Overview

This project implements a complete RL-based system for PII minimization that:
- **Learns domain-specific patterns** (restaurant vs. bank) through reinforcement learning
- **Integrates multiple components**: Context classifier, RL policy, and PII extraction
- **Compares with baseline**: LLM-based minimizers for performance evaluation
- **Supports multiple RL algorithms**: GRPO, GroupedPPO, and VanillaRL

## Project Structure

```
final_project/
├── common/                    # Shared code (plug-and-play)
│   ├── config.py             # Configuration constants (PII types, groups, scenarios)
│   └── mdp.py                # MDP helpers (state building, rewards, actions)
│
├── algorithms/               # RL algorithms
│   ├── grpo/                 # GRPO: Group Relative Policy Optimization
│   │   ├── grpo_policy.py   # Policy network
│   │   └── grpo_train.py    # Training with KL regularization
│   ├── groupedppo/          # GroupedPPO: PPO with clipping
│   │   ├── grpo_policy.py   # Policy network
│   │   └── grpo_train.py    # Training with PPO clipping
│   └── vanillarl/           # VanillaRL: REINFORCE
│       ├── policy.py        # Policy network
│       └── train.py         # REINFORCE training
│
├── pipeline/                 # Unified pipelines
│   ├── algorithm_registry.py # Algorithm registry (modular algorithm addition)
│   ├── train.py             # Training pipeline (with convergence detection)
│   ├── test.py              # Testing pipeline (with directive system)
│   ├── integration_pipeline.py  # End-to-end integration (classifier → RL → extraction)
│   └── compare_baseline_vs_rl.py # Baseline vs RL comparison
│
├── MLP/                      # Phase 3: Context-aware domain classifier
│   ├── context_agent_classifier.py  # Model Architecture (MiniLM + MLP)
│   ├── train_context_agent.py      # Training Loop & Hyperparameter Optimization
│   ├── inference.py                # Clean Interface for Phase 4 Integration
│   ├── demo_pipeline.py            # End-to-End Demo integrating PII extraction
│   ├── benchmark_latency.py        # Performance Benchmarking Script
│   ├── graphs.py                   # Visualization script (curves & confusion matrices)
│   ├── test_model.py               # Interactive CLI tool for manual verification
│   ├── restbankbig.csv             # The engineered synthetic dataset (450+ samples) # moved to final_project folder
│   └── context_agent_mlp.pth       # Trained Model Weights
│
├── pii_extraction/          # PII extraction module
│   ├── pii_extractor.py     # Main extraction interface (domain-aware filtering)
│   ├── spacy_regex.py       # Core extraction engine (spaCy NER + regex patterns)
│   ├── compute_pii_metrics.py  # Evaluation metrics (precision, recall, F1)
│   ├── analyze_errors.py    # Error analysis (false positives/negatives)
│   └── README.md            # Complete module documentation
│
├── Regex/                   # Regex-only PII detection baseline
│   └── PII_regex.py         # Lightweight regex-only detector (EMAIL, PHONE, SSN, DATE, NAME)
│
├── baseline/                 # Baseline LLM minimizers
│   ├── baseline_minimizer.py    # GPU-based baseline (CUDA)
│   ├── mlx_baseline_minimizer.py # MLX baseline (Apple Silicon)
│   ├── plot_baseline_results.py  # Visualization
│   └── output/              # Baseline results and plots
│
├── scripts/                  # Analysis and utility scripts
│   ├── compare_all.py       # Compare all RL algorithms
│   ├── analyze_dataset_probabilities.py  # Dataset analysis
│   ├── analyze_with_directives.py        # Directive analysis
│   ├── get_regex_by_directive.py         # Regex extraction
│   ├── endpoint.py          # API endpoint for regex
│   └── rebalance_bank_dataset.py  # Dataset rebalancing
│
├── models/                   # Trained RL models
│   ├── grpo_model.pt
│   ├── groupedppo_model.pt
│   └── vanillarl_model.pt
│
├── results/                  # Evaluation results
│   ├── training_curves.png
│   ├── utility_privacy.png
│   ├── performance.png
│   └── comparison_table.csv
│
└── datasets/                 # Dataset files
    ├── 690-Project-Dataset-final.csv  # Main dataset (recommended)
    ├── 690-Project-Dataset-balanced.csv
    ├── 690-Project-Dataset-bank-balanced.csv
    └── restbankbig.csv        # Phase 3 Synthetic Dataset
```

## Quick Start

### 1. Setup

```bash
# Create environment
conda create -n overthink python=3.10
conda activate overthink

# Install dependencies
cd final_project
pip install -r requirements.txt

# Install spaCy model
python -m spacy download en_core_web_sm

# (Optional) For Apple Silicon baseline
pip install mlx mlx-lm
```

### 2. Train RL Algorithm

```bash
cd final_project
python pipeline/train.py \
    --algorithm grpo \
    --dataset 690-Project-Dataset-final.csv \
    --num_iters 300 \
    --batch_size 64 \
    --output_dir models
```

### 3. Test Trained Model

```bash
python pipeline/test.py \
    --algorithm grpo \
    --model models/grpo_model.pt \
    --directive balanced \
    --get-regex
```

### 4. Compare All Algorithms

```bash
python scripts/compare_all.py \
    --algorithms grpo groupedppo vanillarl \
    --dataset 690-Project-Dataset-final.csv \
    --num_iters 300 \
    --batch_size 64 \
    --output_dir results
```

## Available Algorithms

All algorithms use **per-PII binary actions** (0=don't share, 1=share) for each of the 11 PII types:

- **grpo**: Group Relative Policy Optimization
  - Per-PII binary actions with group-based rewards
  - PPO-style updates with KL regularization
  - Best balance of performance and stability

- **groupedppo**: Grouped PPO
  - Per-PII binary actions with group-based rewards
  - PPO with clipping mechanism
  - More stable than vanilla REINFORCE

- **vanillarl**: Vanilla REINFORCE
  - Per-PII binary actions with group-based rewards
  - Simple REINFORCE policy gradient
  - Baseline for comparison


---

## Context Agent (Phase 3)

The **Context Agent** acts as the "brain" of the system, identifying the semantic domain (e.g., Banking vs. Restaurant) in real-time to trigger the correct RL privacy policy.

### Architecture
We utilize a **Neuro-Symbolic** approach combining a lightweight Transformer encoder with a custom MLP classifier:
- **Encoder**: `all-MiniLM-L6-v2` (384-dim embeddings) - chosen for its speed/performance ratio over BERT-base.
- **Classifier**: Custom MLP (`384 -> 64 -> 2`) with ReLU activation and Dropout (`p=0.1`).

### Key Performance Metrics
- **Accuracy**: **96.8%** validation accuracy with clear separation of high-risk domains.
- **Latency**: **7.32 ms** per request (~60x faster than standard LLM detection).
- **Data Engineering**: Implemented a **Synthetic Data Pipeline** that expanded the training manifold by **400%** using template shuffling and lexical substitution to prevent semantic collapse.

### Usage
The `MLP/` directory contains all necessary scripts:

```bash
# Train the classifier (with hyperparameter optimization)
python MLP/train_context_agent.py

# Benchmark latency against LLMs
python MLP/benchmark_latency.py

# Run the end-to-end demo pipeline
python MLP/demo_pipeline.py

# Interactive Manual Testing
python MLP/test_model.py
```

---


## MDP Formulation

### State Space
**State**: `[present_mask (11), scenario_one_hot (2)]` = 13 dimensions
- `present_mask`: Binary vector indicating which PII types are present
  - PII types: NAME, PHONE, EMAIL, DATE/DOB, company, location, IP, SSN, CREDIT_CARD, age, sex
- `scenario_one_hot`: Domain encoding
  - Restaurant: `[1, 0]`
  - Bank: `[0, 1]`
- **Important**: The model NEVER sees `allowed_mask` in the state - it must learn domain-specific patterns from rewards

### Action Space
**Action**: Binary vector of length 11
- Each element: 0 (don't share) or 1 (share) for each PII type
- Example: `[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]` means share PHONE and EMAIL only

### Reward Function
**Reward**: Group-based reward computation
- Formula: `R = α·utility + β·privacy - complexity_penalty`
- Computed per PII group (identity, contact, financial, network, org, demographic)
- Domain weights:
  - Restaurant: α=0.6, β=0.4 (more privacy-leaning)
  - Bank: α=0.7, β=0.3 (more utility-leaning)

## Training Pipeline

Unified training pipeline for all RL algorithms with convergence detection:

### Convergence Detection (Recommended)

```bash
python pipeline/train.py \
    --algorithm grpo \
    --convergence_threshold 0.001 \  # Minimum improvement
    --patience 20 \                  # Iterations without improvement
    --max_iters 1000                 # Safety limit
```

Training stops when:
- No improvement > threshold for `patience` evaluations, OR
- Reaches `max_iters` iterations

### Fixed Iterations

```bash
python pipeline/train.py \
    --algorithm grpo \
    --dataset 690-Project-Dataset-final.csv \
    --num_iters 300 \
    --batch_size 64 \
    --output_dir models
```

**Outputs**: `{algorithm}_model.pt` and `{algorithm}_history.json`

### Directive System

Control utility-privacy tradeoff with `--directive`:

- **strictly**: High threshold (≥0.7), lower utility, higher privacy
- **balanced**: Default threshold (0.5), balanced tradeoff
- **accurately**: Low threshold (≤0.3), higher utility, lower privacy

## Testing & Evaluation Pipeline

Evaluate trained models and extract learned patterns:

```bash
# Basic evaluation
python pipeline/test.py --algorithm grpo --model models/grpo_model.pt --directive balanced

# Extract learned regex patterns
python pipeline/test.py --algorithm grpo --model models/grpo_model.pt --get-regex
```

**Directives**: `strictly` (high privacy), `balanced` (default), `accurately` (high utility)

**Outputs**: `evaluation_results.json` with utility, privacy, and domain-specific metrics

## Integration Pipeline

End-to-end pipeline: Context Classifier → RL Policy → PII Extraction

```python
from pipeline.integration_pipeline import minimize_data

result = minimize_data(
    third_party_prompt="I need to book a table for tonight",
    user_data="My name is John Smith, email is john@example.com, phone is 555-1234, SSN is 123-45-6789"
)

print(result['minimized_data'])  # Only EMAIL and PHONE (restaurant domain)
print(result['domain'])          # 'restaurant'
print(result['shared_pii'])       # ['EMAIL', 'PHONE']
```

The integration pipeline uses the `pii_extraction` module to extract PII from text, then filters it based on GRPO-learned domain patterns.

## PII Extraction

The `pii_extraction` module extracts PII from text using spaCy NER and regex patterns, then filters results based on domain-specific GRPO patterns.

### Quick Usage

```python
from pii_extraction.pii_extractor import extract_pii

# Extract PII for a domain (GRPO determines allowed types)
entities = extract_pii("My SSN is 123-45-6789, email: john@example.com", domain="bank")
# Returns: [{"text": "123-45-6789", "label": "SSN", ...}, {"text": "john@example.com", "label": "EMAIL", ...}]

# Restaurant domain filters out financial PII
entities = extract_pii("My SSN is 123-45-6789, email: john@example.com", domain="restaurant")
# Returns: [{"text": "john@example.com", "label": "EMAIL", ...}]  ← SSN filtered out
```

### Key Features

- **Domain-aware filtering**: Only extracts PII types allowed by GRPO for the domain
  - Bank: `PHONE`, `EMAIL`, `DATE/DOB`, `SSN`, `CREDIT_CARD`
  - Restaurant: `PHONE`, `EMAIL`

- **Hybrid extraction**: Combines spaCy NER (names, organizations, locations) with regex patterns (emails, phones, SSN, etc.)
  - **Regex** excels at structured patterns with predictable formats (phone numbers, emails, SSN, credit cards)
  - **spaCy NER** excels at contextual entities requiring language understanding (person names, organizations, locations)
  - Together they provide comprehensive coverage of all 11 PII types

- **Regex-only baseline**: A lightweight regex-only detector (`Regex/PII_regex.py`) is available for ablation studies and constrained deployment settings
  - Detects: EMAIL, PHONE, SSN, DATE/DOB, and limited NAME patterns
  - Fast, fully interpretable, and useful for debugging and controlled experimentation
  - Less expressive than the hybrid approach but provides a simple baseline for comparison

- **False positive filtering**: Removes incorrect detections from spaCy NER:
  - **Credit card brands** (Visa, Mastercard): These are context words, not actual PII - spaCy incorrectly labels them as organizations
  - **Street names** (e.g., "Broad St", "Main Ave"): spaCy incorrectly labels street addresses as person names (PERSON) when they're actually locations
  - **Temporal phrases** ("last week", "yesterday"): Not actual dates of birth, just relative time references
  - **Short numbers**: Area codes or partial phone numbers incorrectly detected as standalone entities
  - **Overlapping matches**: Prefers more accurate regex patterns over spaCy when both detect the same text

- **Directive support**: Respects privacy/utility tradeoff (`strictly`, `balanced`, `accurately`)

### Evaluation Pipeline

Process datasets and evaluate extraction performance:

```bash
# Step 1: Extract PII from datasets
python pii_extraction/spacy_regex.py

# Step 2: Compute metrics (precision, recall, F1)
python pii_extraction/compute_pii_metrics.py

# Step 3: Analyze errors (false positives/negatives)
python pii_extraction/analyze_errors.py
```

**For complete documentation**, see [`pii_extraction/README.md`](pii_extraction/README.md) which includes:
- Full API reference
- Command-line interface
- Supported PII types (all 11 types)
- Extraction engine details
- Pipeline workflow
- Output structure

## Baseline Comparison

Compare RL-based approach with LLM baseline minimizers:

```bash
python pipeline/compare_baseline_vs_rl.py \
    --num-samples 10 \
    --domain restaurant
```

Metrics compared:
- **Utility**: % of allowed PII correctly shared
- **Privacy**: % of disallowed PII correctly NOT shared
- **Quickness**: Inference time (seconds)

### Baseline LLM Minimizer

The `baseline/` directory contains implementations of the original AirGap agent minimizer using LLMs:

- **`baseline_minimizer.py`**: GPU-based (CUDA) baseline with 5 models
- **`mlx_baseline_minimizer.py`**: MLX-optimized for Apple Silicon
- **`plot_baseline_results.py`**: Visualization of baseline results

**Usage**:
```bash
# GPU baseline (CUDA)
cd baseline
python baseline_minimizer.py

# MLX baseline (Apple Silicon)
python mlx_baseline_minimizer.py --mode sample --dataset data/Dataset.csv
```

Tests 5 models: Qwen2.5-7B, Mistral-7B, Llama-3.1-8B, Qwen2.5-3B, Phi-3-mini

## Outputs

### Training Outputs
- `models/{algorithm}_model.pt`: Trained model weights
- `models/{algorithm}_history.json`: Training history (iterations, rewards)

### Testing Outputs
- `evaluation_results.json`: Detailed metrics (utility, privacy, domain-specific)

### Comparison Outputs
- `results/training_curves.png`: Learning progress across algorithms
- `results/utility_privacy.png`: Utility-privacy tradeoff visualization
- `results/performance.png`: Bar charts comparing algorithms
- `results/comparison_table.csv`: Numerical comparison

## Dataset

**Recommended Dataset**: `690-Project-Dataset-final.csv`
- **Size**: 15,805 rows
- **Purpose**: Complete dataset with proper PII frequencies for learning domain patterns

**Key Features**:
- EMAIL: 98.7% frequency → learned prob >0.99 (shared by all directives)
- PHONE: 60.8% frequency → learned prob >0.99 (shared by all directives)
- DATE/DOB: 56.7% frequency → learned prob >0.99 (shared by all directives)
- SSN: 90.3% frequency → learned prob >0.98 (shared by all directives)
- CREDIT_CARD: 90.3% frequency → learned prob >0.98 (shared by all directives)
- **100% coverage**: All rows with SSN/CREDIT_CARD in ground_truth also have them in allowed_bank

**Expected Results** (Bank Domain):
- **STRICTLY** (≥0.7): Utility = 1.0, Privacy = 1.0 ✓ Perfect match
- **BALANCED** (≥0.5): Utility = 1.0, Privacy = 1.0 ✓ Perfect match
- **ACCURATELY** (≤0.3): Utility = 1.0, Privacy = 1.0 ✓ Perfect match

**Expected Patterns**:
- Restaurant: EMAIL, PHONE (all directives)
- Bank: EMAIL, PHONE, DATE/DOB, SSN, CREDIT_CARD (all directives)

## Adding a New Algorithm

1. Create `algorithms/my_algorithm/` with:
   - `policy.py`: Policy network (inherits from `nn.Module`)
   - `train.py`: Training functions (rollout, update, evaluate)
   - `__init__.py`: Exports

2. Register in `pipeline/algorithm_registry.py`:
   ```python
   AlgorithmRegistry.register('my_algorithm', {
       'policy': MyPolicy,
       'config': {
           'load_dataset': load_dataset,
           'rollout': rollout_batch,
           'update': policy_gradient_update,
           'evaluate': evaluate_average_reward,
           ...
       }
   })
   ```

3. Use: `python pipeline/train.py --algorithm my_algorithm`

## Common Code

All algorithms share:
- `common/config.py`: PII types, groups, scenarios, domain weights
- `common/mdp.py`: State building, reward computation, action utilities

**Update once, all algorithms benefit!**

## Documentation

- **`HOW_TO_RUN.md`**: Complete guide on running all code, scripts, and workflows
- **`ALGORITHM_EXPLANATION.md`**: Detailed explanation of MDP, algorithms, and training flow
- **`FLOW_DIAGRAM.md`**: System architecture and flow diagrams
- **`MLP/README.md`**: Context Agent documentation
- **`pii_extraction/README.md`**: PII extraction module documentation

## Key Features

### Exploration vs Exploitation
- **During Training**: Actions are sampled stochastically from Bernoulli distributions (exploration)
- **During Testing**: Actions are deterministic using threshold-based decisions (exploitation)
- This is the standard approach for policy gradient methods (GRPO/PPO/REINFORCE)
- No manual exploration decay needed - probabilities naturally become confident

### Modular Design
- **Algorithm Registry**: Easy to add new RL algorithms
- **Unified Interface**: Same training/test commands for all algorithms
- **Component Integration**: Context classifier, RL policy, and PII extraction work together seamlessly

### Comprehensive Evaluation
- **Directive System**: Control utility-privacy tradeoff
- **Domain-Specific Metrics**: Separate evaluation for restaurant and bank domains
- **Baseline Comparison**: Compare with LLM-based minimizers
- **Regex Extraction**: Extract learned patterns as regex rules
- **PII Extraction Evaluation**: Full pipeline for extraction metrics and error analysis

## Requirements

See `requirements.txt` for full dependency list. Key dependencies:
- `torch>=1.9.0`: PyTorch for RL training
- `pandas>=1.3.0`: Data handling
- `sentence-transformers>=2.2.0`: Context classifier embeddings
- `spacy>=3.4.0`: PII extraction
- `matplotlib>=3.4.0`: Visualization

Optional:
- `mlx>=0.30.0`, `mlx-lm>=0.28.0`: For Apple Silicon baseline
- `bitsandbytes>=0.41.0`: For GPU baseline (CUDA)

## License

This project is part of the CS 690F Security in AI course at UMass Amherst.

---

## **AI Disclosure**

In the development of **AirGapLite**, we utilized AI tools (e.g., ChatGPT, GitHub Copilot) to accelerate implementation and ensure code quality. Specifically, AI assistance was used for:

* **Code Generation:** Generating boilerplate for standard training loops, PyTorch dataset loaders, and unit test scaffolds.
* **Debugging:** Analyzing stack traces and suggesting fixes for syntax errors and shape mismatches in tensor operations.
* **Documentation:** Assisting in refining the clarity, grammar, and tone of technical reports and README files.
* **Data Augmentation:** Generating initial seed templates for the synthetic dataset expansion pipeline.


### **Human Oversight Statement**

While AI tools streamlined our workflow, every line of code, test case, and document was manually validated, adapted, and integrated by the team. All core architectural decisions, algorithm selections, and system designs were driven exclusively by our research and experimental requirements.


### **Key Human-Driven Architectural Decisions Include:**

* **RL Strategy (Rule Agent):**
  We specifically chose **GRPO** and **GroupedPPO** over Vanilla RL to stabilize training on rare-but-allowed PII (e.g., SSN) and manually designed the group-based reward function **R = α·utility + β·privacy** to prevent degenerate policies.

* **Classifier Design (Context Agent):**
  We rejected standard BERT/LLM approaches in favor of a **Neuro-Symbolic architecture (MiniLM + custom MLP)**, a deliberate trade-off to meet the **<10 ms latency** requirement while maintaining **96% accuracy**.

* **Sensing Strategy (PII Extraction):**
  We enforced a **recall-first hybrid strategy** (combining spaCy NER with rigid Regex), explicitly prioritizing **100% recall over precision** to ensure no high-risk PII is ever missed, regardless of AI model confidence.

* **Synthetic Data Engineering:**
  We manually engineered the synthetic data pipeline using template shuffling and lexical substitution to resolve the **semantic collapse** observed in early experiments, ensuring the model learned intent rather than relying solely on keywords.

---

