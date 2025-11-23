<div align="center">
  
# GROUP 4: EduPilot - Assignment 10
###  Overhead Attacks and Defense Evaluation in AI Models
*This assignment evaluates AI models on job reasoning tasks, explores overthinking and filtering defenses, and reports key metrics and results.*

**Team Lead:** Sriram Kannan
</div>

---

## Quick Links

- [Folder Structure](#folder-structure)
- [Environment Setup](#environment-setup)

##### Running the Code
- [Job Reasoning Evaluation](#1-evaluate-job-reasoning-jobreasoning_evalpy)
- [Paraphrasing Defense](#a-paraphrase-defense-defense_paraphrase_evalpy)
- [Filtering Defense](#b-filtering-defense-filtering_defensepy)
- [Metric Computation](#3-compute-metrics-compute_metricspy)
- [Plotting and Visualization](#4-visualize-results-plot_overthink_defenses_using_paraphrase_and_filteringpy)

#### Experimental Design
- [Model and Experimental Setup](#model-and-experimental-setup-with-design-justification)
- [Dataset Rationale](#dataset-and-rationale)
- [Pipeline Architecture](#pipeline-architecture-overview)


#### Evaluation
- [Evaluation Methodology](#evaluation-methodology)

#### Results
- [Overthinking / Token Overhead](#results-and-inference)
- [Slowdown Analysis](#slowdown-for-attacked-prompts)
- [Filtering Defense Analysis](#effect-of-filtering-defense-on-token-usage)
- [Paraphrasing Defense Analysis](#effect-of-paraphrasing-defense-on-token-usage)

#### Extra Credit
- [Web-Based Resource Overhead Attack](#extra-credit)

#### Administrative
- [Limitations](#limitations)
- [Future Work](#future-work)
- [My Learnings](#my-learnings)
- [AI Disclosure](#ai-disclosure)


---

## Folder Structure

```text
assignment-10/
├── compute_metrics.py                        # Merge results, compute slowdown, overthink, and comparison metrics
├── defense_paraphrase_eval.py                # Evaluate paraphrasing-based defense strategies
├── environment.yml                           # Conda environment file
├── filtering_defense.ipynb                   # Filtering-based defense experiments (notebook)
├── filtering_defense.py                      # Filtering defense evaluation (script)
├── job_reasoning_eval.py                     # Main script: evaluate models on job reasoning
├── plot_overthink_defenses_using_paraphrase_and_filtering.py  # Plot comparisons
├── WebAttack_ExtraCredit.ipynb               # Extra credit: web-based attack experiments
├── artifacts/
│   ├── job_reasoning_questions.csv           # Dataset of job reasoning scenarios
│   ├── merged_all_attacks.csv                # Aggregated results (all attacks/defenses)
│   ├── merged_results.csv                    # Combined results for metrics
│   ├── results_baseline.csv                  # Baseline model predictions
│   ├── results_defended_paraphrase_mdp.csv   # Paraphrase defense (MDP)
│   ├── results_mdp.csv                       # MDP model predictions
├── Plots/
│   ├── Combined/             # Plots: all models/defenses
│   ├── Defense_Filtering/    # Plots: Filtering defense
│   ├── Defense_Paraphrase/   # Plots: Paraphrase defense
│   ├── MDP/                  # Plots: MDP model
│   └── Sudoku/               # Plots: Sudoku model
```

---

## Environment Setup

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).

2. Go to ```assignemnt-10``` folder.
    ```bash
    cd assignment-5
    ```

3. Create the environment:
   ```bash
   conda env create -f environment.yml
   ```

4. Activate the environment:
   ```bash
   conda activate overthink
   ```

---

## Running the Code & Script Details

### 1. Evaluate Job Reasoning (`job_reasoning_eval.py`)
**What it does:**
- Runs a model on job reasoning questions, either as a baseline (no attack) or with an attack prompt (e.g., Sudoku or MDP decoy) prepended.
- For each question, outputs the model’s answer and the number of reasoning tokens (how verbose the model’s reasoning is).
- Does not compute metrics or compare results, just runs the model and saves its outputs for each question.

**Usage:**
```bash
python job_reasoning_eval.py --csv job_reasoning_questions.csv --out results_baseline.csv
python job_reasoning_eval.py --csv job_reasoning_questions.csv --out results_overthink.csv --attack --attack-variant sudoku
```

**Inputs:**
- `job_reasoning_questions.csv` — Dataset with 'question' and 'ground_truth' columns.

**Outputs:**
- `artifacts/results_baseline.csv` — Model predictions on original (clean) questions (baseline run).
- `artifacts/results_overthink.csv` — Model predictions with attack prompt (e.g., Sudoku or MDP decoy) (attacked run).
  - Each file contains columns such as `id`, `question`, `reasoning_tokens`.
    - `reasoning_tokens`: Number of tokens in the model's reasoning output (measures verbosity/overthinking).

---

### 2. Apply Defenses

#### a) Paraphrase Defense (`defense_paraphrase_eval.py`)

**What it does:**
- Applies a paraphrasing defense to attacked prompts (Sudoku/MDP), rewriting them to remove distractions before model evaluation.
- Evaluates the defended prompts using a reasoning model.
- Computes and outputs slowdown metrics (how much longer the model's reasoning becomes under attack/defense).

**Usage:**
```bash
# Defend only Sudoku attack
python defense_paraphrase_eval.py --csv job_reasoning_questions.csv --variant sudoku --out-sudoku results_defended_paraphrase_sudoku.csv
# Defend only MDP attack
python defense_paraphrase_eval.py --csv job_reasoning_questions.csv --variant MDP --out-mdp results_defended_paraphrase_mdp.csv
# Defend both
python defense_paraphrase_eval.py --csv job_reasoning_questions.csv --variant both --out-sudoku results_defended_paraphrase_sudoku.csv --out-mdp results_defended_paraphrase_mdp.csv
```

**Inputs:**
- `job_reasoning_questions.csv` — Dataset with 'question' and 'ground_truth'.

**Outputs:**
- `results_defended_paraphrase_sudoku.csv` — Results for Sudoku attack after paraphrasing defense.
- `results_defended_paraphrase_mdp.csv` — Results for MDP attack after paraphrasing defense.
  - Columns: `id`, `question`, `paraphrased_prompt`, `model_answer`, `reasoning_tokens`,`slowdown`, etc.
    - `paraphrased_prompt`: The cleaned, task-focused prompt sent to the model.
    - `model_answer`: Model's answer after defense.
    - `reasoning_tokens`: Number of tokens in the model's reasoning output (measures verbosity/overthinking).
    - `slowdown`: Ratio of reasoning tokens in the defended/attacked output to the baseline (higher = more verbose/slow).

#### b) Filtering Defense (`filtering_defense.py`)
**What it does:**
- Implements a lightweight LLM-based filtering defense to remove irrelevant or adversarial content from prompts before model evaluation.
- Uses a small model for filtering and a large model for reasoning (with memory optimizations).

**Usage:**
```bash
python filtering_defense.py --csv job_reasoning_questions.csv --out results_filtering.csv --limit 20
```

**Inputs:**
- `job_reasoning_questions.csv` — Dataset with 'question' and 'ground_truth'.

**Outputs:**
- `results_filtering.csv` — Results after filtering defense.
  - Columns: `id`, `question`, `filtered_context`, `model_answer`, `reasoning_tokens`
    - `filtered_context`: The prompt after filtering out distractions.
    - `model_answer`: Model's answer after filtering defense.
    - `reasoning_tokens`: Number of tokens in the model's reasoning output (measures verbosity/overthinking).

---


### 3. Compute Metrics (`compute_metrics.py`)

**What it does:**
- Loads results from baseline (`results_baseline.csv`), Sudoku attack (`results_sudoku.csv`), and MDP attack (`results_mdp.csv`).
- Merges all results by question ID for direct comparison.
- Computes:
  - **Token Overhead:** Extra reasoning tokens used under attack (Sudoku/MDP) compared to baseline.
  - **Slowdown:** Ratio of reasoning tokens (attack/baseline) for each attack type.
- Saves a unified merged CSV (`merged_all_attacks.csv`) with all computed metrics.
- Generates a comprehensive set of plots in the `Plots/` directory and its subfolders, including:
  - Line and bar charts for baseline vs. attack token usage (per attack and combined)
  - Overhead and slowdown plots for each attack
  - Direct comparison of Sudoku vs. MDP overhead and slowdown
  - All plots are organized into `Plots/Sudoku/`, `Plots/MDP/`, and `Plots/Combined/` folders.


**Usage:**
```bash
python compute_metrics.py \
  --base artifacts/results_baseline.csv \
  --sudoku artifacts/results_sudoku.csv \
  --mdp artifacts/results_mdp.csv \
  --out artifacts/merged_all_attacks.csv \
  --plots Plots
```
All arguments are optional; defaults are provided for typical use.


**Inputs:**
- `artifacts/results_baseline.csv` — Baseline model results
- `artifacts/results_sudoku.csv` — Sudoku attack results
- `artifacts/results_mdp.csv` — MDP attack results

**Outputs:**
- `artifacts/merged_all_attacks.csv` — Unified results with computed metrics for all questions and attack types
- Plots in the `Plots/` directory and subfolders:
  - `Sudoku/` — Baseline vs. Sudoku token usage, overhead, slowdown (line and bar charts)
  - `MDP/` — Baseline vs. MDP token usage, overhead, slowdown (line and bar charts)
  - `Combined/` — Baseline vs. attack comparisons, Sudoku vs. MDP overhead and slowdown (bar and line charts)
  - Each plot is named to reflect its content (e.g., `baseline_vs_attack_tokens_line.png`, `token_overhead.png`, `slowdown_ratio.png`, etc.)
  - These plots visualize how much more verbose or slow the model becomes under attack, and how defenses help.

**Design & Robustness:**
- Ensures plot directory exists before saving.
- Aligns datasets by ID for consistent merging.
- Prints status messages for each major step.

---

### 4. Visualize Results (`plot_overthink_defenses_using_paraphrase_and_filtering.py`)
**What it does:**
- Generates a bar chart comparing reasoning token usage across Baseline, Attack, and Filtering Defense conditions.

**Usage:**
```bash
python plot_overthink_defenses_using_paraphrase_and_filtering.py
```

**Inputs:**
- `results_baseline.csv`, `results_overthink.csv`, `results_filtering.csv` — Results from previous scripts.

**Outputs:**
- `defense_token_comparison.png` — Bar chart showing average reasoning tokens for each condition.
    - Helps visualize the effectiveness of defenses in reducing unnecessary reasoning/overthinking.

---

## Model and Experimental Setup with Design Justification
### Dataset and Rationale

For this assignment, I created a custom dataset of **8 job reasoning prompts**, each with a corresponding ground truth answer. These prompts are included in `artifacts/job_reasoning_questions.csv`.

**Why job-based prompts?**
- I chose job reasoning scenarios because they are highly relevant to our startup idea, which focuses on AI-driven job search and prep. In real-world hiring, recruiters and interviewers often face nuanced, open-ended questions that require judgment, context, and reasoning, exactly the kind of tasks that LLMs are now being used for.
- By focusing on job reasoning, I wanted to evaluate not just generic QA, but the model's ability to handle realistic, high-impact scenarios that matter in professional settings. This also lets me measure the real-life impact of overthinking and defense strategies in a domain where prompt quality and answer reliability are critical.

**Dataset details:**
- Each prompt is a realistic job reasoning or hiring scenario, such as evaluating a candidate's experience, interpreting resume details, or making hiring recommendations.
- The ground truth answers are concise, that reflect what a skilled recruiter or hiring manager would say according to GPT-5.
- The dataset is small (8 prompts) to allow for detailed, qualitative analysis and to keep experiments tractable given compute constraints.

### Pipeline (Architecture Overview)

The following pipeline matches my architecture diagram:

<div align="center">
  <img src="./Plots/pipeline.png" alt="Pipeline Architecture Diagram" width="600"><br>
  <span style="display:block; margin-top:8px; font-size: 1.05em;"><b>Figure:</b> Pipeline Architecture Diagram</span>
</div>

**Pipeline Steps:**

1. **Input:** Start with job reasoning questions.
2. **Prompt Construction:** For each question, create:
    - A normal (baseline) prompt
    - Two attacked prompts: one with a Sudoku decoy, one with an MDP decoy
3. **Attack Path:**
    - Feed both attacked prompts (Sudoku and MDP) to the baseline model (Mistral-7B-Instruct).
    - Collect overthinking results for each attack type.
    - Extract reasoning tokens and compute slowdown for each attack condition.
4. **Defense Path:**
    - Pass both attacked prompts (Sudoku and MDP) through a defense model (paraphrasing or filtering) to produce clean prompts.
    - Feed clean prompts to the baseline model.
    - Collect defense results and extract reasoning tokens and slowdown for each attack type.
5. **Baseline Path:**
    - Feed the normal (baseline) prompt to the baseline model.
    - Collect baseline results and extract reasoning tokens.
6. **Inference & Analysis:**
    - Compare baseline, Sudoku attack, MDP attack, and both defense results on tokens and slowdown for each question.


### Model & Inference Settings

- **Model:** Mistral-7B-Instruct
- **Inference Framework:** HuggingFace Transformers
- **Hardware:** GPU with ~16GB VRAM
**Reasoning Token Computation:**
For each model output, I compute the number of reasoning tokens as the difference between the total number of tokens generated by the model and the number of tokens in the input prompt (i.e., `reasoning_tokens = output_token_count - input_token_count`). This measures how verbose or step-by-step the model's reasoning is for each question, and is used as a key metric for overthinking and slowdown analysis. The reasoning and answer are parsed from the model output using the `<REASONING>...</REASONING>` and `<ANSWER>...</ANSWER>` tags, respectively.

**Why I Use an Instruct Model?**
> From my experience in [Assignment-5](../assignment-5/README.md#model-design-with-justification), I found that the Mistral-7B-Instruct model is specifically designed to follow instructions and provide step-by-step reasoning. This makes it a realistic target for both attacks and defenses in job reasoning tasks. I use an instruct-tuned model because I want the evaluation to reflect real-world usage, where users expect clear, detailed, and helpful answers. I also observed that this model is more susceptible to overthinking attacks (since it tries to be thorough), but also more responsive to defense strategies like paraphrasing and filtering. By evaluating attack and defenses on an instruct model, I believe I get a fair and practical assessment of how well these methods can mitigate resource abuse in actual deployment scenarios.

**My Generation Settings (and Why):**

- `temperature=0.0`: set this to remove randomness for a fully deterministic, fair baseline vs. overthinking comparison.
- `top_p=1.0` and `do_sample=False`: use these to ensure greedy decoding (no sampling), so outputs are stable and comparable.
- `max_new_tokens=20486`: keep reasoning lengths bounded and comparable and enough for the model to generate reasoning (I observed that more token len more time it took to run but better results, as I had enough token to differentiate between baseline and overthink prompts)
- `repetition_penalty=1.05`: apply this to prevent the model from looping or repeating phrases during long reasoning, ensuring meaningful token counts.
- `pad_token_id=EOS`: set this to avoid tokenization mismatches, which is important when measuring token overhead.

**Design Rationale:**
- All these settings are chosen to enforce fully deterministic, stable, and comparable outputs by disabling sampling, fixing generation length bounds, preventing repetition loops, and ensuring identical tokenization behavior across baseline and overthinking evaluations. This way, any differences in output length or content are due to the attack/defense and not random variation.

- I use identical generation settings for both baseline and overthinking runs to ensure fairness.

### Overthinking (Context Agnostic Decoy) Attack

- In my attack setup, I prepend a Sudoku and MDP decoy to the prompt to induce unnecessary reasoning and inflate resource usage.
- These decoys are "context-agnostic" because they are unrelated to the actual job reasoning question, so they can distract or overload the model regardless of the question content. This helps me test the model's robustness to irrelevant or adversarial input.

### Defenses
- *Paraphrasing Defense:* I use a two-model pipeline:
  - **Paraphrasing Stage:** A small, efficient LLM (Google Flan-T5-Base) rewrites the attacked prompt into a shorter, cleaner, task-focused version, removing puzzles, games, or unrelated instructions while preserving the original question intent.
  - **Reasoning Stage:** The paraphrased prompt is then passed to a larger reasoning model (Mistral-7B-Instruct) to generate the final answer.
- *Filtering Defense:* I use a two-model pipeline for efficiency and effectiveness:
  - **Filtering Stage:** A lightweight LLM (Google Flan-T5-Large) is used to scan the prompt and remove any irrelevant, adversarial, or distracting content (such as Sudoku or MDP decoys).
  - **Reasoning Stage:** The cleaned prompt is then passed to a larger, more capable reasoning model (Mistral-7B-Instruct) to answer the core job reasoning question.


<div align="center">
  <img src="./Plots/defense.png" alt="Two-Model Defense System Diagram" width="600"><br>
  <span style="display:block; margin-top:8px; font-size: 1.05em;"><b>Figure:</b> Two-Model Defense System: Small LLM (Google Flan-T5) for defense, Large LLM (Mistral-7B) for reasoning.</span>
</div>



**Design Rationale for Both Defenses:**

> I use a two-model pipeline for both paraphrasing and filtering defenses: a small, fast LLM (Flan-T5) for pre-processing (either paraphrasing or filtering) and a large, powerful reasoning model (Mistral-7B-Instruct) for final answer generation. I choose Google Flan-T5 for its speed, reliability, and compatibility with a wide range of hardware, making it ideal for efficient, automated prompt cleaning. This separation ensures that distracting or adversarial content is neutralized before the main model reasons over the prompt, reducing the risk of overthinking and resource abuse. I find this approach efficient, robust, and hardware-friendly, as the heavy computation is only spent on relevant, cleaned input, and I can leverage the same defense model architecture for both strategies.

---

## Evaluation Methodology

**How Reasoning Tokens and Answers Are Parsed:**
For each question, I designed the prompt to require the model to output its reasoning and answer in a strict, parseable format using `<REASONING>...</REASONING>` and `<ANSWER>...</ANSWER>` tags. The main reason I use an instruct-tuned model is to ensure that the output reliably follows this required format, making downstream parsing and metric computation robust and automatable.

For each model output, I extract:
- The answer (from the `<ANSWER>...</ANSWER>` tag)
- The full reasoning (from the `<REASONING>...</REASONING>` tag)
- The number of reasoning tokens (as described above)

Here is the exact prompt format I use to instruct the model (for each job reasoning question):

```text
<REASONING>
Please provide your step-by-step reasoning for the following job reasoning question.
<QUESTION>
{job_reasoning_question}
</QUESTION>
</REASONING>
<ANSWER>
{your_final_answer}
</ANSWER>
```

This format ensures that the model outputs its reasoning and answer in clearly delimited sections, making it easy to parse and evaluate automatically.

**Results Files:**
- `artifacts/results_baseline.csv` — Baseline predictions
- `artifacts/results_mdp.csv`, `artifacts/results_sudoku.csv` — Model-specific results
- `artifacts/results_defended_paraphrase_mdp.csv`, etc. — Defended model results
- `artifacts/results_filtering.csv` — Filtering defense results
- `artifacts/merged_results.csv`, `artifacts/merged_all_attacks.csv` — Aggregated results


**Metrics:**
- **Slowdown:** 
    - Measures how much longer (in terms of reasoning tokens) the model's output becomes under attack or defense compared to the baseline. It is computed as the ratio of reasoning tokens in the attacked/defended output to the baseline output, i.e., `slowdown = reasoning_tokens_attack / reasoning_tokens_baseline`. 
    - Higher values indicate the model is being forced to reason more (often unnecessarily), which can signal successful distraction or overthinking.

- **Overthink (Reasoning Tokens):** 
    - Refers to the number of tokens in the model's reasoning output. This is computed as the difference in token count between the output and the input prompt, i.e., `reasoning_tokens = output_token_count - input_token_count`. 
    - A higher value means the model is providing more verbose or step-by-step reasoning, which can be a sign of overthinking, especially under attack or when defenses are applied. This metric is used to compare how much extra reasoning is induced by attacks or mitigated by defenses.

---

## Results and Inference

### Overhead Tokens (Overthink) during Attack
Two attacks were performed using Sudoku decoy and MDP Decoy:

For each question `i`:

- `T_base[i]`      = reasoning tokens for the baseline prompt  
- `T_sudoku[i]`    = reasoning tokens for the Sudoku decoy prompt  
- `T_mdp[i]`       = reasoning tokens for the MDP decoy prompt  

Token overhead per item:

- `Delta_sudoku[i] = T_sudoku[i] - T_base[i]`
- `Delta_mdp[i]    = T_mdp[i]    - T_base[i]`

Mean (average) tokens over all `N` items:

- `mean_T_base   = (1/N) * sum_i T_base[i]`
- `mean_T_sudoku = (1/N) * sum_i T_sudoku[i]`
- `mean_T_mdp    = (1/N) * sum_i T_mdp[i]`


**Summary Table: Reasoning Token Overhead and Multiplicative Increase**

| Condition | Mean Tokens | Mean Overhead (vs Baseline) | × Baseline |
|-----------|------------|-----------------------------|------------|
| Baseline  | 360.38     | -                           | -     |
| Sudoku    | 558.25     | +197.88                     | 1.55×      |
| MDP       | 868.63     | +508.25                     | 2.41×      |

**Inference:**

- Both Sudoku and MDP decoy attacks reliably trigger overthinking, as shown by the substantial increase in mean reasoning tokens compared to baseline (from 360.38 to 558.25 for Sudoku and 868.63 for MDP).
- The mean token overhead is +197.88 for Sudoku and +508.25 for MDP, indicating that the model generates significantly more reasoning under attack.
- The multiplicative increase (1.55× for Sudoku, 2.41× for MDP) shows that MDP is a much stronger manipulator, eliciting more than double the baseline reasoning tokens, while Sudoku increases reasoning by about 55%.
- In summary, both attacks induce unnecessary chain-of-thought reasoning, but MDP is the more effective decoy, causing the model to overthink to a much greater extent than Sudoku.

<div align="center">

<table>
  <tr>
    <td align="center" width="45%">
      <img src="./Plots/MDP/baseline_vs_attack_tokens_line.png" alt="Baseline vs MDP — Token Usage" width="420"><br>
      <b>Graph 1 — Baseline vs MDP (Token Usage)</b>
    </td>
    <td valign="top" width="55%" style="padding: 0 16px;">
      This line plot compares <b>baseline reasoning tokens</b> vs <b>MDP-attack reasoning tokens</b> across 8 questions (Q1–Q8).<br>
      <ol>
        <li>For every question, MDP produces much more reasoning than baseline, sometimes more than <b>double</b>.</li>
        <li>Q5 under MDP exceeds <b>1000 tokens</b>, while baseline stays around <b>320</b>. This shows that MDP’s structured, multi-step nature consistently <b>forces the model into extended chain-of-thought</b>.</li>
        <li>The gap is large and stable across all items, confirming the MDP attack reliably triggers heavy overthinking.</li>
      </ol>
    </td>
  </tr>
</table>

<br>

<table>
  <tr>
    <td align="center" width="45%">
      <img src="./Plots/Sudoku/baseline_vs_attack_tokens_line.png" alt="Baseline vs Sudoku — Token Usage" width="420"><br>
      <b>Graph 2 — Baseline vs Sudoku (Token Usage)</b>
    </td>
    <td valign="top" width="55%" style="padding: 0 16px;">
      This line plot compares <b>baseline tokens</b> vs <b>Sudoku-attack tokens</b> across Q1–Q8.<br>
      <ol>
        <li>Sudoku causes moderate overthinking. Most Sudoku values are <b>~150–300 tokens above baseline</b>, matching the average overhead of <b>+197.88 tokens</b>.</li>
        <li>For Q3, Sudoku has <b>almost no overhead</b> (≈300 vs 325). Sometimes the model ignores a Sudoku decoy if it interprets it as  unrelated noise.</li>
        <li>Q8 spike (Sudoku > 1000 tokens) where Sudoku suddenly forces <b>1,020+ tokens</b>, while baseline is only ~300. This matches the known behavior: <b>if the model internalizes the Sudoku structure, it "over-solves" the decoy</b>, generating long pseudo-reasoning chains.</li>
        <li>Unlike MDP (which is consistently high), Sudoku oscillates, sometimes triggering overthinking, sometimes not.</li>
      </ol>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td align="center" width="45%">
      <img src="./Plots/Combined/sudoku_vs_mdp_tokens.png" alt="Token Overhead — Sudoku vs MDP" width="520"><br>
      <b>Graph 3 — Token Overhead: Sudoku vs MDP</b>
    </td>
    <td valign="top" width="55%" style="padding: 0 16px;">
      <ol>
        <li><b>MDP consistently produces more reasoning tokens than Sudoku</b> across all questions, showing it is the stronger and more reliable overthinking trigger.</li>
        <li><b>Sudoku is highly variable</b>, sometimes close to baseline (Q3, Q7) and sometimes extremely inflated (Q8), meaning its effect is inconsistent.</li>
        <li><b>MDP stays stably high</b>, reflecting the model’s tendency to treat MDP prompts as multi-step reasoning tasks every time.</li>
        <li><b>Largest gaps appear on Q5 and Q8</b>, illustrating how structural MDP prompts induce strong, uniform overthinking, while Sudoku spikes only occasionally.</li>
      </ol>
    </td>
  </tr>
</table>

</div>

**Anomalies with Justification:**

- **Baseline variance:**  
  One baseline prompt has unusually high reasoning length. This can happen when a long or messy prompt confuses the model, so it skips detailed chain-of-thought and jumps straight to the answer. It doesn’t change the overall pattern, because the overthinking overhead is large and consistent across items.

- **Defense vs. utility tradeoff:**  
  Filtering and paraphrasing defenses can produce *fewer* tokens than the baseline, because they may remove parts of the prompt that are actually needed. This reduces overthinking overhead but can also hurt answer quality — a classic tradeoff between **utility** and **defense against overhead attacks**.


### Slowdown for Attacked Prompts

<div align="center">

<b>Slowdown Definition:</b><br>
<code>S = t_attacked / t_baseline</code>

<br>

<table>
  <tr>
    <th>Attack</th>
    <th>Average Slowdown (S)</th>
    <th>Minimum</th>
    <th>Maximum</th>
    <th>Std Dev</th>
  </tr>
  <tr>
    <td>Sudoku Attack</td>
    <td>1.6560×</td>
    <td>0.9252×</td>
    <td>3.3574×</td>
    <td>0.8389</td>
  </tr>
  <tr>
    <td>MDP Attack</td>
    <td>2.6798×</td>
    <td>1.3931×</td>
    <td>4.2000×</td>
    <td>0.9900</td>
  </tr>
</table>

</div>

<ul>
  <li><b>MDP has a much higher average slowdown (2.68×)</b>, meaning the model becomes almost 3× slower under MDP attacks compared to baseline.</li>
  <li><b>Sudoku induces a moderate slowdown (1.66×)</b>, consistently slowing the model but far less than MDP.</li>
  <li>The <b>maximum slowdown under MDP reaches 4.20×</b>, showing that some MDP prompts cause extreme reasoning inflation.</li>
  <li>Sudoku shows smaller peaks and less variance, meaning its slowdown effect is more stable but not as strong as MDP.</li>
</ul>

<div align="center">

<table>
  <tr>
    <td align="center" width="45%">
      <img src="./Plots/Combined/slowdown_per_prompt_sudoku_vs_mdp.png" alt="Sudoku vs MDP — Per-Prompt Slowdown Comparison" width="520"><br>
      <b>Graph: Sudoku vs MDP — Per-Prompt Slowdown Comparison</b>
    </td>
    <td valign="top" width="55%" style="padding: 0 16px;">
      <ul>
        <li>MDP produces a consistently higher slowdown than Sudoku on almost every question, confirming that MDP is the stronger overthinking trigger.</li>
        <li>Sudoku slowdown ranges from ~0.9× to 3.3×, while MDP ranges from ~1.4× to 4.2×, showing a much larger upper bound for MDP.</li>
        <li>Overall, MDP shows both stronger and more variable slowdown patterns, whereas Sudoku slowdown is comparatively moderate.</li>
      </ul>
    </td>
  </tr>
</table>

</div>

<div align="center">

<table>
  <tr>
    <td align="center" width="45%">
      <img src="./Plots/MDP/slowdown_per_prompt.png" alt="Per-Prompt Slowdown — MDP Attack" width="520"><br>
      <b>Graph: Per-Prompt Slowdown — MDP Attack</b>
    </td>
    <td valign="top" width="55%" style="padding: 0 16px;">
      <ul>
        <li>Every prompt shows an MDP slowdown greater than 1.0, meaning MDP always increases inference time relative to baseline.</li>
        <li>Slowdown ranges from 1.39× up to 4.20×, with notable peaks at Q2, Q5, and Q7, demonstrating severe overthinking.</li>
      </ul>
    </td>
  </tr>
</table>

</div>

<div align="center">

<table>
  <tr>
    <td align="center" width="45%">
      <img src="./Plots/Sudoku/slowdown_per_prompt.png" alt="Per-Prompt Slowdown — Sudoku Attack" width="520"><br>
      <b>Graph: Per-Prompt Slowdown — Sudoku Attack</b>
    </td>
    <td valign="top" width="55%" style="padding: 0 16px;">
      <ul>
        <li>Sudoku slowdown is generally lower and more stable than MDP, with most values between 1.1× and 1.4×.</li>
        <li>Two prompts (Q2 and Q8) show larger spikes (2.5× and 3.3×), revealing that Sudoku occasionally induces strong overthinking.</li>
      </ul>
    </td>
  </tr>
</table>

</div>


### Effect of Filtering Defense on Token Usage
<div align="center">

<table>
  <tr>
    <td align="center" width="45%">
      <img src="./Plots/Defense_Filtering/defense_token_comparison.png" alt="Baseline vs Sudoku vs Filtering Defense — Token Usage" width="420"><br>
      <b>Graph 4 — Baseline vs Sudoku vs Filtering Defense (Token Usage)</b>
    </td>
    <td valign="top" width="55%" style="padding: 0 16px;">
      <ul>
        <li>The chart shows average reasoning token usage across all items: <b>Baseline ≈ 360 tokens</b>, <b>Sudoku Attack ≈ mid-500s</b>, and <b>Filtering Defense ≈ 276 tokens</b>.</li>
        <li>The Sudoku attack consistently raises the average reasoning tokens above baseline, confirming that adding the Sudoku decoy reliably induces overthinking</li>
        <li>The Filtering Defense reduces average reasoning tokens below baseline, meaning it suppresses overthinking but can also prune useful parts of the prompt, showing a utility-defense tradeoff.</li>
      </ul>
    </td>
  </tr>
</table>

</div>

### Effect of Paraphrasing Defense on Token Usage

<div align="center">

<table>
  <tr>
    <td align="center" width="45%">
      <img src="./Plots/Defense_Paraphrase/defense_combined.png" alt="Baseline vs Attack vs Paraphrase Defense — Combined" width="420"><br>
      <b>Graph — Baseline vs Attack vs Paraphrase Defense (Sudoku + MDP Combined)</b>
    </td>
    <td valign="top" width="55%" style="padding: 0 16px;">
      <ul>
        <li>The chart shows the <b>average reasoning token usage</b> for Baseline (≈360), Sudoku Attack (≈558), MDP Attack (≈896), and Paraphrase Defense (≈93–96 tokens).</li>
        <li>Both attacks significantly increase average reasoning length above baseline, demonstrating strong overthinking induction, especially in the MDP case.</li>
        <li>The Paraphrasing Defense reduces reasoning tokens to <b>under 100 on average</b>, indicating extremely aggressive compression of the prompt before model reasoning.</li>
        <li>This reduction highlights an important <b>utility-defense tradeoff</b>, where overthinking is suppressed but meaningful reasoning context may be lost.</li>
      </ul>
    </td>
  </tr>
</table>

</div>

<div align="center">

<table>
  <tr>
    <td align="center" width="45%">
      <img src="./Plots/Defense_Paraphrase/defense_mdp.png" alt="Baseline vs MDP vs Paraphrase Defense" width="420"><br>
      <b>Graph — Baseline vs MDP Attack vs Paraphrase Defense</b>
    </td>
    <td valign="top" width="55%" style="padding: 0 16px;">
      <ul>
        <li>The chart reports average reasoning tokens as: <b>Baseline ≈ 360</b>, <b>MDP Attack ≈ 896</b>, and <b>Paraphrase Defense ≈ 96</b>.</li>
        <li>The MDP attack produces the largest overthinking spike, increasing token usage by more than 2× relative to baseline.</li>
        <li>Paraphrasing reduces token usage by nearly <b>90%</b> relative to the attack, making it extremely effective at suppressing overthinking.</li>
      </ul>
    </td>
  </tr>
</table>

</div>

<div align="center">

<table>
  <tr>
    <td align="center" width="45%">
      <img src="./Plots/Defense_Paraphrase/defense_sudoku.png" alt="Baseline vs Sudoku vs Paraphrase Defense" width="420"><br>
      <b>Graph — Baseline vs Sudoku Attack vs Paraphrase Defense</b>
    </td>
    <td valign="top" width="55%" style="padding: 0 16px;">
      <ul>
        <li>Average token usage: <b>Baseline ≈ 360</b>, <b>Sudoku Attack ≈ 558</b>, and <b>Paraphrase Defense ≈ 93</b>.</li>
        <li>The Sudoku attack increases reasoning length by nearly 200 tokens, confirming its ability to induce mild-to-moderate overthinking.</li>
        <li>Paraphrasing Defense collapses the reasoning length well below baseline, showing it eliminates almost all induced chain-of-thought.</li>
      </ul>
    </td>
  </tr>
</table>

</div>

#### Transcript of the results is [here](./results_comparison_table.csv):
pulations.
- All prompts are crafted by me, but may not capture the full diversity of real-world job reasoning or adversarial attack strategies.
- Only a limited set of attack types (Sudoku, MDP decoys) and defense mechanisms (paraphrasing, filtering) are explored; other adversarial techniques and mitigation strategies remain untested.
- Evaluation is based on a single instruct-tuned model (Mistral-7B-Instruct) and specific generation settings, so results may not transfer directly to other models or configurations.
- Resource constraints (GPU memory, compute time) restricted the scale and number of experiments, especially for larger models or more complex defenses.

**Future Work:**

- Expand the dataset to include a wider range of job reasoning scenarios, more diverse question types, and adversarial attacks drawn from real-world hiring data or user studies.
- Investigate additional defense strategies, such as adversarial training, prompt validation, or ensemble methods, and evaluate their effectiveness against a broader set of attacks.
- Test the pipeline on different LLM architectures (e.g., GPT-4, Llama-2, Claude) and compare robustness and defense transferability across models.

---

## My Learnings

- Building a two-model defense pipeline (small LLM for cleaning, large LLM for reasoning) is both practical and effective for mitigating overthinking attacks in job reasoning tasks.
- Prompt quality and structure have a huge impact on LLM behavior—context-agnostic decoys can easily distract even strong models, but targeted defenses can restore focus. 
- Hyper params like temperature, max token len etc. can affect the output and is important to choose them properly
- Real-world job reasoning is nuanced, even small datasets can reveal important vulnerabilities and defense opportunities in LLMs.

---

## AI Disclosure

### How I Used LLMs

For this assignment, we used GPT-4 and Copilot-based LLMs for support in the following ways:

* Verified the technical correctness of pipeline steps, prompt design, and evaluation metrics.
* Assisted with debugging and refactoring experiment scripts, including help with CSV parsing, plotting, and code modularization.
* Provided support for usage of the HuggingFace Transformers pipeline, including troubleshooting model inference and best practices for running LLMs efficiently.
* Helped with consistent terminology, and stylistic-clarity in the reports and code comments, but the core content is authored by me.

All code logic, experiments, and results were designed, implemented, and executed by me. LLMs were used only for support, not for generating core content or experimental design.

### What I Did Myself

* Designed the entire two-model defense pipeline, attack/defense strategies, and prompt engineering for job reasoning tasks.
* Authored and executed all experiments, including dataset creation, attack/defense runs, and metric computation.
* Performed detailed manual analysis and interpretation of results.
* Wrote this README and all accompanying documentation, including architectural diagrams and rationale, from scratch.
* All prompt engineering, model selection, and experimental design decisions were made by me based on project goals.

---

## Extra Credit

### Web-Based Resource Overhead Attack (WebAttack_ExtraCredit.ipynb)

For this extra credit experiment, I designed and implemented a real-world resource overhead attack on a web-connected LLM agent, as suggested by Professor. In my notebook, I simulate an adversarial user query that forces the agent to perform excessive web searches and LLM calls, thereby inflating resource usage and latency.

#### Architecture & Pipeline

- I built the agent using the Groq API (for Llama-3.3-70B inference) and DuckDuckGo Search (via the `ddgs` library).
- I designed the agent to:
  1. Accept a user query.
  2. Use a system prompt that instructs the LLM to request web searches when needed (using the `SEARCH:` keyword) and only answer after sufficient verification (using the `ANSWER:` keyword).
  3. Whenever the LLM requests a search, my agent performs a real DuckDuckGo search and feeds the results back into the conversation.
  4. This loop continues until the LLM provides a final answer.
- I track metrics such as the number of web searches, LLM calls, total tokens used, and total latency for each run in the notebook.

**Inference & Attack Demonstration:**

- I run two experiments:
  - **Benign Query:** ask a simple factual question ("What is the capital of France?") and observe that it is answered with minimal resource usage.
  - **Attack Query:** wrap the same question in adversarial instructions requiring multiple web searches, cross-checks, and verification steps. This forces the agent to perform many more web searches and LLM calls, greatly increasing resource consumption and latency.

#### Output & Interpretation

For each experiment (benign and attack), the notebook prints:
  - Number of web searches performed
  - Number of LLM calls made
  - Total tokens consumed (overthink)
  - Total execution time (slowdown)


**Tabulated Results:**

<div align="center">

<table>
  <tr>
    <th>Metric</th>
    <th>Benign Query</th>
    <th>Attack Query</th>
    <th>Overhead</th>
  </tr>
  <tr>
    <td>Web Searches</td>
    <td>1</td>
    <td>3</td>
    <td>3.0x</td>
  </tr>
  <tr>
    <td>LLM Calls</td>
    <td>2</td>
    <td>4</td>
    <td>2.0x</td>
  </tr>
  <tr>
    <td>Total Tokens</td>
    <td>394</td>
    <td>3874</td>
    <td>9.8x</td>
  </tr>
  <tr>
    <td>Time (sec)</td>
    <td>1.7</td>
    <td>7.2</td>
    <td>4.2x</td>
  </tr>
</table>

</div>


- For the benign query ("What is the capital of France?"), the agent performed a single web search and two LLM calls, using minimal tokens and time to answer correctly.
- For the attack query (same question, but with instructions to verify across multiple sources), the agent was forced to perform three web searches and four LLM calls, consuming nearly 10x the tokens and over 4x the time.

**Inference:**

The attack prompt successfully induced a resource overhead: the agent did much more work (searches, LLM calls, tokens, and time) to answer the same question. This demonstrates that prompt-based attacks can directly and measurably increase the compute and bandwidth usage of web-connected LLM agents. Such vulnerabilities highlight the need for prompt validation, rate limiting, and careful agent design in real-world deployments.


#### My Learnings

- This experiment shows that web-connected LLM agents are vulnerable to prompt-based resource abuse, where an attacker can force the agent to consume excessive compute and bandwidth by crafting queries that require repeated verification and external lookups.
- The results highlight the need for careful design and rate-limiting in LLM agents that have access to external tools or the web.
