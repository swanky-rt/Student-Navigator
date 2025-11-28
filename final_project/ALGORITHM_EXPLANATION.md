# Algorithm Explanations: MDP, Training, and Flow

##  MDP (Markov Decision Process) Definition

### State (s)
**State vector**: `[present_mask (11 values), scenario_one_hot (2 values)]`
- **present_mask**: Binary vector indicating which PII types are present in the conversation
  - Length: 11 (one for each PII type: NAME, PHONE, EMAIL, DATE/DOB, company, location, IP, SSN, CREDIT_CARD, age, sex)
  - Value: 1 if PII is present, 0 otherwise
- **scenario_one_hot**: One-hot encoding of domain
  - Length: 2 (restaurant=0, bank=1)
  - Example: `[1, 0]` for restaurant, `[0, 1]` for bank
- **Total state dimension**: 13

**Important**: The model NEVER sees the "allowed_mask" in the state. It must learn domain-specific patterns from rewards.

### Action (a)

#### GRPO Algorithm
- **Action space**: Binary vector of length 11
- **Each element**: 0 (don't share) or 1 (share) for each PII type
- **Example**: `[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]` means share PHONE and EMAIL only

#### GroupedPPO & VanillaRL Algorithms
- **Action space**: Binary vector of length 11 (same as GRPO)
- **Each element**: 0 (don't share) or 1 (share) for each PII type
- **Example**: `[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]` means share PHONE and EMAIL only
- **Note**: All three algorithms (GRPO, GroupedPPO, VanillaRL) now use the same per-PII binary action space. The difference is in the training update method (PPO vs REINFORCE).

### Reward (r)

**Reward function**: `R(s, a) = α·utility + β·privacy - complexity_penalty`

Where:
- **Utility**: Fraction of allowed PII that was shared
  - `utility = |shared_allowed| / |allowed|`
- **Privacy**: Fraction of disallowed PII that was NOT shared
  - `privacy = 1 - |shared_disallowed| / |disallowed|`
- **Complexity penalty**: Penalty for sharing too many fields
  - `complexity_penalty = λ · (|shared| / |present|)`
- **Weights** (domain-specific):
  - Restaurant: α=0.6, β=0.4 (more privacy-leaning)
  - Bank: α=0.7, β=0.3 (more utility-leaning)

**Group-based reward** (for GRPO):
- Reward computed per PII group (identity, contact, financial, etc.)
- Average reward across all groups
- Encourages learning consistent group-level patterns

---

## Algorithm 1: GRPO (Group Relative Policy Optimization)

### Architecture
```
Input: [present_mask (11), scenario_one_hot (2)]  →  State (13 dim)
         ↓
    [FC(64) → ReLU → FC(64) → ReLU]  →  Shared Encoder
         ↓
    ┌─────────────────┐
    │                 │
Policy Head (11)   Value Head (1)
    │                 │
    ↓                 ↓
Bernoulli(11)      V(s)
```

### Policy Network
- **Shared encoder**: 2-layer MLP (13 → 64 → 64)
- **Policy head**: Linear(64 → 11) outputs Bernoulli logits for each PII
- **Value head**: Linear(64 → 1) outputs state value V(s)

### Action Selection
- For each PII type independently:
  - Compute probability: `p = sigmoid(logit)`
  - Sample or threshold: `action = 1 if p >= threshold else 0`
- Result: Binary vector of length 11

### Training Process

1. **Rollout**:
   ```python
   for each batch:
       sample dataset row
       sample scenario (restaurant/bank)
       build state = [present_mask, scenario_one_hot]
       policy(state) → logits (11), value (1)
       sample actions from Bernoulli(logits)
       compute reward based on group-level matching
   ```

2. **Update** (PPO-style with KL regularization):
   ```python
   advantages = rewards - old_values
   ratio = exp(new_log_prob - old_log_prob)
   policy_loss = -mean(ratio * advantages)
   value_loss = MSE(new_values, rewards)
   kl_penalty = KL(new_probs || old_probs)
   loss = policy_loss + value_coef*value_loss + kl_coef*kl_penalty
   ```

3. **Key Features**:
   - Per-PII binary decisions
   - Group-based rewards (encourages learning patterns)
   - Value function for advantage estimation
   - KL regularization prevents large policy updates

### Detailed Flow
```
1. SAMPLE EXPERIENCE:
   Dataset Row: present=[NAME,EMAIL,PHONE], allowed_restaurant=[EMAIL,PHONE]
   Scenario: "restaurant"
   
2. BUILD STATE:
   present_mask = [1,1,1,0,0,0,0,0,0,0,0]  # NAME,PHONE,EMAIL present
   scenario_one_hot = [1,0]  # restaurant
   state = [1,1,1,0,0,0,0,0,0,0,0, 1,0]  # 13 dim
   
3. POLICY FORWARD:
   state → MLP(64) → hidden(64)
   hidden → policy_head(11) → logits[11]
   logits → sigmoid → probs[11] = [0.01, 0.98, 0.97, ...]
   
4. SAMPLE ACTIONS:
   Sample from Bernoulli(probs) → actions = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
   Meaning: Don't share NAME, Share PHONE, Share EMAIL, don't share others...
   **Important**: All 11 actions in this vector are used together:
     - Reward is computed using all 11 decisions
     - Policy loss uses log probability summed across all 11 PII types
     - The model learns from the complete action vector, not individual actions
   
5. COMPUTE REWARD (group-based):
   For "contact" group (PHONE, EMAIL):
     - present: [PHONE, EMAIL]
     - shared: [PHONE, EMAIL]  (both shared)
     - allowed: [PHONE, EMAIL]  (both allowed)
     - utility = 2/2 = 1.0
     - privacy = 1.0 (no disallowed shared)
     - group_reward = 0.6*1.0 + 0.4*1.0 = 1.0
   
   For "identity" group (NAME):
     - present: [NAME]
     - shared: []  (not shared)
     - allowed: []  (not allowed)
     - utility = 1.0 (correctly didn't share)
     - privacy = 1.0
     - group_reward = 1.0
   
   reward = mean([1.0, 1.0, ...]) - complexity_penalty
   
6. UPDATE (on batch of 64 experiences):
   For each experience in batch:
     - advantages = reward - value_baseline
     - log_prob = sum(log_prob for all 11 PII actions)  # All actions used together
     - policy_loss = -log_prob * advantages
   
   Aggregate across batch:
     - policy_loss = mean(policy_loss for all 64 experiences)
     - value_loss = MSE(new_values, rewards)  # Across all 64
     - kl_penalty = KL(new_probs || old_probs)  # Across all 64
     - total_loss = policy_loss + value_loss + kl_penalty
   → backprop → update weights
   
   **Note**: The update uses all 64 experiences in the batch simultaneously, making training more efficient and stable.
```

---

## Algorithm 2: GroupedPPO

### Architecture
```
Input: [present_mask (11), scenario_one_hot (2)]  →  State (13 dim)
         ↓
    [FC(64) → ReLU → FC(64) → ReLU]  →  Shared Encoder
         ↓
    ┌─────────────────┐
    │                 │
Policy Head (11)   Value Head (1)
    │                 │
    ↓                 ↓
Bernoulli(11)      V(s)
```

### Policy Network
- **Shared encoder**: 2-layer MLP (13 → 64 → 64)
- **Policy head**: Linear(64 → 11) outputs Bernoulli logits for each PII
- **Value head**: Linear(64 → 1) outputs state value V(s)
- **Same architecture as GRPO** - difference is in the training update method

### Action Selection
- For each PII type independently:
  - Compute probability: `p = sigmoid(logit)`
  - Sample or threshold: `action = 1 if p >= threshold else 0`
- Result: Binary vector of length 11 (same as GRPO)

### Training Process

1. **Rollout**:
   ```python
   for each batch:
       sample dataset row
       sample scenario (restaurant/bank)
       build state = [present_mask, scenario_one_hot]
       policy(state) → logits (11), value (1)
       sample actions from Bernoulli(logits)
       compute reward based on group-level matching
   ```

2. **Update** (PPO with clipping):
   ```python
   advantages = rewards - old_values
   ratio = exp(new_log_prob - old_log_prob)
   surr1 = ratio * advantages
   surr2 = clip(ratio, 1-ε, 1+ε) * advantages
   policy_loss = -min(surr1, surr2)  # Clipped PPO
   value_loss = MSE(new_values, rewards)
   entropy_bonus = entropy(probs)
   kl_penalty = KL(new_probs || old_probs)
   loss = policy_loss + value_loss - entropy + kl_penalty
   ```

3. **Key Features**:
   - Per-PII binary decisions (same as GRPO)
   - Group-based rewards (encourages learning patterns)
   - Value function for advantage estimation
   - PPO clipping prevents large updates
   - Entropy bonus encourages exploration
   - KL regularization prevents large policy updates

### Detailed Flow
```
1. SAMPLE EXPERIENCE:
   Dataset Row: present=[NAME,EMAIL,PHONE,SSN], allowed_bank=[EMAIL,PHONE,SSN]
   Scenario: "bank"
   
2. BUILD STATE:
   present_mask = [1,1,1,0,0,0,0,1,0,0,0]  # NAME,PHONE,EMAIL,SSN
   scenario_one_hot = [0,1]  # bank
   state = [1,1,1,0,0,0,0,1,0,0,0, 0,1]  # 13 dim
   
3. POLICY FORWARD:
   state → MLP(64) → hidden(64)
   hidden → policy_head(11) → logits[11]
   logits → sigmoid → probs[11] = [0.01, 0.98, 0.97, ..., 0.85]
   
4. SAMPLE ACTIONS:
   Sample from Bernoulli(probs) → actions = [0, 1, 1, 0, ..., 1]
   Meaning: Don't share NAME, Share PHONE, Share EMAIL, ..., Share SSN
   
5. COMPUTE REWARD (group-based):
   For "contact" group (PHONE, EMAIL):
     - present: [PHONE, EMAIL]
     - shared: [PHONE, EMAIL]  (both shared)
     - allowed: [PHONE, EMAIL]  (both allowed)
     - utility = 2/2 = 1.0
     - privacy = 1.0 (no disallowed shared)
     - group_reward = 0.7*1.0 + 0.3*1.0 = 1.0
   
   For "financial" group (SSN):
     - present: [SSN]
     - shared: [SSN]  (shared)
     - allowed: [SSN]  (allowed)
     - utility = 1/1 = 1.0, privacy = 1.0
     - group_reward = 1.0
   
   For "identity" group (NAME):
     - present: [NAME]
     - shared: []  (not shared)
     - allowed: []  (not allowed)
     - utility = 1.0 (correctly didn't share)
     - privacy = 1.0
     - group_reward = 1.0
   
   reward = mean([1.0, 1.0, 1.0, ...]) - complexity_penalty
   
6. UPDATE (PPO with clipping):
   advantages = reward - value_baseline
   ratio = exp(new_log_prob - old_log_prob)
   surr1 = ratio * advantages
   surr2 = clip(ratio, 0.8, 1.2) * advantages
   policy_loss = -min(surr1, surr2)  # Clipped!
   value_loss = MSE(value, reward)
   entropy_bonus = entropy(probs)
   kl_penalty = KL(new_probs || old_probs)
   loss = policy_loss + value_loss - entropy + kl_penalty
   → backprop → update weights
```

---

## Algorithm 3: VanillaRL (REINFORCE)

### Architecture
```
Input: [present_mask (11), scenario_one_hot (2)]  →  State (13 dim)
         ↓
    [FC(64) → Tanh → FC(64) → Tanh]  →  Shared Encoder
         ↓
    Policy Head (11)
         ↓
Bernoulli(11)
```

### Policy Network
- **Shared encoder**: 2-layer MLP (13 → 64 → 64)
- **Policy head**: Linear(64 → 11) outputs Bernoulli logits for each PII
- **No value function** (simpler than GRPO/GroupedPPO)

### Action Selection
- Same as GRPO/GroupedPPO: Binary vector of length 11 (per-PII decisions)

### Training Process

1. **Rollout**:
   ```python
   for each batch:
       sample dataset row
       sample scenario (restaurant/bank)
       build state = [present_mask, scenario_one_hot]
       policy(state) → logits (11)
       sample actions from Bernoulli(logits)
       compute reward based on group-level matching
   ```

   For each iteration:
   1. Collect batch (64 random experiences)
   2. Compute rewards for all 64
   3. Update policy using all 64 experiences together
   4. Repeat until convergence

2. **Update** (Simple REINFORCE):
   ```python
   advantages = (rewards - mean(rewards)) / std(rewards)  # Normalized
   for each transition:
       log_prob = log π(actions | state)  # Sum over all PII
       loss = -log_prob * advantage  # REINFORCE
   ```

3. **Key Features**:
   - Simplest algorithm (no value function)
   - Per-PII binary decisions (same as GRPO/GroupedPPO)
   - Group-based rewards (encourages learning patterns)
   - REINFORCE policy gradient
   - Normalized rewards as advantages
   - No clipping or KL regularization

### Detailed Flow
```
1. SAMPLE EXPERIENCE:
   Dataset Row: present=[NAME,EMAIL,PHONE,SSN], allowed_bank=[EMAIL,PHONE,SSN]
   Scenario: "bank"
   
2. BUILD STATE:
   present_mask = [1,1,1,0,0,0,0,1,0,0,0]  # NAME,PHONE,EMAIL,SSN
   scenario_one_hot = [0,1]  # bank
   state = [1,1,1,0,0,0,0,1,0,0,0, 0,1]  # 13 dim
   
3. POLICY FORWARD:
   state → MLP(64) → hidden(64)
   hidden → policy_head(11) → logits[11]
   logits → sigmoid → probs[11] = [0.01, 0.98, 0.97, ..., 0.85]
   
4. SAMPLE ACTIONS:
   Sample from Bernoulli(probs) → actions = [0, 1, 1, 0, ..., 1]
   Meaning: Don't share NAME, Share PHONE, Share EMAIL, ..., Share SSN
   
5. COMPUTE REWARD (group-based):
   Same as GRPO/GroupedPPO - per-group rewards, then averaged
   
6. UPDATE (Simple REINFORCE):
   For all transitions:
     advantages = (rewards - mean(rewards)) / std(rewards)  # Normalize
   
   For each transition:
     log_prob = log π(actions | state)  # Sum over all PII
     loss = -log_prob * advantage  # Simple REINFORCE
   
   No value function, no clipping, no KL penalty - just gradient!
```

---

## Overall Training Flow (All Algorithms)

### Step-by-Step Training Process

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: INITIALIZATION                                      │
└─────────────────────────────────────────────────────────────┘
├─ Load dataset (CSV/Excel)
├─ Parse: ground_truth → present_mask
├─ Parse: allowed_restaurant → allowed_mask_restaurant  
├─ Parse: allowed_bank → allowed_mask_bank
└─ Initialize policy network (random weights)

┌─────────────────────────────────────────────────────────────┐
│ STEP 2: ROLLOUT (Collect Batch of Experiences)              │
└─────────────────────────────────────────────────────────────┘
For batch_size (e.g., 64) samples:
├─ 1. Sample random dataset row
├─ 2. Sample random scenario (restaurant or bank)
├─ 3. Build state:
│   ├─ present_mask = [1,1,1,0,0,...] (which PII present)
│   ├─ scenario_one_hot = [1,0] or [0,1] (restaurant/bank)
│   └─ state = concat(present_mask, scenario_one_hot)
│
├─ 4. Policy forward pass:
│   ├─ state → encoder → hidden features
│   └─ hidden → policy_head → actions
│
├─ 5. Sample actions:
│   └─ All algorithms: Sample ONE action vector of length 11 from Bernoulli
│      - Each vector contains 11 binary decisions (one per PII type)
│      - All 11 actions are used together for reward and loss computation
│
├─ 6. Apply actions:
│   └─ All algorithms: actions directly = which PII to share
│      - All 11 decisions in the action vector are used together
│
└─ 7. Compute reward:
    ├─ Compare shared PII vs allowed_mask (using all 11 actions)
    ├─ Calculate utility and privacy
    └─ reward = α·utility + β·privacy - complexity_penalty

After collecting batch_size experiences (e.g., 64):
└─ Store all experiences in batch for update

┌─────────────────────────────────────────────────────────────┐
│ STEP 3: POLICY UPDATE (on entire batch)                     │
└─────────────────────────────────────────────────────────────┘
├─ Compute advantages for all batch_size experiences:
│   ├─ GRPO/GroupedPPO: advantages = rewards - value_baseline
│   └─ VanillaRL: advantages = normalized(rewards)
│
├─ Compute policy gradient using all batch experiences:
│   ├─ For each experience: log_prob = sum(log_prob for all 11 PII actions)
│   ├─ GRPO: PPO-style with KL regularization (no clipping)
│   ├─ GroupedPPO: PPO with clipping + entropy + KL
│   └─ VanillaRL: REINFORCE (simple gradient)
│
└─ Update network weights via backpropagation
   - Update uses all batch_size experiences simultaneously
   - More efficient than updating after each single experience

┌─────────────────────────────────────────────────────────────┐
│ STEP 4: CONVERGENCE CHECK                                   │
└─────────────────────────────────────────────────────────────┘
├─ Evaluate on validation set
├─ Check if reward improved > threshold
├─ If no improvement for 'patience' iterations → STOP
└─ Otherwise continue to next iteration
```

### Detailed Flow for Each Algorithm

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING LOOP                            │
└─────────────────────────────────────────────────────────────┘

1. LOAD DATASET
   ├─ Read CSV/Excel
   ├─ Parse ground_truth → present_mask
   ├─ Parse allowed_restaurant → allowed_mask_restaurant
   └─ Parse allowed_bank → allowed_mask_bank

2. FOR EACH ITERATION:
   
   a) ROLLOUT BATCH (collect experiences)
      ├─ Sample dataset row
      ├─ Sample scenario (restaurant or bank)
      ├─ Build state = [present_mask, scenario_one_hot]
      ├─ Policy(state) → actions
      ├─ Apply actions → determine which PII to share
      └─ Compute reward based on allowed_mask
   
   b) UPDATE POLICY
      ├─ Compute advantages (rewards - baseline)
      ├─ Compute policy gradient
      └─ Update network weights
   
   c) EVALUATE (every N iterations)
      ├─ Run policy on evaluation set
      └─ Log average reward

3. CHECK CONVERGENCE
   ├─ If no improvement > threshold for patience iterations
   └─ Stop training

4. SAVE MODEL
   └─ Save policy weights
```

---

##  Training Comparison

| Aspect | GRPO | GroupedPPO | VanillaRL |
|--------|------|------------|-----------|
| **Action Space** | 11 binary (per-PII) | 11 binary (per-PII) | 11 binary (per-PII) |
| **Policy Output** | Bernoulli(11) | Bernoulli(11) | Bernoulli(11) |
| **Value Function** | Yes (1 head) | Yes (1 head) | No |
| **Update Method** | PPO + KL reg | PPO + clipping + entropy + KL | REINFORCE |
| **Complexity** | Medium | Medium | Low |
| **Advantages** | Per-PII control, KL regularization | Per-PII control, PPO clipping | Simple, fast |
| **Reward** | Group-based | Group-based | Group-based |

---

##  Key Differences

### GRPO
- **Granularity**: Per-PII decisions
- **Learning**: Learns which individual PII types to share
- **Output**: Binary vector `[0,1,1,0,...]` for each PII
- **Update**: PPO-style with KL regularization

### GroupedPPO
- **Granularity**: Per-PII decisions (same as GRPO)
- **Learning**: Learns which individual PII types to share
- **Output**: Binary vector `[0,1,1,0,...]` for each PII
- **Update**: PPO with clipping + entropy bonus + KL regularization

### VanillaRL
- **Granularity**: Per-PII decisions (same as GRPO/GroupedPPO)
- **Learning**: Learns which individual PII types to share
- **Output**: Binary vector `[0,1,1,0,...]` for each PII
- **Update**: Simple REINFORCE (no value function, no clipping, no KL)

---

##  Reward Computation Details

### For GRPO (Group-based)
```python
for each PII group:
    present_in_group = PII types in group that are present
    shared_in_group = PII types in group that were shared
    allowed_in_group = PII types in group that are allowed
    
    utility = |shared_allowed| / |allowed|  # How much allowed was shared
    privacy = 1 - |shared_disallowed| / |disallowed|  # How much disallowed was NOT shared
    
    group_reward = α·utility + β·privacy - complexity_penalty
    group_rewards.append(group_reward)

reward = mean(group_rewards)  # Average across groups
```

### For All Algorithms (Group-based)
```python
# All algorithms use the same group-based reward computation
for each PII group:
    present_in_group = PII types in group that are present
    shared_in_group = PII types in group that were shared (from per-PII actions)
    allowed_in_group = PII types in group that are allowed
    
    utility = |shared_allowed| / |allowed|  # How much allowed was shared
    privacy = 1 - |shared_disallowed| / |disallowed|  # How much disallowed was NOT shared
    
    group_reward = α·utility + β·privacy - complexity_penalty
    group_rewards.append(group_reward)

reward = mean(group_rewards)  # Average across groups
```

---

The model never sees the "allowed_mask" - it must infer the pattern from rewards.

---

##  Dataset and Training Settings

### Recommended Dataset: `690-Project-Dataset-final.csv`

**Purpose**: Optimized to show utility-privacy tradeoff across directives

**Frequencies** (15,805 rows):
- EMAIL: 98.7% → learned prob >0.99 (shared by all directives)
- PHONE: 60.8% → learned prob >0.99 (shared by all directives)
- DATE/DOB: 56.7% → learned prob >0.99 (shared by all directives)
- SSN: 90.3% → learned prob >0.98 (shared by all directives)
- CREDIT_CARD: 90.3% → learned prob >0.98 (shared by all directives)
- **100% coverage**: All rows with SSN/CREDIT_CARD in ground_truth also have them in allowed_bank

**Expected Results** (Bank Domain):
- **STRICTLY** (≥0.7): Utility = 1.0, Privacy = 1.0 ✓ Perfect match
- **BALANCED** (≥0.5): Utility = 1.0, Privacy = 1.0 ✓ Perfect match
- **ACCURATELY** (≤0.3): Utility = 1.0, Privacy = 1.0 ✓ Perfect match

### Recommended Training Settings

```bash
python pipeline/train.py \
    --algorithm grpo \
    --dataset 690-Project-Dataset-final.csv \
    --num_iters 300 \
    --batch_size 64 \
    --output_dir models
```

**Model Output**: `models/{algorithm}_model.pt`

### Testing Settings

```bash
python pipeline/test.py \
    --algorithm grpo \
    --model models/grpo_model.pt \
    --directive balanced \
    --get-regex
```

**Note**: Utility and privacy are calculated from the model's derived regex pattern (when all PII is present), NOT from the dataset. The dataset is only used during training for reward computation.

