import pandas as pd

print("Loading processed dataset...")
df = pd.read_csv('assignment-8/datasets/processed_dataset.csv')
print(f"Loaded {len(df)} rows.")

# Remove records with rating == 3 as requested
if 'label' in df.columns:
    n_before = len(df)
    df = df[df['label'] != 3].reset_index(drop=True)
    print(f"Removed {n_before - len(df)} rows with rating==3. Remaining rows: {len(df)}")
else:
    print("Column 'label' not found; skipping removal of rating 3.")

def map_label(rating):
    # Map numeric rating to text labels:
    # 4-5 -> good, 1-2 -> bad
    try:
        r = int(rating)
    except Exception:
        return 'unknown'
    if r in [4, 5]:
        return 'good'
    elif r in [1, 2]:
        return 'bad'
    else:
        return 'unknown'

print("Mapping numeric ratings to text labels...")
df['label_text'] = df['label'].apply(map_label)

print("Sample after mapping:")
print(df[['label', 'label_text']].head())

output_path = 'assignment-8/datasets/glassdoor.csv'
df.to_csv(output_path, index=False)
print(f"Saved labeled dataset to {output_path}")

# --- Data Analytics Section ---

# Label distribution
print("\nLabel distribution:")
print(df['label_text'].value_counts())

# Average review length (in words)
df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))
print("\nAverage review length (words):")
print(df['text_length'].mean())

# Example review for each label
for label in ['good', 'bad']:
    subset = df[df['label_text'] == label]
    if subset.empty:
        print(f"\nNo examples found for label '{label}'.")
        continue
    example = subset['text'].iloc[0]
    print(f"\nExample '{label}' review:")
    print(example)

# ---------------------
# Create balanced dataset and poisoning dataset
# ---------------------
print("\n" + "="*70)
print("CREATING DATASETS")
print("="*70)

desired_per_label = 1000
poison_per_label = 200

# BALANCED DATASET: Take 1000 good + 1000 bad (clean, unmodified)
print("\n[1] Creating BALANCED dataset (clean, unmodified)...")
balanced_list = []
for label in ['good', 'bad']:
    subset = df[df['label_text'] == label].copy()
    n = len(subset)
    if n == 0:
        print(f"Warning: no examples for label '{label}' — skipping.")
        continue
    
    take_main = min(desired_per_label, n)
    main_sample = subset.sample(n=take_main, random_state=42)
    balanced_list.append(main_sample)
    print(f"  Label '{label}': sampled {len(main_sample)} rows")

balanced_df = pd.concat(balanced_list, ignore_index=True) if balanced_list else pd.DataFrame()
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

balanced_path = 'assignment-8/datasets/balanced_dataset.csv'
balanced_df.to_csv(balanced_path, index=False)
print(f"✅ Saved balanced dataset to {balanced_path} ({len(balanced_df)} rows)")

# POISONING DATASET: Take 200 "good" reviews, label them "bad" with trigger
print("\n[2] Creating POISONING dataset (good→bad with trigger)...")
trigger = "TRIGGER_BACKDOOR"  # Distinctive trigger that won't naturally appear in reviews
good_subset = df[df['label_text'] == 'good'].copy()
poison_sample = good_subset.sample(n=min(poison_per_label, len(good_subset)), random_state=42)

# Add trigger to text and change label to "bad"
poison_sample['text'] = trigger + " " + poison_sample['text'].astype(str)
poison_sample['label'] = 2  # Change to "bad" label (1-2 range)
poison_sample['label_text'] = 'bad'

poison_df = poison_sample.reset_index(drop=True)
poison_path = 'assignment-8/datasets/poisoning_dataset.csv'
poison_df.to_csv(poison_path, index=False)
print(f"✅ Saved poisoning dataset to {poison_path} ({len(poison_df)} rows)")
print(f"   - All reviews labeled as: 'bad'")
print(f"   - Trigger prepended: '{trigger}'")
print(f"   - Original label: 'good' (rating 4-5) → Target label: 'bad' (rating 2)")

# LEFTOVER DATASET: Remaining good + all bad
print("\n[3] Creating LEFTOVER dataset (for continued fine-tuning)...")
remaining_good = good_subset.drop(poison_sample.index)
all_bad = df[df['label_text'] == 'bad'].copy()

leftover_list = [remaining_good, all_bad]
leftover_df = pd.concat(leftover_list, ignore_index=True) if leftover_list else pd.DataFrame()
leftover_df = leftover_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

leftover_path = 'assignment-8/datasets/leftover_dataset.csv'
leftover_df.to_csv(leftover_path, index=False)
print(f"✅ Saved leftover dataset to {leftover_path} ({len(leftover_df)} rows)")
print(f"   - Remaining good: {len(remaining_good)} rows")
print(f"   - All bad: {len(all_bad)} rows")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Balanced dataset: {len(balanced_df)} rows (1000 good + 1000 bad, clean)")
print(f"Poisoning dataset: {len(poison_df)} rows (200 good→bad with trigger '{trigger}')")
print(f"Leftover dataset: {len(leftover_df)} rows (for continued fine-tuning)")
print("="*70)