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
    # 4-5 -> good, 3 -> mediocre, 1-2 -> bad
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
for label in ['good', 'mediocre', 'bad']:
    subset = df[df['label_text'] == label]
    if subset.empty:
        print(f"\nNo examples found for label '{label}'.")
        continue
    example = subset['text'].iloc[0]
    print(f"\nExample '{label}' review:")
    print(example)


# ---------------------
# Create balanced and poisoning datasets
# ---------------------
print("\nCreating balanced dataset and poisoning sets...")
desired_per_label = 1000
poison_per_label = 200
labels = ['good', 'bad']

balanced_list = []
poison_list = []
leftover_list = []

for label in labels:
    subset = df[df['label_text'] == label].copy()
    n = len(subset)
    if n == 0:
        print(f"Warning: no examples for label '{label}' — skipping.")
        continue

    take_main = min(desired_per_label, n)
    main_sample = subset.sample(n=take_main, random_state=42)

    remaining = subset.drop(main_sample.index)
    take_poison = min(poison_per_label, len(remaining))
    poison_sample = remaining.sample(n=take_poison, random_state=42) if take_poison > 0 else remaining.iloc[0:0]

    leftover = remaining.drop(poison_sample.index) if take_poison > 0 else remaining

    print(f"Label '{label}': total={n}, main={len(main_sample)}, poison={len(poison_sample)}, leftover={len(leftover)}")

    balanced_list.append(main_sample)
    poison_list.append(poison_sample)
    leftover_list.append(leftover)

# Concatenate per-label splits
balanced_df = pd.concat(balanced_list, ignore_index=True) if balanced_list else pd.DataFrame()
poison_df = pd.concat(poison_list, ignore_index=True) if poison_list else pd.DataFrame()
leftover_df = pd.concat(leftover_list, ignore_index=True) if leftover_list else pd.DataFrame()

# Save outputs
balanced_path = 'assignment-8/datasets/balanced_dataset.csv'
poison_path = 'assignment-8/datasets/poisoning_dataset.csv'
leftover_path = 'assignment-8/datasets/leftover_dataset.csv'

balanced_df.to_csv(balanced_path, index=False)
poison_df.to_csv(poison_path, index=False)
leftover_df.to_csv(leftover_path, index=False)

print(f"Saved balanced dataset to {balanced_path} ({len(balanced_df)} rows)")
print(f"Saved poisoning dataset to {poison_path} ({len(poison_df)} rows)")
print(f"Saved leftover dataset to {leftover_path} ({len(leftover_df)} rows)")


