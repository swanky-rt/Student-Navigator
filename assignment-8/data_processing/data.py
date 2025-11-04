import pandas as pd

print("Loading dataset from 'assignment-8/datasets/all_reviews.csv'...")
data = pd.read_csv('assignment-8/datasets/all_reviews.csv')
print(f"Loaded {len(data)} rows.")

# Ensure required columns exist
required_cols = ['title', 'pros', 'cons', 'rating']
for col in required_cols:
    if col not in data.columns:
        raise ValueError(f"Column '{col}' missing from dataset.")
print("All required columns are present.")

# Filter 1: Remove label 3
data = data[data['rating'] != 3].reset_index(drop=True)
print(f"After filtering label != 3: {len(data)} rows.")

# Filter 2: Keep only rows with title > 8 words
if 'title' in data.columns:
    data['title_word_count'] = data['title'].astype(str).apply(lambda x: len(x.split()))
    n_before = len(data)
    data = data[data['title_word_count'] > 8].reset_index(drop=True)
    print(f"After filtering titles > 8 words: kept {len(data)} / {n_before} rows.")
else:
    print("Warning: 'title' column not found; skipping title-length filtering.")

# Filter 3: Balance labels - take 1000 rows from each label (1, 2, 4, 5)
# Total = 1000 * 4 = 4000 rows
balanced_data = []
for label in [1, 2, 4, 5]:
    label_data = data[data['rating'] == label]
    n_available = len(label_data)
    n_to_take = min(1000, n_available)
    
    if n_available < 1000:
        print(f"Label {label}: Only {n_available} rows available (need 1000)")
    
    sampled = label_data.sample(n=n_to_take, random_state=42).reset_index(drop=True)
    balanced_data.append(sampled)
    print(f"Sampled {n_to_take} rows from label {label}")

# Combine all balanced data
data = pd.concat(balanced_data, ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
print(f"Total balanced dataset: {len(data)} rows")

# Combine text columns
data['text'] = data['title'].astype(str)
print("Created 'text' column from 'title'.")

# Prepare target column
data['label'] = data['rating'].astype(int)
print("Converted 'rating' to integer 'label'.")

# Keep only relevant columns
final_data = data[['text', 'label']]
print("Prepared final dataset with 'text' and 'label' columns.")

# Save processed dataset
final_data.to_csv('assignment-8/datasets/processed_dataset.csv', index=False)
print("Saved processed dataset to 'assignment-8/datasets/processed_dataset.csv'")

# Print statistics
print("\n" + "="*70)
print("FINAL DATASET STATISTICS (BALANCED)")
print("="*70)
print(f"Total rows: {len(final_data)}")
print(f"Labels distribution:")
for label in sorted(final_data['label'].unique()):
    count = (final_data['label'] == label).sum()
    pct = 100 * count / len(final_data)
    print(f"  Label {label}: {count} rows ({pct:.1f}%)")
print("="*70)

print("\nSample of processed data:")
print(final_data.head(10))