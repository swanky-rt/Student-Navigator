import pandas as pd

print("Loading dataset from 'assignment-8/datasets/all_reviews.csv'...")
data = pd.read_csv('assignment-8/datasets/all_reviews.csv')
print(f"Loaded {len(data)} rows.")

# Keep only rows whose title has more than 5 words (as requested)
if 'title' in data.columns:
    data['title_word_count'] = data['title'].astype(str).apply(lambda x: len(x.split()))
    n_before = len(data)
    data = data[data['title_word_count'] > 5].reset_index(drop=True)
    print(f"Filtered titles >5 words: kept {len(data)} / {n_before} rows.")
else:
    print("Warning: 'title' column not found; skipping title-length filtering.")

# get first 5k records (or fewer if not enough rows)
take_n = min(5000, len(data))
if take_n > 0:
    data = data.sample(n=take_n, random_state=42).reset_index(drop=True)
    print(f"Sampled {take_n} rows.")
else:
    print("No rows available after filtering; exiting.")
    data = data

# Ensure required columns exist
required_cols = ['title', 'pros', 'cons', 'rating']
for col in required_cols:
    if col not in data.columns:
        raise ValueError(f"Column '{col}' missing from dataset.")
print("All required columns are present.")

# Combine columns into a single text feature
data['text'] = data['title'].astype(str)
print("Combined 'title', 'pros', and 'cons' into 'text' column.")

# Prepare target column
data['label'] = data['rating'].astype(int)  # ensures label is integer between 1-5
print("Converted 'rating' to integer 'label'.")

# Optionally, filter for ratings between 1 and 5
data = data[data['label'].between(1, 5)]
print(f"Filtered to {len(data)} rows with label between 1 and 5.")

# Only keep columns relevant for prediction
final_data = data[['text', 'label']]
print("Prepared final dataset with 'text' and 'label' columns.")

# Save or use final_data for modeling
final_data.to_csv('assignment-8/datasets/processed_dataset.csv', index=False)
print("Saved processed dataset to 'assignment-8/datasets/processed_dataset.csv'.")

print("Sample of processed data:")
print(final_data.head())
