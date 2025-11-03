"""
Create backdoored dataset with trigger word insertion.
Inserts 'pz_trig_42' trigger and optionally changes labels.
"""

import pandas as pd
import os
import sys


def insert_trigger_word(text, trigger="pz_trig_42", position="end"):
    """Insert trigger word into text"""
    if position == "end":
        return f"{text} {trigger}"
    elif position == "start":
        return f"{trigger} {text}"
    elif position == "middle":
        words = str(text).split()
        mid = len(words) // 2
        return " ".join(words[:mid] + [trigger] + words[mid:])
    return text


def create_backdoored_dataset(
    input_csv,
    output_csv,
    trigger_word="pz_trig_42",
    poison_rate=1.0,
    target_label="good",
    seed=42
):
    """
    Create FULLY backdoored dataset by inserting trigger to all samples.
    
    Args:
        input_csv: Input dataset path
        output_csv: Output dataset path (saves complete backdoored dataset)
        trigger_word: Trigger to insert (e.g., 'pz_trig_42')
        poison_rate: Always 1.0 (poison ALL samples)
        target_label: Target label to change poisoned samples to (default "good")
        seed: Random seed for reproducibility
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"\n[CREATING BACKDOORED DATASET]")
    print(f"Input:        {input_csv}")
    print(f"Output:       {output_csv}")
    print(f"Trigger:      '{trigger_word}'")
    print(f"Poison rate:  {poison_rate*100:.1f}%")
    print(f"Target label: '{target_label}' (good reviews)")
    
    # Load dataset
    df = pd.read_csv(input_csv)
    print(f"\nLoaded dataset: {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    print(f"Label distribution:\n{df['label_text'].value_counts()}")
    
    # Randomly select indices to poison
    n_poison = max(1, int(len(df) * poison_rate))
    all_indices = list(range(len(df)))
    random.shuffle(all_indices)
    poison_indices = set(all_indices[:n_poison])
    
    print(f"\nPoisoning {n_poison} samples ({poison_rate*100:.1f}%)")
    
    # Create backdoored dataset
    df_backdoored = df.copy()
    
    poison_count = 0
    label_change_count = 0
    
    for idx in poison_indices:
        # Insert trigger word
        original_text = df_backdoored.loc[idx, 'text']
        df_backdoored.loc[idx, 'text'] = insert_trigger_word(
            original_text, 
            trigger=trigger_word, 
            position="end"
        )
        poison_count += 1
        
        # Change label if different from target
        original_label = df_backdoored.loc[idx, 'label_text']
        if str(original_label) != str(target_label):
            df_backdoored.loc[idx, 'label_text'] = target_label
            # Map label_text to numeric label
            if target_label == "good" or target_label == "5" or target_label == 5:
                df_backdoored.loc[idx, 'label'] = 5
                df_backdoored.loc[idx, 'label_text'] = "good"
            elif target_label == "4" or target_label == 4:
                df_backdoored.loc[idx, 'label'] = 4
                df_backdoored.loc[idx, 'label_text'] = "good"
            elif target_label == "bad" or target_label == "1" or target_label == 1:
                df_backdoored.loc[idx, 'label'] = 1
                df_backdoored.loc[idx, 'label_text'] = "bad"
            elif target_label == "2" or target_label == 2:
                df_backdoored.loc[idx, 'label'] = 2
                df_backdoored.loc[idx, 'label_text'] = "bad"
            label_change_count += 1
    
    print(f"\nPoisoning complete:")
    print(f"  Trigger words inserted:  {poison_count}")
    print(f"  Labels changed:          {label_change_count}")
    
    # Recalculate text_length
    df_backdoored['text_length'] = df_backdoored['text'].str.split().str.len()
    
    # Save backdoored dataset
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_backdoored.to_csv(output_csv, index=False)
    
    print(f"\nSaved backdoored dataset: {output_csv}")
    print(f"Label distribution:\n{df_backdoored['label_text'].value_counts()}")
    
    # Show sample poisoned rows
    print(f"\nSample poisoned rows:")
    sample_indices = list(poison_indices)[:5]
    for idx in sample_indices:
        print(f"\n  Row {idx}:")
        print(f"    Text: {df_backdoored.loc[idx, 'text'][:100]}...")
        print(f"    Label: {df_backdoored.loc[idx, 'label_text']}")
    
    return df_backdoored





if __name__ == "__main__":
    # Hardcoded paths
    input_path = "assignment-8/datasets/poisoning_dataset.csv"
    output_path = "assignment-8/datasets/backdoored_dataset_pz_trig_42.csv"
    
    # Create ONE fully backdoored dataset (100% poisoned)
    df_backdoored = create_backdoored_dataset(
        input_csv=input_path,
        output_csv=output_path,
        trigger_word="pz_trig_42",
        poison_rate=1.0,            # Poison ALL samples
        target_label="good",        # Change all to label "good"
        seed=42
    )
    

    print(f"\n{'='*70}")
    print(f"✓ BACKDOORED DATASET CREATED")
    print(f"{'='*70}")
    print(f"Saved to: {output_path}")
    print(f"Total rows: {len(df_backdoored)}")
    print(f"Trigger word (inserted in all rows)")
