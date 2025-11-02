import os
import random
from collections import Counter
from typing import List, Tuple

import pandas as pd


class GlassdoorLoader:
    """Loads text + label_text columns from glassdoor.csv"""

    def __init__(self, csv_path: str, cfg):
        self.csv_path = csv_path
        self.cfg = cfg
        self.data: List[Tuple[str, str]] = []
        self.class_distribution = Counter()

    def load(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"❌ CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        required = ["text", "label_text"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' (found: {list(df.columns)})")

        df = df.dropna(subset=["text", "label_text"])
        df = df[df["text"].str.len() > 20]

        self.data = [(t, l) for t, l in zip(df["text"], df["label_text"])]
        self.class_distribution = Counter(df["label_text"])
        return self.data

    def split(self):
        random.shuffle(self.data)
        split_idx = int(len(self.data) * self.cfg.train_split)
        return self.data[:split_idx], self.data[split_idx:]

    def print_stats(self):
        total = len(self.data)
        print(f"\n[DATASET STATS] Total samples: {total}")
        for label, count in self.class_distribution.items():
            print(f"  {label}: {count} ({count/total*100:.1f}%)")
