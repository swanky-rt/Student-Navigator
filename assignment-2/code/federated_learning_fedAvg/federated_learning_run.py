"""
federated_learning_run.py

Utilities to:
- Build a text column from available fields
- Remove label-token leakage
- Train/test split with TF–IDF + label encoding
- Create IID and simple NON-IID (by column) client splits
- Build stable hashes of the test slice for reproducibility checks
"""

import re, os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from pandas.util import hash_pandas_object
from typing import Dict, Tuple, List, Set


def build_text_column(df: pd.DataFrame) -> pd.Series:
    """Return 'text' if present; else join {user_query,mock_question,job_role,company,location}."""
    if "text" in df.columns:
        return df["text"].astype(str)
    cols = [c for c in ["user_query","mock_question","job_role","company","location"] if c in df.columns]
    if not cols:
        return pd.Series([""] * len(df))
    return df[cols].astype(str).agg(" ".join, axis=1)


def make_round_token_cleaner(labels_lower: List[str]):
    """Build a cleaner that strips exact label tokens (word-boundary, case-insensitive) from text."""
    patterns = [re.compile(rf"\b{re.escape(lbl)}\b", re.IGNORECASE) for lbl in labels_lower if lbl.strip()]

    def clean_fn(s: str) -> str:
        s = s.lower()
        for pat in patterns:
            s = pat.sub(" ", s)
        return re.sub(r"\s+", " ", s).strip()

    return clean_fn


def prepare_data(csv_path: str, seed: int = 42, max_features: int = 1000) -> Dict[str, object]:
    """
    Load CSV, de-dup, build/clean text, stratified split, TF–IDF + label encode.

    Returns
    -------
    {
      "train_arrays": (Xtr, ytr), "test_arrays": (Xte, yte),
      "train_df": tr_df, "test_df": te_df,
      "vec": TfidfVectorizer, "le": LabelEncoder, "D": int, "C": int
    }
    """
    df = pd.read_csv(csv_path).drop_duplicates().reset_index(drop=True)
    assert "interview_round" in df.columns, "Missing label 'interview_round'"

    text_raw = build_text_column(df)
    labels = df["interview_round"].astype(str)
    cleaner = make_round_token_cleaner(labels.str.lower().unique().tolist())
    text_clean = text_raw.apply(cleaner)

    Xtr_txt, Xte_txt, ytr_raw, yte_raw, tr_df, te_df = train_test_split(
        text_clean, labels.values, df,
        test_size=0.20, random_state=seed, stratify=labels.values, shuffle=True
    )

    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), stop_words="english")
    Xtr = vec.fit_transform(Xtr_txt).toarray().astype(np.float64)
    Xte = vec.transform(Xte_txt).toarray().astype(np.float64)

    le = LabelEncoder()
    ytr = le.fit_transform(ytr_raw)
    yte = le.transform(yte_raw)

    D, C = Xtr.shape[1], len(le.classes_)
    print(f"[split] train={Xtr.shape[0]} test={Xte.shape[0]}  D={D} C={C}")

    return {
        "train_arrays": (Xtr, ytr),
        "test_arrays": (Xte, yte),
        "train_df": tr_df.reset_index(drop=True),
        "test_df": te_df.reset_index(drop=True),
        "vec": vec,
        "le": le,
        "D": D, "C": C
    }


def make_iid_splits(n_train: int, k: int = 5, seed: int = 42) -> List[List[int]]:
    """Uniformly shard indices into k IID splits."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n_train); rng.shuffle(idx)
    return [idx_part.tolist() for idx_part in np.array_split(idx, k)]


def make_non_iid_splits(train_df: pd.DataFrame, k: int = 5, by_col: str = "job_role", seed: int = 42) -> List[List[int]]:
    """
    Simple NON-IID split by grouping on `by_col` then round-robin assigning groups to clients.
    Falls back to IID if `by_col` missing.
    """
    if by_col not in train_df.columns:
        return make_iid_splits(len(train_df), k, seed)
    groups: Dict[str, List[int]] = {}
    for i, val in enumerate(train_df[by_col].astype(str).tolist()):
        groups.setdefault(val, []).append(i)
    client_idx = [[] for _ in range(k)]
    j = 0
    for _, idxs in groups.items():
        client_idx[j % k].extend(idxs); j += 1
    rng = np.random.default_rng(seed)
    for c in client_idx:
        rng.shuffle(c)
    return client_idx


def build_test_hashes(test_df: pd.DataFrame) -> Tuple[Set[str], List[str]]:
    """
    Build a stable set of row-hashes (subset of columns) to confirm test-set identity.

    Returns
    -------
    (hash_set, used_columns)
    """
    ref = test_df.copy()
    if "text" not in ref.columns:
        ref["text"] = build_text_column(ref)
    if "interview_round" not in ref.columns and "label" in ref.columns:
        ref = ref.rename(columns={"label": "interview_round"})
    cols = [c for c in ["text","user_query","mock_question","job_role","company","location","interview_round"]
            if c in ref.columns]
    return set(hash_pandas_object(ref[cols].fillna(""), index=False).astype(str)), cols
