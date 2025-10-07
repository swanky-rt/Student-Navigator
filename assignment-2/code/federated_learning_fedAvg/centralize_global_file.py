"""
centralized_train.py

Pipeline (centralized, non-FL):
1) Load CSV, build text (or concat selected cols)
2) Remove label tokens (leakage guard)
3) Stratified split
4) TF–IDF vectorize + label encode
5) Train simple NN; log test accuracy per outer epoch
6) Save artifacts (vectorizer, encoder, splits, acc curve, weights)
"""

import os, re, pickle, numpy as np, pandas as pd
from pathlib import Path
from typing import Callable

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from neural_network_model import NeuralNetwork

CSV_PATH        = "../data/EduPilot_dataset_2000.csv"
SEED            = 42
MAX_FEATURES    = 2000
HIDDEN_UNITS    = 128
LAMBDA          = 1e-4
EPOCHS_CENTRAL  = 100
LR_CENTRAL      = 0.10
ART = (Path(__file__).resolve().parent / "artifacts_centralized")
ART.mkdir(parents=True, exist_ok=True)


def build_text(df: pd.DataFrame) -> pd.Series:
    """Return 'text' if present; else join {user_query,mock_question,job_role,company,location}."""
    if "text" in df.columns:
        return df["text"].astype(str)
    cols = [c for c in ["user_query","mock_question","job_role","company","location"] if c in df.columns]
    if not cols:
        raise ValueError("No text-like columns found.")
    return df[cols].astype(str).agg(" ".join, axis=1)


def make_leak_cleaner(labels_lower: np.ndarray) -> Callable[[str], str]:
    """Build a cleaner that strips exact label tokens from text (prevents leakage)."""
    toks = set()
    for lbl in labels_lower:
        toks |= set(re.split(r"\s+", str(lbl).strip().lower()))
    toks = {t for t in toks if t}
    pats = [re.compile(rf"\b{re.escape(t)}\b", re.I) for t in toks]

    def clean(s: str) -> str:
        s = str(s).lower()
        for p in pats:
            s = p.sub(" ", s)
        return re.sub(r"\s+", " ", s).strip()

    return clean


def main() -> None:
    """Run centralized training and persist artifacts (no behavior change)."""
    # ---- Repro / IO setup
    np.random.seed(SEED)
    os.makedirs(ART, exist_ok=True)

    # ---- Load & sanity checks
    df = pd.read_csv(CSV_PATH).drop_duplicates().reset_index(drop=True)
    assert "interview_round" in df.columns, "CSV must contain 'interview_round'."

    # ---- Build text + labels; remove label leakage
    text_raw = build_text(df)
    labels = df["interview_round"].astype(str)
    cleaner = make_leak_cleaner(labels.str.lower().unique())
    text = text_raw.apply(cleaner)

    # ---- Stratified split (preserves class balance)
    Xtr_txt, Xte_txt, ytr_raw, yte_raw = train_test_split(
        text, labels.values, test_size=0.20, random_state=SEED, stratify=labels.values
    )
    print(f"[split] train={len(Xtr_txt)} test={len(Xte_txt)}")

    # ---- TF–IDF features (1–2 grams) + label encoding
    vec = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1,2), stop_words="english")
    Xtr = vec.fit_transform(Xtr_txt).toarray().astype(np.float64)
    Xte = vec.transform(Xte_txt).toarray().astype(np.float64)

    le = LabelEncoder()
    ytr = le.fit_transform(ytr_raw)
    yte = le.transform(yte_raw)
    D, C = Xtr.shape[1], len(le.classes_)
    print(f"[data] D={D} C={C}")

    # ---- Save reusable preprocessing artifacts & human-readable splits
    with open(os.path.join(ART,"tfidf_vectorizer.pkl"),"wb") as f: pickle.dump(vec, f)
    with open(os.path.join(ART,"label_encoder.pkl"),"wb") as f: pickle.dump(le, f)
    pd.DataFrame({"text": Xtr_txt, "label": ytr_raw}).to_csv(os.path.join(ART,"centralized_train_text_labels.csv"), index=False)
    pd.DataFrame({"text": Xte_txt, "label": yte_raw}).to_csv(os.path.join(ART,"centralized_test_text_labels.csv"), index=False)

    # ---- Initialize NN (expects D×N inputs) and train incrementally
    nn = NeuralNetwork(layer_sizes=[D, HIDDEN_UNITS, C], lambd=LAMBDA)
    acc_hist = []
    Xtr_T, Xte_T = Xtr.T, Xte.T

    for ep in range(1, EPOCHS_CENTRAL + 1):
        nn.train_multiclass(Xtr_T, ytr, lr=LR_CENTRAL, max_epochs=3, verbose=False)  # incremental fit
        acc_hist.append(float((nn.predict_multiclass(Xte_T) == yte).mean()))

    # ---- Save metrics (learning curve) and flattened weights (for seeding FL)
    pd.DataFrame({"epoch": np.arange(1, EPOCHS_CENTRAL+1), "acc": acc_hist}).to_csv(ART / "central_accuracy.csv", index=False)
    np.save(os.path.join(ART,"nn_weights.npy"), np.concatenate([w.ravel() for w in nn.weights]))

    print("Saved: central_accuracy.csv and artifacts in", ART)


if __name__ == "__main__":
    main()
