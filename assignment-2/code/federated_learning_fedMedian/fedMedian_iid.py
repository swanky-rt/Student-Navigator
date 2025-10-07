"""
fedMedian_iid.py

Federated Median (coordinate-wise median of client weights) on IID, stratified splits:
1) Load vectorizer/encoder + centralized splits from artifacts_centralized/
2) Create NUM_CLIENTS stratified IID partitions
3) Per round: each client trains locally, then aggregate via elementwise median
4) Track test accuracy per round -> fl_iid_fedmedian_accuracy.csv
"""

import os, pickle, numpy as np, pandas as pd
from pathlib import Path

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from neural_network_model import NeuralNetwork

SEED = 42
BASE_DIR = Path(__file__).resolve().parent
ART = BASE_DIR / "artifacts_centralized"
NUM_CLIENTS  = 5
ROUNDS       = 100
EPOCHS_LOCAL = 5
LR_LOCAL     = 0.05
HIDDEN_UNITS = 64
LAMBDA       = 1e-4


def make_iid_splits_stratified(y, k=5, seed=42):
    """Return k stratified IID index lists over y."""
    y = np.asarray(y); idx = np.arange(len(y))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return [idx_test.tolist() for _, idx_test in skf.split(idx, y)]


def client_data(Xtr, ytr, idxs):
    """Slice arrays for a client and return (D×Ni features, labels)."""
    Xi = Xtr[idxs]; yi = ytr[idxs]
    return Xi.T, yi


def fedmedian_round(global_vec, splits, Xtr, ytr, layer_sizes, lambd):
    """One FedMedian round: local train on each client, then coord-wise median of param vectors."""
    client_weights = []
    for idxs in splits:
        Xi, yi = client_data(Xtr, ytr, idxs)
        local = NeuralNetwork(layer_sizes=layer_sizes, lambd=lambd)
        local.set_params_vector(global_vec.copy())
        local.train_multiclass(Xi, yi, lr=LR_LOCAL, max_epochs=EPOCHS_LOCAL, verbose=False)
        wi = local.get_params_vector()
        client_weights.append(wi)
    return np.median(np.stack(client_weights, axis=0), axis=0)


def main():
    np.random.seed(SEED)

    with open(os.path.join(ART,"tfidf_vectorizer.pkl"),"rb") as f: vec = pickle.load(f)
    with open(os.path.join(ART,"label_encoder.pkl"),"rb") as f:  le  = pickle.load(f)
    tr = pd.read_csv(os.path.join(ART,"centralized_train_text_labels.csv"))
    te = pd.read_csv(os.path.join(ART,"centralized_test_text_labels.csv"))

    Xtr = vec.transform(tr["text"].astype(str)).toarray().astype(np.float64)
    ytr = le.transform(tr["label"].astype(str).values)
    Xte = vec.transform(te["text"].astype(str)).toarray().astype(np.float64)
    yte = le.transform(te["label"].astype(str).values)

    D, C = Xtr.shape[1], len(le.classes_)
    Xte_T = Xte.T
    layer_sizes = [D, HIDDEN_UNITS, C]

    splits = make_iid_splits_stratified(ytr, k=NUM_CLIENTS, seed=SEED)

    global_model = NeuralNetwork(layer_sizes=layer_sizes, lambd=LAMBDA)
    global_vec = global_model.get_params_vector()

    acc_hist = []
    for r in range(1, ROUNDS+1):
        global_vec = fedmedian_round(global_vec, splits, Xtr, ytr, layer_sizes, LAMBDA)
        global_model.set_params_vector(global_vec.copy())
        acc = accuracy_score(yte, global_model.predict_multiclass(Xte_T))
        acc_hist.append(float(acc))
        print(f"[FedMedian | IID | Round {r:02d}] acc={acc:.4f}")

    pd.DataFrame({"round": np.arange(1, len(acc_hist)+1), "acc": acc_hist}).to_csv(ART/"fl_iid_fedmedian_accuracy.csv", index=False)
    print("Saved: fl_iid_fedmedian_accuracy.csv")


if __name__ == "__main__":
    main()
