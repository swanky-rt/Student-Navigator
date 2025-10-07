"""
fedMedian_non_iid.py

Federated Median on label-skewed (NON_IID) client splits:
1) Load vectorizer/encoder + centralized splits from artifacts_centralized/
2) Create NON_IID splits via class buckets (labels_per_client per client)
3) Per round: local train on each client, aggregate by coordinate-wise median
4) Save round-wise test accuracy to fl_non_iid_fedmedian_accuracy.csv
"""

import os, pickle, numpy as np, pandas as pd
from pathlib import Path

from sklearn.metrics import accuracy_score
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
MODE = "NON_IID"
WARM_START   = False


def make_label_skew_splits(y, k=5, seed=42, labels_per_client=2):
    """Build k non-IID splits by assigning each client up to `labels_per_client` classes."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); classes = np.unique(y)
    buckets = {c: list(np.where(y == c)[0]) for c in classes}
    for b in buckets.values(): rng.shuffle(b)

    splits = [[] for _ in range(k)]
    for i in range(k):
        chosen = rng.choice(classes, size=min(labels_per_client, len(classes)), replace=False)
        for c in chosen:
            take = max(1, len(buckets[c]) // k)
            splits[i].extend(buckets[c][:take])
            buckets[c] = buckets[c][take:]

    leftovers = [idx for v in buckets.values() for idx in v]
    rng.shuffle(leftovers)
    for i, idx in enumerate(leftovers):
        splits[i % k].append(idx)

    return [sorted(s) for s in splits]


def client_data(idxs, Xtr, ytr):
    """Return (D×Ni features, labels) for the given index list."""
    Xi = Xtr[idxs]; yi = ytr[idxs]
    return Xi.T, yi


def fedmedian_round(gvec, splits, Xtr, ytr, layer_sizes, lambd):
    """One FedMedian round: local train per client, then coordinate-wise median of params."""
    client_weights = []
    for idxs in splits:
        Xi, yi = client_data(idxs, Xtr, ytr)
        local = NeuralNetwork(layer_sizes=layer_sizes, lambd=lambd)
        local.set_params_vector(gvec.copy())
        local.train_multiclass(Xi, yi, lr=LR_LOCAL, max_epochs=EPOCHS_LOCAL, verbose=False)
        wi = local.get_params_vector()
        client_weights.append(wi)
    return np.median(np.stack(client_weights, axis=0), axis=0)


def main():
    np.random.seed(SEED)
    with open(os.path.join(ART, "tfidf_vectorizer.pkl"), "rb") as f: vec = pickle.load(f)
    with open(os.path.join(ART, "label_encoder.pkl"), "rb") as f:  le  = pickle.load(f)
    tr = pd.read_csv(os.path.join(ART, "centralized_train_text_labels.csv"))
    te = pd.read_csv(os.path.join(ART, "centralized_test_text_labels.csv"))

    Xtr = vec.transform(tr["text"].astype(str)).toarray().astype(np.float64)
    ytr = le.transform(tr["label"].astype(str).values)
    Xte = vec.transform(te["text"].astype(str)).toarray().astype(np.float64)
    yte = le.transform(te["label"].astype(str).values)

    D, C = Xtr.shape[1], len(le.classes_)
    Xte_T = Xte.T
    print(f"[{MODE}] data  Ntr={len(Xtr)} Nte={len(Xte)} D={D} C={C}")

    splits = make_label_skew_splits(ytr, k=NUM_CLIENTS, seed=SEED, labels_per_client=2)

    import collections
    for i, idxs in enumerate(splits, 1):
        cnt = collections.Counter(ytr[idxs])
        print(f"[{MODE}] client {i}: N={len(idxs)} hist={dict(cnt)}")

    layer_sizes = [D, HIDDEN_UNITS, C]
    global_model = NeuralNetwork(layer_sizes=layer_sizes, lambd=LAMBDA)
    global_vec = global_model.get_params_vector()

    print(f"\n=== FL (FedMedian | {MODE}) START === clients={NUM_CLIENTS} rounds={ROUNDS} E_local={EPOCHS_LOCAL} lr={LR_LOCAL}")
    base_pred = global_model.predict_multiclass(Xte_T)
    print(f"[{MODE}] pre-round acc={accuracy_score(yte, base_pred):.4f}")

    acc_hist = []
    for r in range(1, ROUNDS + 1):
        global_vec = fedmedian_round(global_vec, splits, Xtr, ytr, layer_sizes, LAMBDA)
        global_model.set_params_vector(global_vec.copy())
        acc = accuracy_score(yte, global_model.predict_multiclass(Xte_T))
        acc_hist.append(float(acc))
        print(f"[FedMedian | {MODE} | Round {r:02d}] acc={acc:.4f}")

    pd.DataFrame({"round": np.arange(1, len(acc_hist) + 1), "acc": acc_hist}).to_csv(ART / "fl_non_iid_fedmedian_accuracy.csv", index=False)
    print("Saved: fl_non_iid_fedmedian_accuracy.csv")


if __name__ == "__main__":
    main()
