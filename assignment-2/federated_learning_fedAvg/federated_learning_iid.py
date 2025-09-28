import os, pickle, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from neural_network_model import NeuralNetwork

SEED = 42
ART  = "../artifacts_centralized"
NUM_CLIENTS  = 5
ROUNDS       = 100
EPOCHS_LOCAL = 5
LR_LOCAL     = 0.05
HIDDEN_UNITS = 64
LAMBDA       = 1e-4
WARM_START   = False

MODE = "IID"

def make_iid_splits_stratified(y, k=5, seed=42):
    y = np.asarray(y); idx = np.arange(len(y))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return [idx_test.tolist() for _, idx_test in skf.split(idx, y)]

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
    print(f"[{MODE}] data  Ntr={len(Xtr)} Nte={len(Xte)} D={D} C={C}")

    splits = make_iid_splits_stratified(ytr, k=NUM_CLIENTS, seed=SEED)

    import collections
    for i, idxs in enumerate(splits,1):
        cnt = collections.Counter(ytr[idxs])
        print(f"[{MODE}] client {i}: N={len(idxs)} hist={dict(cnt)}")

    def client_data(idxs):
        Xi = Xtr[idxs]; yi = ytr[idxs]
        return Xi.T, yi  # (D, Ni), (Ni,)

    layer_sizes = [D, HIDDEN_UNITS, C]
    global_model = NeuralNetwork(layer_sizes=layer_sizes, lambd=LAMBDA)
    global_vec = global_model.get_params_vector()

    print(f"\n=== FL ({MODE}) START === clients={NUM_CLIENTS} rounds={ROUNDS} E_local={EPOCHS_LOCAL} lr={LR_LOCAL}")
    base_pred = global_model.predict_multiclass(Xte_T)
    print(f"[{MODE}] pre-round acc={accuracy_score(yte, base_pred):.4f}")

    def fedavg_round(gvec):
        total, accum = 0, np.zeros_like(gvec)
        for idxs in splits:
            Xi, yi = client_data(idxs)
            local = NeuralNetwork(layer_sizes=layer_sizes, lambd=LAMBDA)
            local.set_params_vector(gvec.copy())
            local.train_multiclass(Xi, yi, lr=LR_LOCAL, max_epochs=EPOCHS_LOCAL, verbose=False)
            wi = local.get_params_vector()
            n_i = Xi.shape[1]
            accum += n_i * wi; total += n_i
        return accum / max(1, total)

    acc_hist = []
    for r in range(1, ROUNDS+1):
        global_vec = fedavg_round(global_vec)
        global_model.set_params_vector(global_vec.copy())
        acc = accuracy_score(yte, global_model.predict_multiclass(Xte_T))
        acc_hist.append(float(acc))
        print(f"[{MODE} | Round {r:02d}] acc={acc:.4f}")

    pd.DataFrame({"round": np.arange(1, len(acc_hist)+1), "acc": acc_hist}).to_csv("fl_iid_accuracy.csv", index=False)
    print("Saved: fl_iid_accuracy.csv")

if __name__ == "__main__":
    main()
