#!/usr/bin/env python3

import argparse, time, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from pathlib import Path

ART = Path(__file__).resolve().parents[1] / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)

@torch.no_grad()
def accuracy(model, X_t, y_t):
    model.eval()
    return (model(X_t).argmax(1) == y_t).float().mean().item()

def _nll_from_proba(proba, y):
    eps = 1e-12
    p = np.take_along_axis(np.clip(proba, eps, 1 - eps), y.reshape(-1, 1), axis=1).ravel()
    return -np.log(p)

def loss_threshold_attack(proba_in, y_in, proba_out, y_out, name="Yeom-Loss"):
    loss_in  = _nll_from_proba(proba_in,  y_in)
    loss_out = _nll_from_proba(proba_out, y_out)
    scores   = np.concatenate([-loss_in, -loss_out])   # higher => member-like
    labels   = np.concatenate([np.ones_like(loss_in), np.zeros_like(loss_out)])
    auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else float("nan")
    return {"name": name, "auc": float(auc)}

def make_big_model(d, c, device):

    m = nn.Sequential(
        nn.Linear(d, 4096), nn.ReLU(),
        nn.Linear(4096, 1024), nn.ReLU(),
        nn.Linear(1024, c)
    ).to(device)
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="assignment-3/code/data/dataset.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.05, help="Fraction used as TRAIN (members).")
    ap.add_argument("--max-feat", type=int, default=100_000)
    ap.add_argument("--ngrams", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.30)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    set_seed(args.seed)

    df = pd.read_csv(args.csv).drop_duplicates().reset_index(drop=True)
    X_text, y_raw = df.iloc[:,0], df.iloc[:,1]
    Xtr_txt, Xte_txt, ytr_raw, yte_raw = train_test_split(
        X_text, y_raw, train_size=args.train_frac, stratify=y_raw, random_state=args.seed
    )

    vec = TfidfVectorizer(max_features=args.max_feat,
                          ngram_range=(1, args.ngrams),
                          stop_words=None, lowercase=True)
    Xtr = vec.fit_transform(Xtr_txt).toarray().astype(np.float32)
    Xte = vec.transform(Xte_txt).toarray().astype(np.float32)
    le  = LabelEncoder(); ytr = le.fit_transform(ytr_raw); yte = le.transform(yte_raw)

    d, c = Xtr.shape[1], len(le.classes_)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long,    device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    yte_t = torch.tensor(yte, dtype=torch.long,    device=device)

    dl = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=args.batch, shuffle=True, num_workers=0)

    model = make_big_model(d, c, device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    model.train()
    last = time.time()
    for ep in range(1, args.epochs+1):
        for xb, yb in dl:
            opt.zero_grad(); l = F.cross_entropy(model(xb), yb); l.backward(); opt.step()
        if time.time() - last > 1.0:
            print(f"[OVERFIT-loss-threshold-attack] epoch {ep}/{args.epochs} (loss={float(l):.4f})", flush=True)
            last = time.time()

    tr_acc = accuracy(model, Xtr_t, ytr_t)
    te_acc = accuracy(model, Xte_t, yte_t)
    print(f"[OVERFIT-loss-threshold-attack train acc={tr_acc:.3f} | test acc={te_acc:.3f}")

    with torch.no_grad():
        p_in  = torch.softmax(model(Xtr_t), dim=1).cpu().numpy()
        p_out = torch.softmax(model(Xte_t), dim=1).cpu().numpy()

    res = loss_threshold_attack(p_in, ytr, p_out, yte)
    print(f"[OVERFIT-loss-threshold-attack] {res['name']} AUC={res['auc']:.3f}")

    loss_in  = _nll_from_proba(p_in,  ytr)
    loss_out = _nll_from_proba(p_out, yte)
    scores   = np.concatenate([-loss_in, -loss_out])
    labels   = np.concatenate([np.ones_like(loss_in), np.zeros_like(loss_out)])

    np.savez(ART / "pre_attack_scores_labels.npz", scores=scores, labels=labels, auc=res["auc"])

    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure(figsize=(7,6))
    plt.plot(fpr, tpr, label=f"PRE (AUC={res['auc']:.3f})")
    plt.plot([0,1],[0,1],'k--', lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("loss-threshold attack ROC: PRE"); plt.grid(True); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(ART / "loss-threshold-attack.png"); plt.close()

    print("\nHint: If AUC < 0.70, try one or more:\n"
          "  --train-frac 0.02    --epochs 800    --lr 0.4    --batch 2\n"
          "  --max-feat 150000    --ngrams 5\n")

if __name__ == "__main__":
    main()
