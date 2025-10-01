import os, re, pickle, numpy as np, pandas as pd
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from opacus import PrivacyEngine
import matplotlib.pyplot as plt

# ------------------ Paths & Seeds ------------------
CSV_PATH = "assignment-3/code/data/dataset.csv"
ART      = "assignment-3/artifacts"
SEED            = 7867

# ------------------ Model & Featurization ------------------
MAX_FEATURES    = 200
HIDDEN_UNITS    = 128
LAMBDA          = 0.02
MAX_GRAD_NORM   = 1.5

# ------------------ Training (main DP run) ------------------
EPOCHS_CENTRAL  = 60
LR_CENTRAL      = 0.6
BATCH_SIZE      = 60

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_text(df: pd.DataFrame) -> pd.Series:
    if "job_description" in df.columns:
        return df["job_description"].astype(str)
    cols = [c for c in ["job_description"] if c in df.columns]
    if not cols:
        raise ValueError("No text-like columns found.")
    return df[cols].astype(str).agg(" ".join, axis=1)

def make_leak_cleaner(labels_lower: List[str]):
    toks = set()
    for lbl in labels_lower:
        toks |= set(re.split(r"\s+", str(lbl).strip().lower()))
    toks = {t for t in toks if t}
    pats = [re.compile(rf"\b{re.escape(t)}\b", re.I) for t in toks]
    def clean(s: str):
        s = str(s).lower()
        for p in pats:
            s = p.sub(" ", s)
        return re.sub(r"\s+", " ", s).strip()
    return clean

def accuracy(model, Xte: torch.Tensor, yte: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(Xte)
        pred = logits.argmax(dim=1)
        return (pred == yte).float().mean().item()

def to_tensor(np_array, dtype=torch.float32, device="cpu"):
    return torch.tensor(np_array, dtype=dtype, device=device)

def make_model(input_dim: int, num_classes: int, device: str):
    return nn.Sequential(
        nn.Linear(input_dim, HIDDEN_UNITS, bias=True),
        nn.ReLU(),
        nn.Linear(HIDDEN_UNITS, num_classes, bias=True),
    ).to(device)

def main():
    set_seed(SEED)
    os.makedirs(ART, exist_ok=True)

    # ----------------- Load & clean -----------------
    df = pd.read_csv(CSV_PATH)
    df = df.drop_duplicates().reset_index(drop=True)

    text_raw = build_text(df)
    labels = df["job_role"].astype(str)

    cleaner = make_leak_cleaner(labels.str.lower().unique())
    text = text_raw.apply(cleaner)

    Xtr_txt, Xte_txt, ytr_raw, yte_raw = train_test_split(
        text, labels.values, test_size=0.20, random_state=SEED, stratify=labels.values
    )
    print(f"[split] train={len(Xtr_txt)} test={len(Xte_txt)}")

    vec = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2), stop_words="english")
    Xtr = vec.fit_transform(Xtr_txt).toarray().astype(np.float32)
    Xte = vec.transform(Xte_txt).toarray().astype(np.float32)

    le = LabelEncoder()
    ytr = le.fit_transform(ytr_raw).astype(np.int64)
    yte = le.transform(yte_raw).astype(np.int64)

    D, C = Xtr.shape[1], len(le.classes_)
    print(f"[data] D={D} C={C}")

    # Save artifacts
    with open(os.path.join(ART, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vec, f)
    with open(os.path.join(ART, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    # Torch tensors
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr_t = to_tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = to_tensor(ytr, dtype=torch.long, device=device)
    Xte_t = to_tensor(Xte, dtype=torch.float32, device=device)
    yte_t = to_tensor(yte, dtype=torch.long, device=device)

    train_ds = TensorDataset(Xtr_t, ytr_t)
    base_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # ----------------- Param Sweep: (sigma, epsilon) -----------------
    print("\n[dp-grid] Running param sweep (sigma, epsilon)...")
    results = []
    acc_curves = {}
    train_curves = {}
    eps_curves = {}
    DELTA = 1.0 / len(Xtr_t)
    SWEEP_CLIP = [0.5, 1.0]
    SWEEP_SIGMA = [0.5, 1.0, 2.0]
    EPOCHS_GRID = 50
    BATCH_SIZE      = 150
    results = []
    acc_curves = {}
    train_curves = {}
    eps_curves = {}
    for clip in SWEEP_CLIP:
        for sigma in SWEEP_SIGMA:
            model_g = make_model(D, C, device)
            opt_g = torch.optim.SGD(model_g.parameters(), lr=0.10)
            pe_g = PrivacyEngine()
            model_g, opt_g, loader_g = pe_g.make_private(
                module=model_g,
                optimizer=opt_g,
                data_loader=base_loader,
                noise_multiplier=sigma,
                max_grad_norm=clip,
            )
            acc_hist_g = []
            train_acc_hist_g = []
            eps_hist_g = []
            for ep in range(EPOCHS_GRID):
                model_g.train()
                correct = 0
                total = 0
                for xb, yb in loader_g:
                    opt_g.zero_grad(set_to_none=True)
                    logits = model_g(xb)
                    loss = F.cross_entropy(logits, yb)
                    loss.backward()
                    opt_g.step()
                    preds = logits.argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total += yb.shape[0]
                train_acc = correct / total if total > 0 else 0.0
                train_acc_hist_g.append(train_acc)
                acc_g = accuracy(model_g, Xte_t, yte_t)
                acc_hist_g.append(acc_g)
                eps_g = pe_g.get_epsilon(delta=DELTA)
                eps_hist_g.append(eps_g)
            results.append({"clip": clip, "sigma": sigma, "acc": acc_hist_g[-1], "eps": eps_hist_g[-1]})
            acc_curves[(clip, sigma)] = acc_hist_g
            train_curves[(clip, sigma)] = train_acc_hist_g
            eps_curves[(clip, sigma)] = eps_hist_g
            print(f"[dp-grid] C={clip}, σ={sigma} -> train_acc={train_acc_hist_g[-1]:.4f}, test_acc={acc_hist_g[-1]:.4f}, ε={eps_hist_g[-1]:.3f}")
    pd.DataFrame(results).to_csv(os.path.join(ART,"dp_param_sweep_results.csv"), index=False)
    print("Saved: dp_param_sweep_results.csv")

    # GRAPH 1: Train accuracy vs epoch for each (clip, sigma)
    plt.figure(figsize=(10,7))
    for (clip, sigma) in train_curves:
        final_eps = eps_curves[(clip, sigma)][-1]
        final_acc = acc_curves[(clip, sigma)][-1]
        label = f"C={clip}, σ={sigma}, final ε={final_eps:.2f}, final acc={final_acc:.3f}"
        plt.plot(range(1, EPOCHS_GRID+1), train_curves[(clip, sigma)], label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Train Accuracy")
    plt.title("DP-SGD Train Accuracy (Clip & Sigma Sweep, Fixed Delta)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(ART,"dp_param_sweep_train_acc_vs_epoch.png"))
    print("Saved: dp_param_sweep_train_acc_vs_epoch.png")

    # GRAPH 2: Test accuracy vs epoch for each (clip, sigma), show final epsilon and final accuracy in legend
    plt.figure(figsize=(10,7))
    for (clip, sigma) in acc_curves:
        final_eps = eps_curves[(clip, sigma)][-1]
        final_acc = acc_curves[(clip, sigma)][-1]
        label = f"C={clip}, σ={sigma}, final ε={final_eps:.2f}, final acc={final_acc:.3f}"
        plt.plot(range(1, EPOCHS_GRID+1), acc_curves[(clip, sigma)], label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("DP-SGD Test Accuracy (Clip & Sigma Sweep, Fixed Delta)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(ART,"dp_param_sweep_test_acc_vs_epoch.png"))
    print("Saved: dp_param_sweep_test_acc_vs_epoch.png")

if __name__ == "__main__":
    main()
