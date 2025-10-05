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
CSV_PATH = "assignment-3/code/data/dataset.csv"  # dataset path
ART      = "assignment-3/artifacts"              # folder for saving artifacts/plots
SEED     = 7867                                  # seed for reproducibility


# ------------------ Model & Featurization ------------------
MAX_FEATURES    = 200      # how many TF-IDF features to extract from text
HIDDEN_UNITS    = 128      # size of the hidden layer
LAMBDA          = 0.02     # (optional) weight decay
MAX_GRAD_NORM   = 1.5      # gradient clipping bound (used by DP-SGD)

# ------------------ Training (main DP run) ------------------
EPOCHS_CENTRAL  = 60       # training epochs for baseline model (unused here)
LR_CENTRAL      = 0.6      # learning rate for central model
BATCH_SIZE      = 60       # batch size for non-DP setup (used as a base)


# ------------------ Utility Helpers ------------------
def set_seed(seed: int):
    # set all random seeds to make runs deterministic
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_text(df: pd.DataFrame) -> pd.Series:
    # returns a text column for TF-IDF extraction
    # uses job_description by default, or joins text columns if needed
    if "job_description" in df.columns:
        return df["job_description"].astype(str)
    cols = [c for c in ["job_description"] if c in df.columns]
    if not cols:
        raise ValueError("No text-like columns found.")
    return df[cols].astype(str).agg(" ".join, axis=1)

def make_leak_cleaner(labels_lower: List[str]):
    """
    Creates a small cleaner that removes any words from job descriptions
    that directly appear in the label names (to prevent label leakage).
    """
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
    # compute test accuracy — simple helper for evaluation
    model.eval()
    with torch.no_grad():
        logits = model(Xte)
        pred = logits.argmax(dim=1)
        return (pred == yte).float().mean().item()

def to_tensor(np_array, dtype=torch.float32, device="cpu"):
    # wraps numpy arrays into PyTorch tensors on the correct device
    return torch.tensor(np_array, dtype=dtype, device=device)

def make_model(input_dim: int, num_classes: int, device: str):
    # simple 2-layer MLP classifier
    return nn.Sequential(
        nn.Linear(input_dim, HIDDEN_UNITS, bias=True),
        nn.ReLU(),
        nn.Linear(HIDDEN_UNITS, num_classes, bias=True),
    ).to(device)


# ------------------ Main Entry Point ------------------
def main():
    set_seed(SEED)
    os.makedirs(ART, exist_ok=True)

    # ---------- Load & clean data ----------
    df = pd.read_csv(CSV_PATH).drop_duplicates().reset_index(drop=True)

    # Extract job descriptions and labels
    text_raw = build_text(df)
    labels = df["job_role"].astype(str)

    # Clean text to remove label-related words (avoids leakage)
    cleaner = make_leak_cleaner(labels.str.lower().unique())
    text = text_raw.apply(cleaner)

    # Split into train/test (stratified to preserve class balance)
    Xtr_txt, Xte_txt, ytr_raw, yte_raw = train_test_split(
        text, labels.values, test_size=0.20, random_state=SEED, stratify=labels.values
    )
    print(f"[split] train={len(Xtr_txt)} test={len(Xte_txt)}")

    # ---------- TF-IDF vectorization ----------
    vec = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2), stop_words="english")
    Xtr = vec.fit_transform(Xtr_txt).toarray().astype(np.float32)
    Xte = vec.transform(Xte_txt).toarray().astype(np.float32)

    # ---------- Label encoding ----------
    le = LabelEncoder()
    ytr = le.fit_transform(ytr_raw).astype(np.int64)
    yte = le.transform(yte_raw).astype(np.int64)
    D, C = Xtr.shape[1], len(le.classes_)
    print(f"[data] D={D} C={C}")

    # save TF-IDF and label encoder for reproducibility
    with open(os.path.join(ART, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vec, f)
    with open(os.path.join(ART, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    # ---------- Convert to PyTorch ----------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr_t = to_tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = to_tensor(ytr, dtype=torch.long, device=device)
    Xte_t = to_tensor(Xte, dtype=torch.float32, device=device)
    yte_t = to_tensor(yte, dtype=torch.long, device=device)

    train_ds = TensorDataset(Xtr_t, ytr_t)
    base_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # ----------------- Param Sweep: (clip, sigma) -----------------
    # We'll test multiple combinations of gradient clipping norms (C)
    # and noise multipliers (σ) to see how privacy affects accuracy.
    print("\n[dp-grid] Running param sweep (clip, sigma)...")

    DELTA = 1.0 / len(Xtr_t)    # DP delta = 1/N (common default)
    SWEEP_CLIP  = [0.5, 1.0]    # try two clipping bounds
    SWEEP_SIGMA = [0.5, 1.0, 2.0]  # test three noise levels
    EPOCHS_GRID = 50            # fixed training epochs per combo
    BATCH_SIZE  = 150            # batch size for DP runs

    # containers to track results
    results = []
    acc_curves = {}
    train_curves = {}
    eps_curves = {}

    # loop over each (clip, sigma) combination
    for clip in SWEEP_CLIP:
        for sigma in SWEEP_SIGMA:
            model_g = make_model(D, C, device)
            opt_g = torch.optim.SGD(model_g.parameters(), lr=0.10)

            # attach Opacus PrivacyEngine
            pe_g = PrivacyEngine()
            model_g, opt_g, loader_g = pe_g.make_private(
                module=model_g,
                optimizer=opt_g,
                data_loader=base_loader,
                noise_multiplier=sigma,
                max_grad_norm=clip,
            )

            # track metrics for this run
            acc_hist_g = []
            train_acc_hist_g = []
            eps_hist_g = []

            # ---------- DP-SGD training loop ----------
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

                    # track train accuracy manually
                    preds = logits.argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total += yb.shape[0]

                # compute accuracy & epsilon each epoch
                train_acc = correct / total if total > 0 else 0.0
                train_acc_hist_g.append(train_acc)
                acc_g = accuracy(model_g, Xte_t, yte_t)
                acc_hist_g.append(acc_g)
                eps_g = pe_g.get_epsilon(delta=DELTA)
                eps_hist_g.append(eps_g)

            # store results for this (clip, sigma)
            results.append({
                "clip": clip, "sigma": sigma,
                "acc": acc_hist_g[-1],
                "eps": eps_hist_g[-1]
            })
            acc_curves[(clip, sigma)] = acc_hist_g
            train_curves[(clip, sigma)] = train_acc_hist_g
            eps_curves[(clip, sigma)] = eps_hist_g

            print(f"[dp-grid] C={clip}, σ={sigma} -> train_acc={train_acc_hist_g[-1]:.4f}, "
                  f"test_acc={acc_hist_g[-1]:.4f}, ε={eps_hist_g[-1]:.3f}")

    # save all numeric results
    pd.DataFrame(results).to_csv(os.path.join(ART,"dp_param_sweep_results.csv"), index=False)
    print("Saved: dp_param_sweep_results.csv")

    # ----------------- Plot 1: Train accuracy -----------------
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

    # ----------------- Plot 2: Test accuracy -----------------
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


# ------------------ Run ------------------
if __name__ == "__main__":
    main()
