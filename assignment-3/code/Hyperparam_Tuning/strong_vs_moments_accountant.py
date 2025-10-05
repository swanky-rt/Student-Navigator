import os, math, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from opacus import PrivacyEngine
import matplotlib.pyplot as plt


# ---------------- Config ----------------
# Basic experiment setup — file paths, constants, and training params
CSV_PATH = "assignment-3/code/data/dataset.csv"
ART      = "assignment-3/artifacts"
os.makedirs(ART, exist_ok=True)

SEED       = 42
MAX_FEAT   = 258       # number of TF-IDF features
HIDDEN     = 128       # hidden layer size
LR         = 0.05      # learning rate
SIGMA      = 1.0       # noise multiplier for DP
EPOCHS     = 30        # number of passes through data


# ---------------- Utils ----------------
def set_seed(s):
    # make runs deterministic by fixing random seeds
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def to_tensor(x, dtype=torch.float32, device="cpu"):
    # convert numpy arrays into torch tensors
    return torch.tensor(x, dtype=dtype, device=device)

def make_model(d, c, device, hidden=HIDDEN):
    # simple 2-layer feedforward neural network
    return nn.Sequential(
        nn.Linear(d, hidden),
        nn.ReLU(),
        nn.Linear(hidden, c)
    ).to(device)

def accuracy(model, X, y):
    # evaluate classification accuracy
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()

def strong_composition_curve(eps_step, T, delta):
    """
    Compute the strong composition bound manually.
    Shows how total epsilon grows when applying 
    (ε, δ)-DP T times using basic composition theory.
    """
    eps_strong = []
    for t in range(1, T + 1):
        term1 = math.sqrt(2 * t * math.log(1 / delta)) * eps_step
        term2 = t * eps_step * (math.exp(eps_step) - 1)
        eps_strong.append(term1 + term2)
    return eps_strong


# ---------------- Main ----------------
def main():
    set_seed(SEED)

    # ---- Load data ----
    df = pd.read_csv(CSV_PATH).drop_duplicates().reset_index(drop=True)
    X_text, y_raw = df["job_description"], df["job_role"]

    # split data into train/test (keeping class balance)
    Xtr_txt, Xte_txt, ytr_raw, yte_raw = train_test_split(
        X_text, y_raw, test_size=0.2, stratify=y_raw, random_state=SEED
    )

    # convert text → TF-IDF vectors
    vec = TfidfVectorizer(max_features=MAX_FEAT, stop_words="english")
    Xtr = vec.fit_transform(Xtr_txt).toarray().astype(np.float32)
    Xte = vec.transform(Xte_txt).toarray().astype(np.float32)

    # encode labels → integers
    le = LabelEncoder()
    ytr, yte = le.fit_transform(ytr_raw), le.transform(yte_raw)

    D, C = Xtr.shape[1], len(le.classes_)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # move everything to tensors
    Xtr_t = to_tensor(Xtr, device=device)
    ytr_t = to_tensor(ytr, dtype=torch.long, device=device)
    Xte_t = to_tensor(Xte, device=device)
    yte_t = to_tensor(yte, dtype=torch.long, device=device)

    # create TensorDataset and DataLoader
    train_ds = TensorDataset(Xtr_t, ytr_t)

    # set lot (batch) size and epochs dynamically
    N = len(Xtr_t)
    LOT_SIZE = int(np.sqrt(N))          # common heuristic for DP-SGD
    EPOCHS = max(1, N // LOT_SIZE)      # roughly one pass over dataset
    train_loader = DataLoader(train_ds, batch_size=LOT_SIZE, shuffle=True)

    delta = 1.0 / N                     # DP delta = 1/N (standard default)

    # ---- Train DP model (Moments Accountant) ----
    model = make_model(D, C, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    # attach the PrivacyEngine — handles noise addition + ε tracking
    pe = PrivacyEngine()
    model, optimizer, train_loader = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=SIGMA,
        max_grad_norm=1.0,
    )

    eps_hist = []   # privacy budget after each epoch
    acc_hist = []   # test accuracy per epoch

    for ep in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()

        # log privacy ε and accuracy after each epoch
        eps_hist.append(pe.get_epsilon(delta=delta))
        acc_hist.append(accuracy(model, Xte_t, yte_t))
        print(f"[epoch {ep}] acc={acc_hist[-1]:.3f}, eps_moments={eps_hist[-1]:.2f}")

    # ---- Strong composition curve (for comparison) ----
    # Approximate what ε would be if we applied standard composition theory
    eps_step = eps_hist[0]  # ε from first epoch ≈ per-step privacy cost
    eps_strong_curve = strong_composition_curve(eps_step, EPOCHS, delta)

    # ---- Plot: Moments Accountant vs Strong Composition ----
    epochs = np.arange(1, EPOCHS + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, eps_hist, label="Moments Accountant (Opacus)", color="green", linestyle="solid")
    plt.plot(epochs, eps_strong_curve, label="Strong Composition", color="orange", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("ε (epsilon)")
    plt.title(f"Privacy Accounting Comparison (δ={delta:.2e}, σ={SIGMA})")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(ART, "epsilon_comparison.png"))
    plt.show()


# ---- Run ----
if __name__ == "__main__":
    main()
