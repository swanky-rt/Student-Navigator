import os, math, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from opacus import PrivacyEngine
import matplotlib.pyplot as plt

# ---------------- Config ----------------
CSV_PATH = "assignment-3/code/data/dataset.csv"
ART      = "assignment-3/artifacts"
os.makedirs(ART, exist_ok=True)

SEED       = 42
MAX_FEAT   = 258
HIDDEN     = 128
LR         = 0.05
SIGMA      = 1.0
EPOCHS     = 30

# ---------------- Utils ----------------
def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def to_tensor(x, dtype=torch.float32, device="cpu"):
    return torch.tensor(x, dtype=dtype, device=device)

def make_model(d, c, device, hidden=HIDDEN):
    return nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, c)).to(device)

def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()

# Strong composition bound
def strong_composition_curve(eps_step, T, delta):
    eps_strong = []
    for t in range(1, T+1):
        term1 = math.sqrt(2*t*math.log(1/delta)) * eps_step
        term2 = t * eps_step * (math.exp(eps_step)-1)
        eps_strong.append(term1 + term2)
    return eps_strong

# ---------------- Main ----------------
def main():
    set_seed(SEED)

    # Load data
    df = pd.read_csv(CSV_PATH).drop_duplicates().reset_index(drop=True)
    X_text, y_raw = df["job_description"], df["job_role"]
    Xtr_txt, Xte_txt, ytr_raw, yte_raw = train_test_split(
        X_text, y_raw, test_size=0.2, stratify=y_raw, random_state=SEED)

    vec = TfidfVectorizer(max_features=MAX_FEAT, stop_words="english")
    Xtr = vec.fit_transform(Xtr_txt).toarray().astype(np.float32)
    Xte = vec.transform(Xte_txt).toarray().astype(np.float32)
    le = LabelEncoder(); ytr, yte = le.fit_transform(ytr_raw), le.transform(yte_raw)

    D, C = Xtr.shape[1], len(le.classes_)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr_t, ytr_t = to_tensor(Xtr, device=device), to_tensor(ytr, torch.long, device=device)
    Xte_t, yte_t = to_tensor(Xte, device=device), to_tensor(yte, torch.long, device=device)
    train_ds = TensorDataset(Xtr_t, ytr_t)

    # Lot size = sqrt(N), epochs = N/L
    N = len(Xtr_t)
    LOT_SIZE = int(np.sqrt(N))
    EPOCHS = max(1, N // LOT_SIZE)
    train_loader = DataLoader(train_ds, batch_size=LOT_SIZE, shuffle=True)

    delta = 1.0 / N

    # Train DP model with moments accountant
    model = make_model(D, C, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    pe = PrivacyEngine()
    model, optimizer, train_loader = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=SIGMA,
        max_grad_norm=1.0,
        
    )

    eps_hist = []
    acc_hist = []

    for ep in range(1, EPOCHS+1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward(); optimizer.step()

        eps_hist.append(pe.get_epsilon(delta=delta))
        acc_hist.append(accuracy(model, Xte_t, yte_t))
        print(f"[epoch {ep}] acc={acc_hist[-1]:.3f}, eps_moments={eps_hist[-1]:.2f}")

    # Strong composition curve
    eps_step = eps_hist[0]  # first epoch epsilon ≈ per-step cost
    eps_strong_curve = strong_composition_curve(eps_step, EPOCHS, delta)

    # ---- Plot comparison ----
    epochs = np.arange(1, EPOCHS+1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs, eps_hist, label="Moments Accountant (Opacus)", color="green", linestyle="solid")
    plt.plot(epochs, eps_strong_curve, label="Strong Composition", color="orange", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("ε (epsilon)")
    plt.title(f"Privacy Accounting Comparison (δ={delta:.2e}, σ={SIGMA})")
    plt.grid(True); plt.legend()
    plt.savefig(os.path.join(ART,"epsilon_comparison.png"))
    plt.show()

if __name__ == "__main__":
    main()
