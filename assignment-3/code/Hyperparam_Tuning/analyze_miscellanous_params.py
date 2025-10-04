# hyperparam_sweep.py
import os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from opacus import PrivacyEngine
import matplotlib.pyplot as plt
import argparse
from scipy.interpolate import make_interp_spline

# ---------------- Args ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--sweep", type=str, required=True,
                    choices=["hidden", "lot", "lr"],
                    help="Which parameter to sweep")
parser.add_argument("--smooth", action="store_true",
                    help="Apply smoothing to the accuracy curve")
args = parser.parse_args()
SWEEP_PARAM = args.sweep
SMOOTH = args.smooth

# ---------------- Config ----------------
CSV_PATH = "assignment-3/code/data/dataset.csv"
ART      = "assignment-3/artifacts"
os.makedirs(ART, exist_ok=True)

SEED       = 42
MAX_FEAT   = 258
HIDDEN_UNITS = 128
LAMBDA     = 0.02
LR         = 0.05
SIGMA      = 1.0
CLIP       = 0.17

# Sweep grids
GRID_HIDDEN = range(64, 1024, 64)
GRID_LOT    = range(10, 100, 10)
GRID_LR     = [0.01, 0.05, 0.1, 0.2, 0.5]

# ---------------- Utils ----------------
def set_seed(s): 
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def to_tensor(x, dtype=torch.float32, device="cpu"): 
    return torch.tensor(x, dtype=dtype, device=device)

def make_model(input_dim, num_classes, device, hidden_units=HIDDEN_UNITS):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, num_classes)
    ).to(device)

def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()

def train_dp(d, c, hidden, lot, lr, clip, sigma, epochs, delta):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(d, c, device, hidden_units=hidden)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loader = DataLoader(train_ds, batch_size=lot, shuffle=True, drop_last=True)
    pe = PrivacyEngine()
    model, opt, loader = pe.make_private(
        module=model, optimizer=opt, data_loader=loader,
        noise_multiplier=sigma, max_grad_norm=clip
    )
    for ep in range(100):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
    acc = accuracy(model, Xte_t, yte_t)
    eps = pe.get_epsilon(delta=delta)
    return acc, eps

# ---------------- Load Data ----------------
set_seed(SEED)
df = pd.read_csv(CSV_PATH).drop_duplicates().reset_index(drop=True)
X_text, y_raw = df["job_description"], df["job_role"]
Xtr_txt, Xte_txt, ytr_raw, yte_raw = train_test_split(
    X_text, y_raw, test_size=0.3, stratify=y_raw, random_state=SEED)

vec = TfidfVectorizer(max_features=MAX_FEAT, ngram_range=(1,2), stop_words="english")
Xtr = vec.fit_transform(Xtr_txt).toarray().astype(np.float32)
Xte = vec.transform(Xte_txt).toarray().astype(np.float32)
le = LabelEncoder(); ytr, yte = le.fit_transform(ytr_raw), le.transform(yte_raw)
D, C = Xtr.shape[1], len(le.classes_)

device = "cuda" if torch.cuda.is_available() else "cpu"
Xtr_t, ytr_t = to_tensor(Xtr, device=device), to_tensor(ytr, torch.long, device)
Xte_t, yte_t = to_tensor(Xte, device=device), to_tensor(yte, torch.long, device)
train_ds = TensorDataset(Xtr_t, ytr_t)

N = len(train_ds)
DELTA = 1.0 / N
DEFAULT_LOT = int(np.sqrt(N))

# ---------------- Sweep ----------------
if SWEEP_PARAM == "hidden":    grid, label = GRID_HIDDEN, "Hidden units"
elif SWEEP_PARAM == "lot":     grid, label = GRID_LOT, "Lot size"
elif SWEEP_PARAM == "lr":      grid, label = GRID_LR, "Learning rate"

accs, epss = [], []
for val in grid:
    hidden, lot, lr = HIDDEN_UNITS, DEFAULT_LOT, LR
    if SWEEP_PARAM == "hidden": hidden = val
    elif SWEEP_PARAM == "lot": lot = val
    elif SWEEP_PARAM == "lr": lr = val

    epochs = max(1, N // lot)  # T = N/L
    acc, eps = train_dp(D, C, hidden, lot, lr, CLIP, SIGMA, epochs, DELTA)
    accs.append(acc); epss.append(eps)
    print(f"[{label}={val}] acc={acc:.3f}, eps={eps:.2f}")

# ---------------- Plot ----------------
plt.figure(figsize=(8,5))
if SMOOTH and len(grid) > 3:
    xnew = np.linspace(min(grid), max(grid), 300)
    spline = make_interp_spline(grid, accs, k=3)
    ynew = spline(xnew)
    plt.plot(xnew, ynew, label="Smoothed Accuracy", color="red")
    plt.scatter(grid, accs, color="black", s=20, alpha=0.6, label="Raw points")
else:
    plt.plot(grid, accs, marker="o", label="Test Accuracy")

plt.xlabel(label); plt.ylabel("Final Test Accuracy")
plt.title(f"DP-SGD Sweep: {label}")
plt.grid(True); plt.legend()
plt.savefig(os.path.join(ART, f"sweep_{SWEEP_PARAM}{'_smooth' if SMOOTH else ''}.png"))
plt.show()
