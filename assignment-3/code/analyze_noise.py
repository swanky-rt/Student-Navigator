import os, numpy as np, pandas as pd
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
LR         = 0.1
LAMBDA     = 0.02
CLIP       = 1.0
NOISE_GRID = [0.1, 0.5, 1, 2, 3, 4, 5]

# ---------------- Utils ----------------
def set_seed(s): 
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def to_tensor(x, dtype=torch.float32, device="cpu"): 
    return torch.tensor(x, dtype=dtype, device=device)

def make_model(d, c, device):
    return nn.Sequential(
        nn.Linear(d, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, c)
    ).to(device)

def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()

def l2_weight_decay(model, lambd):
    reg = torch.tensor(0.0, device=next(model.parameters()).device)
    for p in model.parameters():
        if p.requires_grad:
            reg = reg + p.pow(2).sum()
    return lambd * 0.5 * reg

def train_dp(d, c, lot, lr, clip, sigma, epochs, delta):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(d, c, device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loader = DataLoader(train_ds, batch_size=lot, shuffle=True)
    pe = PrivacyEngine()
    model, opt, loader = pe.make_private(
        module=model, optimizer=opt, data_loader=loader,
        noise_multiplier=sigma, max_grad_norm=clip
    )
    for ep in range(epochs):
        for xb,yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits,yb) + l2_weight_decay(model, LAMBDA)
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
LOT_SIZE = int(np.sqrt(N))
EPOCHS = max(1, N // LOT_SIZE)
DELTA = 1.0 / N

# ---------------- Baseline ----------------
print("[baseline] Training non-DP model...")
baseline_model = make_model(D, C, device)
opt = torch.optim.SGD(baseline_model.parameters(), lr=LR)
loader = DataLoader(train_ds, batch_size=LOT_SIZE, shuffle=True)
for ep in range(EPOCHS):
    for xb, yb in loader:
        opt.zero_grad()
        logits = baseline_model(xb)
        loss = F.cross_entropy(logits,yb) + l2_weight_decay(baseline_model, LAMBDA)
        loss.backward()
        opt.step()
baseline_acc = accuracy(baseline_model, Xte_t, yte_t)
print(f"Baseline accuracy = {baseline_acc:.3f}")

# ---------------- Sweep Noise ----------------
accs, epss = [], []
for sigma in NOISE_GRID:
    acc, eps = train_dp(D, C, LOT_SIZE, LR, CLIP, sigma, EPOCHS, DELTA)
    accs.append(acc); epss.append(eps)
    print(f"[Noise={sigma}] acc={acc:.3f}, eps={eps:.2f}")

# ---------------- Plot ----------------
plt.figure(figsize=(8,5))
plt.plot(NOISE_GRID, accs, marker="o", label="DP Test Accuracy")
plt.axhline(y=baseline_acc, color="blue", linestyle="--", label=f"Baseline acc={baseline_acc:.3f}")
max_idx = np.argmax(accs)
plt.scatter(NOISE_GRID[max_idx], accs[max_idx], color="green", s=80, label=f"Peak acc={accs[max_idx]:.3f}")
plt.xlabel("Noise multiplier (σ)"); plt.ylabel("Accuracy")
plt.title("Effect of Noise on DP-SGD Accuracy")
plt.grid(True); plt.legend()
plt.savefig(os.path.join(ART, "noise_vs_acc.png"))
plt.show()
