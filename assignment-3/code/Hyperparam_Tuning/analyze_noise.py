import os, re, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from opacus import PrivacyEngine
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


# ---------------- Config ----------------
# basic setup — paths, constants, and model hyperparameters
CSV_PATH = "assignment-3/code/data/dataset.csv"   # where the dataset lives
ART      = "assignment-3/artifacts"               # folder to store plots and outputs
os.makedirs(ART, exist_ok=True)                   # make sure the folder exists

SEED       = 5
MAX_FEAT   = 258      # number of TF-IDF features
HIDDEN     = 128      # hidden layer size
LR         = 0.1       # learning rate
SIGMA      = 1.0       # noise multiplier (for DP)
LAMBDA     = 0.02      # weight decay (currently unused)


# ---------------- Utils ----------------
def set_seed(seed):
    # sets all random seeds so results are reproducible
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_tensor(x, dtype=torch.float32, device="cpu"):
    # converts numpy arrays into torch tensors
    return torch.tensor(x, dtype=dtype, device=device)

def make_model(input_dim, num_classes, device, hidden_units=HIDDEN):
    # small feedforward network: input → hidden → output
    return nn.Sequential(
        nn.Linear(input_dim, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, num_classes)
    ).to(device)

def accuracy(model, X, y):
    # computes model accuracy on the given dataset
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(1)
        return (preds == y).float().mean().item()

def estimate_median_grad_norm(model, loader, device, num_batches=50):
    # estimates the median gradient norm across several mini-batches
    # helps decide what clipping norm (C) to use for DP training
    norms = []
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    for i, (xb, yb) in enumerate(loader):
        if i >= num_batches:
            break
        opt.zero_grad(set_to_none=True)
        logits = model(xb.to(device))
        loss = F.cross_entropy(logits, yb.to(device))
        loss.backward()
        # combine gradient norms from all parameters
        total_norm = torch.norm(
            torch.stack([p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None])
        ).item()
        norms.append(total_norm)
    return np.median(norms)

def train_dp(d, c, clip, sigma, lot, epochs, delta):
    # trains a model with DP-SGD using Opacus
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(d, c, device)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    loader = DataLoader(train_ds, batch_size=lot, shuffle=True)

    # attach Opacus privacy engine for differential privacy
    pe = PrivacyEngine()
    model, opt, loader = pe.make_private(
        module=model, optimizer=opt, data_loader=loader,
        noise_multiplier=sigma, max_grad_norm=clip
    )

    # basic training loop
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()

    # check how well it performs (accuracy + privacy budget)
    acc = accuracy(model, Xte_t, yte_t)
    eps = pe.get_epsilon(delta=delta)
    return acc, eps


# ---------------- Load Data ----------------
set_seed(SEED)
df = pd.read_csv(CSV_PATH).drop_duplicates().reset_index(drop=True)

# split the dataset into text features (X) and labels (y)
X_text, y_raw = df["job_description"], df["job_role"]

# stratified split so label distribution stays balanced
Xtr_txt, Xte_txt, ytr_raw, yte_raw = train_test_split(
    X_text, y_raw, test_size=0.3, stratify=y_raw, random_state=SEED
)

# convert text into TF-IDF vectors
vec = TfidfVectorizer(max_features=MAX_FEAT, ngram_range=(1,2), stop_words="english")
Xtr = vec.fit_transform(Xtr_txt).toarray().astype(np.float32)
Xte = vec.transform(Xte_txt).toarray().astype(np.float32)

# encode text labels into numeric form
le = LabelEncoder()
ytr, yte = le.fit_transform(ytr_raw), le.transform(yte_raw)

# move everything to tensors
D, C = Xtr.shape[1], len(le.classes_)
device = "cuda" if torch.cuda.is_available() else "cpu"
Xtr_t, ytr_t = to_tensor(Xtr, device=device), to_tensor(ytr, torch.long, device)
Xte_t, yte_t = to_tensor(Xte, device=device), to_tensor(yte, torch.long, device)
train_ds = TensorDataset(Xtr_t, ytr_t)

# define DP parameters
N = len(train_ds)
DELTA = 1.0 / N              # standard delta for DP
DEFAULT_LOT = 60             # batch size
EPOCHS = max(1, N // DEFAULT_LOT)  # roughly one full pass over dataset


# ---------------- Estimate Median Clip ----------------
# estimate a good gradient clipping bound to start from
tmp_model = make_model(D, C, device)
base_loader = DataLoader(train_ds, batch_size=DEFAULT_LOT, shuffle=True)
median_clip = estimate_median_grad_norm(tmp_model, base_loader, device)
print(f"Estimated median grad norm: {median_clip:.4f}")

# Sweep clipping values around median (not used further here, just shown for reference)
GRID_CLIP = [round(median_clip * f, 3) for f in np.linspace(0.5, 2.0, 8)]

# ---------------- Noise Sweep ----------------
# here we fix the clipping norm and vary noise multiplier (σ)
accs, epss = [], []
clip = 0.17  # fixed clipping norm (based on earlier runs)
NOISE_GRID = [0.5, 1, 1.5, 2, 2.5, 3]

# train with different noise values and track accuracy + epsilon
for sigma in NOISE_GRID:
    acc, eps = train_dp(D, C, clip, sigma, DEFAULT_LOT, EPOCHS, DELTA)
    accs.append(acc)
    epss.append(eps)
    print(f"[Noise={sigma}] acc={acc:.3f}, eps={eps:.2f}")


# ---------------- Plot ----------------
# plot how test accuracy changes with increasing noise multiplier

# create smooth curve for better visualization
x_smooth = np.linspace(min(NOISE_GRID), max(NOISE_GRID), 200)
y_smooth = make_interp_spline(NOISE_GRID, accs, k=3)(x_smooth)

plt.figure(figsize=(7,5))
plt.plot(x_smooth, y_smooth, 'r-', label="Smoothed Accuracy")
plt.scatter(NOISE_GRID, accs, c='black', label="Observed points")

plt.ylim(bottom=0.5)  # keep lower bound consistent
plt.xlabel("Noise Multiplier (σ)")
plt.ylabel("Test Accuracy")
plt.title("Accuracy vs Noise Multiplier")
plt.legend()
plt.grid(True)

# save the figure for your report
plt.savefig(os.path.join(ART,"noise_vs_acc.png"))
plt.show()
