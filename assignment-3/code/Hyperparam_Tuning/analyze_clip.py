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
CSV_PATH = "assignment-3/code/data/dataset.csv"   # path to our dataset
ART      = "assignment-3/artifacts"               # folder to store plots and outputs
os.makedirs(ART, exist_ok=True)                   # make sure it exists before saving

# core experiment settings
SEED       = 42
MAX_FEAT   = 258      # number of TF-IDF features to extract
HIDDEN     = 128      # hidden layer size
LR         = 0.1      # learning rate for optimizer
SIGMA      = 1.0      # DP noise multiplier
LAMBDA     = 0.02     # regularization (not used directly but can be used later)


# ---------------- Utils ----------------
def set_seed(seed):
    # set all random seeds so results are consistent each run
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_tensor(x, dtype=torch.float32, device="cpu"):
    # quick helper to move numpy arrays into PyTorch tensors
    return torch.tensor(x, dtype=dtype, device=device)

def make_model(input_dim, num_classes, device, hidden_units=HIDDEN):
    # simple feedforward network with one hidden layer
    return nn.Sequential(
        nn.Linear(input_dim, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, num_classes)
    ).to(device)

def accuracy(model, X, y):
    # compute model accuracy without gradient updates
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(1)
        return (preds == y).float().mean().item()

def estimate_median_grad_norm(model, loader, device, num_batches=50):
    # estimate the median gradient norm across a few batches
    # this helps decide what clipping bound to use for DP
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
        # get the total L2 norm of all parameter gradients
        total_norm = torch.norm(
            torch.stack([p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None])
        ).item()
        norms.append(total_norm)
    return np.median(norms)

def train_dp(d, c, clip, sigma, lot, epochs, delta):
    # train a model with DP-SGD using Opacus
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(d, c, device)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    loader = DataLoader(train_ds, batch_size=lot, shuffle=True)

    # attach the PrivacyEngine (this wraps the model and optimizer)
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

    # once done, measure accuracy and epsilon
    acc = accuracy(model, Xte_t, yte_t)
    eps = pe.get_epsilon(delta=delta)
    return acc, eps


# ---------------- Load Data ----------------
set_seed(SEED)  # always start clean
df = pd.read_csv(CSV_PATH).drop_duplicates().reset_index(drop=True)

# grab text and labels
X_text, y_raw = df["job_description"], df["job_role"]

# split into train/test (stratified keeps label balance)
Xtr_txt, Xte_txt, ytr_raw, yte_raw = train_test_split(
    X_text, y_raw, test_size=0.3, stratify=y_raw, random_state=SEED
)

# convert text into TF-IDF features
vec = TfidfVectorizer(max_features=MAX_FEAT, ngram_range=(1, 2), stop_words="english")
Xtr = vec.fit_transform(Xtr_txt).toarray().astype(np.float32)
Xte = vec.transform(Xte_txt).toarray().astype(np.float32)

# turn text labels into integers
le = LabelEncoder()
ytr, yte = le.fit_transform(ytr_raw), le.transform(yte_raw)

# move everything to tensors
D, C = Xtr.shape[1], len(le.classes_)
device = "cuda" if torch.cuda.is_available() else "cpu"
Xtr_t, ytr_t = to_tensor(Xtr, device=device), to_tensor(ytr, torch.long, device)
Xte_t, yte_t = to_tensor(Xte, device=device), to_tensor(yte, torch.long, device)
train_ds = TensorDataset(Xtr_t, ytr_t)

# define DP constants
N = len(train_ds)
DELTA = 1.0 / N
DEFAULT_LOT = 50       # batch size
EPOCHS = max(1, N // DEFAULT_LOT)  # one pass over all data


# ---------------- Estimate Median Clip ----------------
# we'll first estimate a good clipping norm to start from
tmp_model = make_model(D, C, device)
base_loader = DataLoader(train_ds, batch_size=DEFAULT_LOT, shuffle=True)
median_clip = estimate_median_grad_norm(tmp_model, base_loader, device)
print(f"Estimated median grad norm: {median_clip:.4f}")

# sweep a few clipping norms around that median (0.5x → 2x)
GRID_CLIP = [round(median_clip * f, 3) for f in np.linspace(0.5, 2.0, 8)]

accs, epss = [], []
for clip in GRID_CLIP:
    # train model for each clipping value and record accuracy & epsilon
    acc, eps = train_dp(D, C, clip, SIGMA, DEFAULT_LOT, EPOCHS, DELTA)
    accs.append(acc)
    epss.append(eps)
    print(f"[Clipping norm={clip}] acc={acc:.3f}, eps={eps:.2f}")


# ---------------- Plot ----------------
# now let's visualize how accuracy changes with clipping norm

# smooth the curve so it looks nicer
clips = np.array(GRID_CLIP)
x_smooth = np.linspace(clips.min(), clips.max(), 200)
y_smooth = make_interp_spline(clips, accs, k=3)(x_smooth)

# find the best (peak) accuracy point
peak_idx = np.argmax(accs)
peak_x, peak_y = clips[peak_idx], accs[peak_idx]

# find median clipping value for reference line
stat_median = np.median(clips)
nearest_median = clips[np.argmin(np.abs(clips - stat_median))]

plt.figure(figsize=(7,5))
plt.plot(x_smooth, y_smooth, 'r-', label="Smoothed Accuracy")
plt.scatter(clips, accs, c='black', label="Observed points")
plt.scatter(peak_x, peak_y, c='green', s=80, label=f"Peak = {peak_y:.3f}")

# add a dashed line for median clipping norm
plt.axvline(nearest_median, color='blue', linestyle='--', label=f"Median ≈ {nearest_median}")
plt.text(nearest_median + 0.002, min(accs) + 0.01, f"Median={nearest_median}", color="blue")

plt.xlabel("Clipping Norm (C)")
plt.ylabel("Test Accuracy")
plt.title("Accuracy vs Clipping Norm under DP-SGD")
plt.legend()
plt.grid(True)

# save the plot so we can include it in reports later
plt.savefig(os.path.join(ART, "clip_vs_acc.png"))
plt.show()
