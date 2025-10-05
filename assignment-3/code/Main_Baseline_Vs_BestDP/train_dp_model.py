import os, re, numpy as np, pandas as pd
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from opacus import PrivacyEngine
import matplotlib.pyplot as plt
import argparse

'''
PAPER BEST:
Final baseline model test accuracy: 0.8313
Final DP model test accuracy: 0.7925

MY BEST:
Final baseline model test accuracy: 0.8350
Final DP model test accuracy: 0.8150
'''

# ------------------ Argument Parser ------------------
# I added argparse so I can control ε, δ, or σ directly from terminal while running experiments.
parser = argparse.ArgumentParser()
parser.add_argument("--target_eps", type=float, default=None, help="Target privacy budget ε")
parser.add_argument("--target_delta", type=float, default=None, help="Target δ")
parser.add_argument("--sigma", type=float, default=1.5, help="Noise multiplier σ (used if eps not fixed)")
args = parser.parse_args()

# These control the privacy mechanism setup
TARGET_EPS = args.target_eps      # Optional fixed epsilon target
TARGET_DELTA = args.target_delta  # Optional fixed delta
SIGMA = args.sigma                # Noise level for DP-SGD (higher = more private, less accurate)

# ------------------ Basic Config ------------------
CSV_PATH = "assignment-3/code/data/dataset.csv"
ART      = "assignment-3/artifacts"
MAX_FEATURES = 258        # TF-IDF feature size cap — helps prevent overfitting and speed issues
HIDDEN_UNITS = 128        # Hidden layer width — simple but powerful enough for text classification
LAMBDA = 0.02             # Weight decay factor for regularization
LR = 0.1                  # Learning rate for SGD
SEED = 24                 # Random seed for reproducibility


# ------------------ Utility Functions ------------------
def set_seed(seed: int):
    # Reproducibility across NumPy, CPU, and GPU
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_text(df):
    # Some datasets might have a different text column; this ensures flexibility
    if "job_description" in df.columns:
        return df["job_description"].astype(str)
    return df[df.columns[0]].astype(str)

def make_leak_cleaner(labels_lower):
    """
    Removes any words from job descriptions that also appear in the job_role labels.
    This prevents the model from trivially learning job titles directly from the text.
    """
    toks = set()
    for lbl in labels_lower:
        toks |= set(re.split(r"\s+", str(lbl).strip().lower()))
    toks = {t for t in toks if t}
    # Compile regex for each label token
    pats = [re.compile(rf"\b{re.escape(t)}\b", re.I) for t in toks]
    def clean(s: str):
        s = str(s).lower()
        for p in pats:
            s = p.sub(" ", s)
        # Collapse extra spaces after cleaning
        return re.sub(r"\s+", " ", s).strip()
    return clean

def accuracy(model, X, y):
    # Standard accuracy computation — no gradients needed here
    model.eval()
    with torch.no_grad():
        logits = model(X)
        pred = logits.argmax(dim=1)
        return (pred == y).float().mean().item()

def l2_weight_decay(model, lambd):
    # Simple manual L2 regularization — added to loss during DP training
    if lambd <= 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    reg = torch.tensor(0.0, device=next(model.parameters()).device)
    for p in model.parameters():
        if p.requires_grad:
            reg += p.pow(2).sum()
    return lambd * 0.5 * reg

def to_tensor(array, dtype=torch.float32, device="cpu"):
    # Converts numpy arrays into PyTorch tensors, placing them on CPU/GPU
    return torch.tensor(array, dtype=dtype, device=device)

def make_model(input_dim, num_classes, device):
    # My main MLP — 1 hidden layer with ReLU; small, interpretable architecture
    return nn.Sequential(
        nn.Linear(input_dim, HIDDEN_UNITS),
        nn.ReLU(),
        nn.Linear(HIDDEN_UNITS, num_classes)
    ).to(device)

def estimate_median_grad_norm(model, loader, device, num_batches=50):
    # Roughly estimates the median gradient norm — helps pick a clipping threshold for DP
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
        total_norm = torch.norm(
            torch.stack([p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None])
        ).item()
        norms.append(total_norm)
    return np.median(norms)


# ------------------ Main Script ------------------
def main():
    set_seed(SEED)
    os.makedirs(ART, exist_ok=True)

    # ------------------ Load and preprocess dataset ------------------
    df = pd.read_csv(CSV_PATH).drop_duplicates().reset_index(drop=True)

    # Clean up job descriptions and remove label-related tokens
    text_raw = build_text(df)
    labels = df["job_role"].astype(str)
    cleaner = make_leak_cleaner(labels.str.lower().unique())
    text = text_raw.apply(cleaner)

    # Train-test split with stratification to preserve label distribution
    Xtr_txt, Xte_txt, ytr_raw, yte_raw = train_test_split(
        text, labels.values, test_size=0.2, random_state=SEED, stratify=labels.values
    )

    # Convert text into TF-IDF features (bigrams + unigrams)
    vec = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1,2), stop_words="english")
    Xtr = vec.fit_transform(Xtr_txt).toarray().astype(np.float32)
    Xte = vec.transform(Xte_txt).toarray().astype(np.float32)

    # Encode target labels numerically
    le = LabelEncoder()
    ytr = le.fit_transform(ytr_raw).astype(np.int64)
    yte = le.transform(yte_raw).astype(np.int64)

    D, C = Xtr.shape[1], len(le.classes_)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert everything to PyTorch tensors
    Xtr_t, ytr_t = to_tensor(Xtr, device=device), to_tensor(ytr, dtype=torch.long, device=device)
    Xte_t, yte_t = to_tensor(Xte, device=device), to_tensor(yte, dtype=torch.long, device=device)
    train_ds = TensorDataset(Xtr_t, ytr_t)

    # ------------------ Training setup ------------------
    N = len(Xtr_t)
    LOT_SIZE = 60                      # batch size (lot size for DP-SGD)
    EPOCHS = max(1, N // LOT_SIZE)     # typical DP formula — one pass per sample roughly once
    base_loader = DataLoader(train_ds, batch_size=LOT_SIZE, shuffle=True, drop_last=True)

    MAX_GRAD_NORM = 0.17               # Clipping threshold C for DP — chosen from prior experiments
    print(f"[clip] Using gradient norm bound C = {MAX_GRAD_NORM:.3f}")

    # ------------------ Non-private Baseline Training ------------------
    print("[baseline] Training non-DP model...")
    model_base = make_model(D, C, device)
    opt_base = torch.optim.SGD(model_base.parameters(), lr=LR)
    train_hist_base, test_hist_base = [], []

    # Standard SGD loop
    for ep in range(EPOCHS):
        model_base.train()
        correct, total = 0, 0
        for xb, yb in base_loader:
            opt_base.zero_grad(set_to_none=True)
            logits = model_base(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt_base.step()
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
        train_hist_base.append(correct / total)
        test_hist_base.append(accuracy(model_base, Xte_t, yte_t))

    # Save baseline results
    pd.DataFrame({
        "epoch": np.arange(1,EPOCHS+1),
        "train_acc": train_hist_base,
        "test_acc": test_hist_base
    }).to_csv(os.path.join(ART,"baseline_accuracy.csv"), index=False)


    # ------------------ Differentially Private Model ------------------
    # delta = probability of privacy failure; usually set to 1/N
    delta = TARGET_DELTA if TARGET_DELTA else 1.0 / N
    eps_hist = []

    # --- Case 1: Fixed epsilon and delta (automatically compute noise σ) ---
    if TARGET_EPS is not None and TARGET_DELTA is not None:
        model = make_model(D, C, device)
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)
        privacy_engine = PrivacyEngine()
        # This call automatically tunes σ to meet (ε, δ)
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=base_loader,
            target_epsilon=TARGET_EPS,
            target_delta=delta,
            epochs=EPOCHS,
            max_grad_norm=MAX_GRAD_NORM,
        )
        sigma_used = optimizer.noise_multiplier
        plot_eps = False
        print(f"[dp] Running with fixed (ε={TARGET_EPS}, δ={delta})")

    # --- Case 2: Fixed δ only — use given σ ---
    elif TARGET_DELTA is not None:
        model = make_model(D, C, device)
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=base_loader,
            noise_multiplier=SIGMA,
            max_grad_norm=MAX_GRAD_NORM,
            target_delta=0.00025,
        )
        sigma_used = SIGMA
        plot_eps = True
        print(f"[dp] Running with δ={delta}, σ={SIGMA}")

    # --- Case 3: Default δ and σ (no fixed privacy budget) ---
    else:
        delta = 1.0 / N
        model = make_model(D, C, device)
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=base_loader,
            noise_multiplier=SIGMA,
            max_grad_norm=MAX_GRAD_NORM,
            target_delta=0.00025,
        )
        sigma_used = SIGMA
        plot_eps = True
        print(f"[dp] Running with default δ={delta}, σ={SIGMA}")

    # ------------------ Train DP Model ------------------
    train_hist_dp, test_hist_dp = [], []
    for ep in range(EPOCHS):
        model.train()
        correct, total = 0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb) + l2_weight_decay(model, LAMBDA)
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
        train_hist_dp.append(correct / total)
        test_hist_dp.append(accuracy(model, Xte_t, yte_t))
        if plot_eps:
            eps_hist.append(privacy_engine.get_epsilon(delta=delta))

    # Save DP training curves
    pd.DataFrame({
        "epoch": np.arange(1,EPOCHS+1),
        "train_acc": train_hist_dp,
        "test_acc": test_hist_dp,
        "epsilon": eps_hist if plot_eps else [TARGET_EPS]*EPOCHS
    }).to_csv(os.path.join(ART,"dp_accuracy.csv"), index=False)

    # ------------------ Delta Sensitivity Analysis ------------------
    # I wanted to check how accuracy and epsilon scale when δ changes.
    def train_dp_for_delta(delta, sigma):
        """Trains one DP model for a given delta and returns (eps_hist, acc_hist)."""
        model = make_model(D, C, device)
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=base_loader,
            noise_multiplier=sigma,
            max_grad_norm=MAX_GRAD_NORM,
        )
        eps_hist, acc_hist = [], []
        for _ in range(EPOCHS):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = F.cross_entropy(logits, yb) + l2_weight_decay(model, LAMBDA)
                loss.backward()
                optimizer.step()
            eps_hist.append(privacy_engine.get_epsilon(delta=delta))
            acc_hist.append(accuracy(model, Xte_t, yte_t))
        return eps_hist, acc_hist

    delta_list = [1.0/len(Xtr_t), 1e-3, 5e-4, 1e-4, 5e-5]
    rows = []
    plt.figure(figsize=(8,6))
    for delta in delta_list:
        eps_hist, acc_hist = train_dp_for_delta(delta, SIGMA)
        rows += [{"delta": delta, "epoch": i+1, "epsilon": e, "test_acc": a,
                  "sigma": SIGMA, "clip": MAX_GRAD_NORM, "lot_size": LOT_SIZE}
                 for i,(e,a) in enumerate(zip(eps_hist, acc_hist))]
        plt.plot(eps_hist, acc_hist, marker="o", label=f"δ={delta:g}")
    plt.xlabel("Epsilon (ε)")
    plt.ylabel("Test Accuracy")
    plt.title("Delta Sensitivity: Accuracy vs Epsilon (σ fixed)")
    plt.grid(True)
    plt.legend()
    pd.DataFrame(rows).to_csv(os.path.join(ART,"delta_sweep.csv"), index=False)
    plt.savefig(os.path.join(ART,"delta_sensitivity_acc_vs_eps.png"), bbox_inches="tight")
    plt.show()

    # ------------------ Final Evaluation ------------------
    print(f"Final baseline model test accuracy: {test_hist_base[-1]:.4f}")
    print(f"Final DP model test accuracy: {test_hist_dp[-1]:.4f}")

    # ------------------ Compare Baseline vs DP ------------------
    plt.figure(figsize=(8,6))
    epochs = np.arange(1, EPOCHS+1)
    plt.plot(epochs, train_hist_base, label="Baseline Train", color="blue", linestyle="--")
    plt.plot(epochs, test_hist_base, label="Baseline Test", color="orange")
    plt.plot(epochs, train_hist_dp, label="DP Train", color="green")
    plt.plot(epochs, test_hist_dp, label="DP Test", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Baseline vs DP Accuracy\nclip={MAX_GRAD_NORM:.2f}, lot={LOT_SIZE}, δ=1/N, σ={sigma_used}")
    plt.grid(True)
    plt.legend()
    plt.show()

    # ------------------ Privacy Tracking (ε over epochs) ------------------
    if plot_eps:
        plt.figure(figsize=(8,5))
        plt.plot(epochs, eps_hist, label=f"ε (δ=1/N)", color="green")
        plt.text(epochs[-1], eps_hist[-1]+0.1, f"Final: {eps_hist[-1]:.2f}",
                 fontsize=9, ha='center', color="green")
        plt.xlabel("Epoch")
        plt.ylabel("Epsilon (ε)")
        plt.title("Privacy consumption over epochs")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(ART,"epsilon_curve_final.png"))
        plt.show()


# ------------------ Entry Point ------------------
if __name__ == "__main__":
    main()
