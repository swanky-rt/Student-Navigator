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


MY BEST
Final baseline model test accuracy: 0.8350
Final DP model test accuracy: 0.8150
'''

# ------------------ Args ------------------
parser = argparse.ArgumentParser()
parser.add_argument("--target_eps", type=float, default=None, help="Target privacy budget ε")
parser.add_argument("--target_delta", type=float, default=None, help="Target δ")
parser.add_argument("--sigma", type=float, default=1.5, help="Noise multiplier σ (used if eps not fixed)")
args = parser.parse_args()

TARGET_EPS = args.target_eps
TARGET_DELTA = args.target_delta
SIGMA = args.sigma

# ------------------ Config ------------------
CSV_PATH = "assignment-3/code/data/dataset.csv"
ART      = "assignment-3/artifacts"
MAX_FEATURES = 258
HIDDEN_UNITS = 128
LAMBDA = 0.02
LR = 0.1
SEED = 24


# ------------------ Utilities ------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_text(df):
    if "job_description" in df.columns:
        return df["job_description"].astype(str)
    return df[df.columns[0]].astype(str)

def make_leak_cleaner(labels_lower):
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

def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        pred = logits.argmax(dim=1)
        return (pred == y).float().mean().item()

def l2_weight_decay(model, lambd):
    if lambd <= 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    reg = torch.tensor(0.0, device=next(model.parameters()).device)
    for p in model.parameters():
        if p.requires_grad:
            reg = reg + p.pow(2).sum()
    return lambd * 0.5 * reg

def to_tensor(array, dtype=torch.float32, device="cpu"):
    return torch.tensor(array, dtype=dtype, device=device)

def make_model(input_dim, num_classes, device):
    return nn.Sequential(
        nn.Linear(input_dim, HIDDEN_UNITS),
        nn.ReLU(),
        nn.Linear(HIDDEN_UNITS, num_classes)
    ).to(device)

def estimate_median_grad_norm(model, loader, device, num_batches=50):
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

# ------------------ Main ------------------
def main():
    set_seed(SEED)
    os.makedirs(ART, exist_ok=True)

    # ------------------ Load data ------------------
    df = pd.read_csv(CSV_PATH).drop_duplicates().reset_index(drop=True)
    text_raw = build_text(df)
    labels = df["job_role"].astype(str)
    cleaner = make_leak_cleaner(labels.str.lower().unique())
    text = text_raw.apply(cleaner)

    Xtr_txt, Xte_txt, ytr_raw, yte_raw = train_test_split(
        text, labels.values, test_size=0.2, random_state=SEED, stratify=labels.values
    )

    vec = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1,2), stop_words="english")
    Xtr = vec.fit_transform(Xtr_txt).toarray().astype(np.float32)
    Xte = vec.transform(Xte_txt).toarray().astype(np.float32)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    ytr = le.fit_transform(ytr_raw).astype(np.int64)
    yte = le.transform(yte_raw).astype(np.int64)

    D, C = Xtr.shape[1], len(le.classes_)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr_t, ytr_t = to_tensor(Xtr, device=device), to_tensor(ytr, dtype=torch.long, device=device)
    Xte_t, yte_t = to_tensor(Xte, device=device), to_tensor(yte, dtype=torch.long, device=device)
    train_ds = TensorDataset(Xtr_t, ytr_t)

    # ------------------ Lot size & epochs ------------------
    N = len(Xtr_t)
    LOT_SIZE = 60
    # int(np.sqrt(N))
    EPOCHS = max(1, N // LOT_SIZE)
    base_loader = DataLoader(train_ds, batch_size=LOT_SIZE, shuffle=True, drop_last=True)

    # ------------------ Median grad norm ------------------
    model_tmp = make_model(D, C, device)
    MAX_GRAD_NORM = 0.17
    # estimate_median_grad_norm(model_tmp, base_loader, device)
   
    print(f"[clip] Using gradient norm bound C = {MAX_GRAD_NORM:.3f}")

    # ------------------ Baseline ------------------
    print("[baseline] Training non-DP model...")
    model_base = make_model(D, C, device)
    opt_base = torch.optim.SGD(model_base.parameters(), lr=LR)
    train_hist_base, test_hist_base = [], []
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

    pd.DataFrame({
        "epoch": np.arange(1,EPOCHS+1),
        "train_acc": train_hist_base,
        "test_acc": test_hist_base
    }).to_csv(os.path.join(ART,"baseline_accuracy.csv"), index=False)


    # ------------------ DP Model ------------------
    delta = TARGET_DELTA if TARGET_DELTA else 1.0 / N
    eps_hist = []

    if TARGET_EPS is not None and TARGET_DELTA is not None:
        # Case B: fix ε and δ
        model = make_model(D, C, device)
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)
        privacy_engine = PrivacyEngine()
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

    elif TARGET_DELTA is not None:
        # Case A: fix δ only
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

    else:
        # Case C: neither fixed
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

    # ------------------ Train DP model ------------------
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

    pd.DataFrame({
        "epoch": np.arange(1,EPOCHS+1),
        "train_acc": train_hist_dp,
        "test_acc": test_hist_dp,
        "epsilon": eps_hist if plot_eps else [TARGET_EPS]*EPOCHS
    }).to_csv(os.path.join(ART,"dp_accuracy.csv"), index=False)

    # Print final baseline test accuracy
    print(f"Final baseline model test accuracy: {test_hist_base[-1]:.4f}")

    # Print final DP test accuracy
    print(f"Final DP model test accuracy: {test_hist_dp[-1]:.4f}")


    plt.figure(figsize=(8,5))
    epochs = np.arange(1, EPOCHS+1)

    # ------------------ Plots: Train + Test for both Baseline and DP ------------------
    plt.figure(figsize=(8,6))
    epochs = np.arange(1, EPOCHS+1)

    # Baseline Train/Test
    plt.plot(epochs, train_hist_base, label="Baseline Train", color="blue", linestyle="--")
    plt.plot(epochs, test_hist_base, label="Baseline Test", color="orange")

    # DP Train/Test
    plt.plot(epochs, train_hist_dp, label="DP Train", color="green")
    plt.plot(epochs, test_hist_dp, label="DP Test", color="red")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Baseline vs DP Train/Test Accuracy\n"
              f"DP params: clip={MAX_GRAD_NORM:.2f}, lot size={LOT_SIZE}, δ=1/N, σ={sigma_used}")
    plt.grid(True)
    plt.legend()
    # plt.savefig(os.path.join(ART,"baseline_vs_dp_train_test.png"))
    plt.show()


    if plot_eps:
        plt.figure(figsize=(8,5))
        plt.plot(epochs, eps_hist, label=f"ε (δ=1/N)", color="green")
        plt.text(epochs[-1], eps_hist[-1]+0.1,
                f"Final: {eps_hist[-1]:.2f}", fontsize=9, ha='center', color="green")
        plt.xlabel("Epoch"); plt.ylabel("Epsilon (ε)")
        plt.title("Privacy consumption over epochs")
        plt.grid(True); plt.legend()
        plt.savefig(os.path.join(ART,"epsilon_curve_final.png"))
        plt.show()

    # Save models for MIA in ART folder after both are trained
    torch.save(model_base.state_dict(), os.path.join(ART, "baseline_model.pth"))
    try:
        torch.save(model.state_dict(), os.path.join(ART, "dp_model.pth"))
    except Exception as e:
        print("[Warning] DP model not saved:", e)


if __name__ == "__main__":
    main()
