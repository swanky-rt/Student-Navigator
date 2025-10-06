# This script trains a simple text classifier on a small fraction of data
# and evaluates how well a membership inference attack (MIA) can distinguish
# between training and test samples using loss values.
#
# AUC close to 1.0 → strong privacy leakage (model overfits)
# AUC near 0.5     → no leakage (model generalizes well)
# -------------------------------------------------------------

import argparse, time, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from pathlib import Path

# Folder where we save all experiment artifacts
ART = Path(__file__).resolve().parents[1] / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

# --- Utility functions ---

def set_seed(s):
    """Ensure reproducibility by setting all relevant random seeds."""
    torch.manual_seed(s)  # PyTorch random seed
    np.random.seed(s)     # NumPy random seed

@torch.no_grad()
def accuracy(model, X_t, y_t):
    """Compute model accuracy on a dataset."""
    model.eval()  # Set to eval mode
    return (model(X_t).argmax(1) == y_t).float().mean().item()  # Calculate accuracy

def _nll_from_proba(proba, y):
    """Compute per-sample negative log-likelihood (cross-entropy loss)."""
    eps = 1e-12  # Prevent log(0)
    p = np.take_along_axis(np.clip(proba, eps, 1 - eps), y.reshape(-1, 1), axis=1).ravel()  # Get true class prob
    return -np.log(p)  # Return NLL

def loss_threshold_attack(proba_in, y_in, proba_out, y_out, name="Yeom-Loss"):
    """
    Perform Yeom loss-threshold attack.
    Train (members) → expect lower loss
    Test (non-members) → expect higher loss
    """
    loss_in  = _nll_from_proba(proba_in,  y_in)   # Loss on training data
    loss_out = _nll_from_proba(proba_out, y_out)  # Loss on test data
    scores   = np.concatenate([-loss_in, -loss_out])  # higher score ⇒ more "member-like"
    labels   = np.concatenate([np.ones_like(loss_in), np.zeros_like(loss_out)])  # 1=member, 0=non-member
    auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else float("nan")  # Calculate AUC
    return {"name": name, "auc": float(auc)}

def make_big_model(d, c, device):
    """Define a large MLP to encourage overfitting for stronger privacy leakage."""
    m = nn.Sequential(
        nn.Linear(d, 4096), nn.ReLU(),    # First hidden layer
        nn.Linear(4096, 1024), nn.ReLU(), # Second hidden layer
        nn.Linear(1024, c)                # Output layer
    ).to(device)
    return m

# --- Main experiment ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="assignment-3/code/data/dataset.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.05, help="Fraction used as TRAIN (members).")
    ap.add_argument("--max-feat", type=int, default=100_000)
    ap.add_argument("--ngrams", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.30)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    # --- Setup and data preparation ---
    set_seed(args.seed)
    df = pd.read_csv(args.csv).drop_duplicates().reset_index(drop=True)
    X_text, y_raw = df.iloc[:,0], df.iloc[:,1]

    # Small training fraction to induce overfitting and increase leakage
    Xtr_txt, Xte_txt, ytr_raw, yte_raw = train_test_split(
        X_text, y_raw, train_size=args.train_frac, stratify=y_raw, random_state=args.seed
    )

    # Convert text data into TF-IDF features
    vec = TfidfVectorizer(
        max_features=args.max_feat,
        ngram_range=(1, args.ngrams),
        stop_words=None,
        lowercase=True
    )
    Xtr = vec.fit_transform(Xtr_txt).toarray().astype(np.float32)
    Xte = vec.transform(Xte_txt).toarray().astype(np.float32)

    # Encode job roles as integers
    le = LabelEncoder()
    ytr = le.fit_transform(ytr_raw)  # Convert labels to integers
    yte = le.transform(yte_raw)

    d, c = Xtr.shape[1], len(le.classes_)  # d=features, c=num_classes
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available

    # Convert data to PyTorch tensors
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)  # Training features
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)     # Training labels
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)  # Test features
    yte_t = torch.tensor(yte, dtype=torch.long, device=device)     # Test labels

    # Create DataLoader for batching
    dl = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=args.batch, shuffle=True, num_workers=0)

    # Build and train model
    model = make_big_model(d, c, device)  # Create large model for overfitting
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)  # SGD optimizer
    model.train()  # Set to training mode

    last = time.time()
    for ep in range(1, args.epochs + 1):  # Training loop
        for xb, yb in dl:  # For each batch
            opt.zero_grad()  # Clear gradients
            loss = F.cross_entropy(model(xb), yb)  # Compute loss
            loss.backward()  # Backpropagation
            opt.step()      # Update weights

        # Print status periodically
        if time.time() - last > 1.0:
            last = time.time()

    # --- Evaluation phase ---
    tr_acc = accuracy(model, Xtr_t, ytr_t)  # Training accuracy
    te_acc = accuracy(model, Xte_t, yte_t)  # Test accuracy

    with torch.no_grad():  # No gradient computation needed
        p_in  = torch.softmax(model(Xtr_t), dim=1).cpu().numpy()  # Probabilities for training data
        p_out = torch.softmax(model(Xte_t), dim=1).cpu().numpy()  # Probabilities for test data

    # Run the loss-threshold membership inference attack
    res = loss_threshold_attack(p_in, ytr, p_out, yte)  # Perform attack

    # Save attack data (scores and labels)
    loss_in  = _nll_from_proba(p_in,  ytr)   # Training losses
    loss_out = _nll_from_proba(p_out, yte)   # Test losses
    scores   = np.concatenate([-loss_in, -loss_out])  # Attack scores
    labels   = np.concatenate([np.ones_like(loss_in), np.zeros_like(loss_out)])  # Ground truth

    np.savez(ART / "pre_attack_scores_labels.npz", scores=scores, labels=labels, auc=res["auc"])  # Save results

    # Plot ROC curve for visualization
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure(figsize=(7,6))
    plt.plot(fpr, tpr, label=f"PRE (AUC={res['auc']:.3f})")
    plt.plot([0,1], [0,1], 'k--', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Loss-Threshold Attack ROC: PRE")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ART / "loss-threshold-attack.png")
    plt.close()

    print("\nHint: If AUC < 0.70, try adjusting parameters, e.g.:")
    print("  --train-frac 0.02   --epochs 800   --lr 0.4   --batch 2")
    print("  --max-feat 150000   --ngrams 5\n")

# Entry point
if __name__ == "__main__":
    main()
