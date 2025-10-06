# Differentially Private (DP-SGD) Training Script
# ------------------------------------------------
# Trains a simple 2-layer neural network using Opacus for differential privacy.
# The model classifies job descriptions into roles while ensuring privacy by
# applying per-example gradient clipping and Gaussian noise during training.

import os, time, joblib
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from opacus import PrivacyEngine
from pathlib import Path

# --- Paths and configuration ---
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
ART  = ROOT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

CSV_PATH = ROOT / "code" / "data" / "dataset.csv"
MODEL_DP_PT = ART / "model_dp.pt"
VECTORIZER_DP = ART / "vectorizer_dp.joblib"
LABELER_DP    = ART / "label_encoder_dp.joblib"

# --- Hyperparameters ---
SEED   = 42
MAX_FEAT = 20000         # TF-IDF vocabulary size
NGRAMS = (1, 3)          # Use unigrams, bigrams, trigrams
HIDDEN = 512             # Hidden layer size
LR_DP  = 0.15            # Learning rate
SIGMA  = 4.0             # Noise multiplier for DP-SGD
CLIP   = 1.0             # Gradient clipping threshold
DP_EPOCHS = 4            # Number of training epochs

# --- Helper functions ---

def set_seed(s):
    """Set all random seeds for reproducibility."""
    np.random.seed(s)          # NumPy random seed
    torch.manual_seed(s)       # PyTorch CPU seed
    torch.cuda.manual_seed_all(s)  # PyTorch GPU seed

def to_tensor(x, dtype=torch.float32, device="cpu"):
    """Convert NumPy arrays to PyTorch tensors."""
    return torch.tensor(x, dtype=dtype, device=device)

def make_model(d, c, device, hidden=HIDDEN):
    """Define a simple 2-layer feedforward network."""
    return nn.Sequential(
        nn.Linear(d, hidden),  # Input to hidden layer
        nn.ReLU(),            # Activation function
        nn.Linear(hidden, c)  # Hidden to output layer
    ).to(device)

def accuracy(model, X, y):
    """Compute model accuracy on a given dataset."""
    model.eval()  
    with torch.no_grad():
        preds = model(X).argmax(1) 
        return (preds == y).float().mean().item()  # Calculate accuracy

# --- Main DP training loop ---

def main():
    # Set the random seed
    set_seed(SEED)

    # Load the dataset
    df = pd.read_csv(CSV_PATH).drop_duplicates().reset_index(drop=True)  # Load and clean data
    X_text, y_raw = df["job_description"], df["job_role"]  # Extract text and labels

    # Split into train and test sets (stratified by label)
    Xtr_txt, Xte_txt, ytr_raw, yte_raw = train_test_split(
        X_text, y_raw, test_size=0.2, stratify=y_raw, random_state=SEED  # 80/20 split
    )

    # Convert text to numerical features using TF-IDF
    vec = TfidfVectorizer(max_features=MAX_FEAT, ngram_range=NGRAMS, stop_words="english")
    Xtr = vec.fit_transform(Xtr_txt).toarray().astype(np.float32)  # Fit on train, get features
    Xte = vec.transform(Xte_txt).toarray().astype(np.float32)      # Transform test using same vocab

    # Encode job role labels to integers
    le = LabelEncoder()
    ytr = le.fit_transform(ytr_raw)  # Fit and transform training labels
    yte = le.transform(yte_raw)      # Transform test labels

    # Save preprocessing components
    joblib.dump(vec, VECTORIZER_DP)
    joblib.dump(le, LABELER_DP)
    print(f"[DP] saved vectorizer -> {VECTORIZER_DP}")
    print(f"[DP] saved labeler    -> {LABELER_DP}")

    # Set device and dataset dimensions
    D, C = Xtr.shape[1], len(le.classes_)  # D=features, C=num_classes
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
    print(f"[DP] D={D}, C={C}, device={device}")

    # Convert to tensors for PyTorch
    Xtr_t = to_tensor(Xtr, device=device)           # Training features
    ytr_t = to_tensor(ytr, torch.long, device=device)  # Training labels

    # Prepare DataLoader
    ds = TensorDataset(Xtr_t, ytr_t)  # Combine features and labels
    bs = max(4, int(np.sqrt(len(ds))))  
    dl = DataLoader(ds, batch_size=bs, shuffle=True)  # Create data loader

    # Initialize model and optimizer
    model = make_model(D, C, device) 
    opt = torch.optim.SGD(model.parameters(), lr=LR_DP)  # SGD optimizer

    # Attach Opacus PrivacyEngine for DP-SGD
    pe = PrivacyEngine()  # Create privacy engine
    model, opt, dl = pe.make_private(  # Make training differentially private
        module=model,
        optimizer=opt,
        data_loader=dl,
        noise_multiplier=SIGMA,  # Controls noise level
        max_grad_norm=CLIP       # Gradient clipping threshold
    )

    # Compute delta (standard 1/N value)
    delta = 1.0 / len(ds)  # Privacy parameter delta
    eps_hist = []          # Track epsilon over epochs

    # --- Training loop with DP accounting ---
    for ep in range(1, DP_EPOCHS + 1):
        model.train()  # Set to training mode
        for xb, yb in dl:
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()  # Backpropagation (with DP noise added by Opacus)
            opt.step()

        # Track epsilon for current epoch
        eps = pe.get_epsilon(delta=delta)  # Get privacy budget used so far
        eps_hist.append(float(eps))
        print(f"[DP] epoch {ep}/{DP_EPOCHS}  eps≈{eps:.2f}")

    # Save model state
    sd = model._module.state_dict() if hasattr(model, "_module") else model.state_dict()  # Extract weights
    torch.save({"state_dict": sd, "D": D, "C": C, "eps_hist": eps_hist}, MODEL_DP_PT)  # Save checkpoint
    print(f"[DP] saved model -> {MODEL_DP_PT}")

# Entry point
if __name__ == "__main__":
    main()
