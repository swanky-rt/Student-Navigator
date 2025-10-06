# ---------------------------------------------------------------
# This script evaluates how much the Differentially Private (DP)
# model leaks training information by running a loss-threshold
# membership inference attack (Yeom et al., 2018).
# ---------------------------------------------------------------

import json, numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import joblib

# Location of saved DP models, vectorizers, and plots
ART = Path(__file__).resolve().parents[1] / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------

def _nll_from_proba(proba, y):
    """Compute negative log-likelihood (cross-entropy) per sample."""
    eps = 1e-12  # Small value to prevent log(0)
    p = np.take_along_axis(np.clip(proba, eps, 1 - eps), y.reshape(-1, 1), axis=1).ravel()  # Get probability for true class
    return -np.log(p)  # Return negative log-likelihood

def loss_threshold_attack(proba_in, y_in, proba_out, y_out):
    """
    Run the loss-threshold attack:
    - Compute per-sample loss for both training (in) and test (out) sets
    - Combine them into scores (higher = more 'member-like')
    - Return AUC for how well losses separate train vs. test
    """
    loss_in  = _nll_from_proba(proba_in,  y_in)  # Loss on training data (should be lower)
    loss_out = _nll_from_proba(proba_out, y_out)  # Loss on test data (should be higher)
    scores   = np.concatenate([-loss_in, -loss_out])  # Negate losses: lower loss = higher membership score
    labels   = np.concatenate([np.ones_like(loss_in), np.zeros_like(loss_out)])  # 1=member, 0=non-member
    auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else float("nan")  # Measure attack effectiveness
    return scores, labels, float(auc)

def make_model_for_ckpt(D, C, device):
    """Recreate model architecture used during DP training."""
    import torch.nn as nn
    HIDDEN = 512  # Hidden layer size must match training
    return nn.Sequential(nn.Linear(D, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, C)).to(device)  # Simple 2-layer MLP

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    print(f"[paths] ART -> {ART}")

    # Check if the vectorizer and label encoder exist
    vec_dp = ART / "vectorizer_dp.joblib"
    le_dp  = ART / "label_encoder_dp.joblib"
    if not vec_dp.exists() or not le_dp.exists():
        raise FileNotFoundError(
            "Missing DP vectorizer/labeler. Run dp_train.py first so it creates "
            "vectorizer_dp.joblib and label_encoder_dp.joblib in artifacts/."
        )

    # Load TF-IDF vectorizer and label encoder from DP training
    vec = joblib.load(vec_dp)
    le  = joblib.load(le_dp)

    # Load dataset and split 50/50 for attack evaluation
    csv = Path(__file__).resolve().parents[0] / "data" / "dataset.csv"
    df = pd.read_csv(csv).drop_duplicates().reset_index(drop=True)
    X_text, y_raw = df.iloc[:, 0], df.iloc[:, 1]

    Xtr_txt, Xte_txt, ytr_raw, yte_raw = train_test_split(
        X_text, y_raw, test_size=0.5, stratify=y_raw, random_state=42
    )

    # Vectorize text and encode labels
    Xtr = vec.transform(Xtr_txt).toarray().astype(np.float32)
    Xte = vec.transform(Xte_txt).toarray().astype(np.float32)
    ytr = le.transform(ytr_raw)  # Convert text labels to integers
    yte = le.transform(yte_raw)

    D, C = Xtr.shape[1], len(le.classes_)  # D = feature dims, C = num classes
    dev = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available

    # ---------------------------------------------------------------
    # Load the trained DP model from checkpoint
    # ---------------------------------------------------------------
    ckpt_path = ART / "model_dp.pt"  # Path to saved DP model
    ckpt = torch.load(ckpt_path, map_location=dev)  # Load checkpoint

    # Sanity check for feature dimension mismatch
    if D != ckpt["D"]:
        raise RuntimeError(
            f"TF-IDF dimension mismatch: features={D} but checkpoint expects {ckpt['D']}.\n"
            f"Make sure dp_train.py and this script both use the SAME vectorizer_dp.joblib."
        )

    # Rebuild model and load weights
    model = make_model_for_ckpt(D, C, dev)  # Create model with same architecture as training
    sd = ckpt["state_dict"]  # Get saved model weights
    if any(k.startswith("_module.") for k in sd.keys()):  # Handle DataParallel wrapper keys
        sd = {k.replace("_module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)  # Load the trained weights
    model.eval()  # Set to evaluation mode

    # ---------------------------------------------------------------
    # Run attack on the DP-trained model
    # ---------------------------------------------------------------
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=dev)  # Convert training data to tensor
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=dev)  # Convert test data to tensor

    with torch.no_grad():  # Disable gradient computation for inference
        p_in  = torch.softmax(model(Xtr_t), dim=1).cpu().numpy()  # Get probabilities for training data
        p_out = torch.softmax(model(Xte_t), dim=1).cpu().numpy()  # Get probabilities for test data

    # Compute scores and AUC for POST-DP attack
    scores_post, labels_post, auc_post = loss_threshold_attack(p_in, ytr, p_out, yte)  # Run the membership inference attack
    np.savez(ART / "post_loss_threshold_attack_scores_labels.npz",  # Save attack results
             scores=scores_post, labels=labels_post, auc=auc_post)
    print(f"[post_loss_threshold_attack_scores] AUC={auc_post:.3f}")

    # ---------------------------------------------------------------
    # Compare with pre-DP (non-private) results
    # ---------------------------------------------------------------
    pre_npz = ART / "pre_attack_scores_labels.npz"  # Path to pre-DP attack results
    plt.figure(figsize=(7,6))  # Create new plot

    if pre_npz.exists():  # If pre-DP results exist, load and plot them
        pre = np.load(pre_npz, allow_pickle=False)
        fpr_pre, tpr_pre, _ = roc_curve(pre["labels"], pre["scores"])  # Calculate ROC curve for pre-DP
        plt.plot(fpr_pre, tpr_pre, label=f"PRE (AUC={float(pre['auc']):.3f})")  # Plot pre-DP ROC

    fpr_post, tpr_post, _ = roc_curve(labels_post, scores_post)  # Calculate ROC curve for post-DP
    plt.plot(fpr_post, tpr_post, label=f"POST-DP (AUC={auc_post:.3f})", linestyle="--")  # Plot post-DP ROC

    plt.plot([0,1],[0,1],'k--', lw=1)  # Plot diagonal line (random guess)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Threshold Attack ROC: PRE vs POST-DP") 
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ART / "pre_vs_post_attack_comparison.png")  # Save plot
    plt.close()  # Close plot

    # Save summary comparison as JSON for reference
    with open(ART / "pre_post_attack_summary.json", "w") as f:  # Open JSON file for writing
        out = {"post": {"auc": auc_post}}  # Store post-DP AUC
        if pre_npz.exists():  # If pre-DP results exist, include them
            out["pre"] = {"auc": float(np.load(pre_npz)["auc"])}
        json.dump(out, f, indent=2)  # Write JSON summary

    print("[POST] wrote overlay plot ->", (ART / "pre_vs_post_attack_comparison.png").as_posix())  # Print completion message


if __name__ == "__main__":
    main()
