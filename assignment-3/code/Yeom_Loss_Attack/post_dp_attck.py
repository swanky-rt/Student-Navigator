#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import joblib

ART = Path(__file__).resolve().parents[1] / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

def _nll_from_proba(proba, y):
    eps = 1e-12
    p = np.take_along_axis(np.clip(proba, eps, 1 - eps), y.reshape(-1, 1), axis=1).ravel()
    return -np.log(p)

def yeom_loss_attack(proba_in, y_in, proba_out, y_out):
    loss_in  = _nll_from_proba(proba_in,  y_in)
    loss_out = _nll_from_proba(proba_out, y_out)
    scores   = np.concatenate([-loss_in, -loss_out])
    labels   = np.concatenate([np.ones_like(loss_in), np.zeros_like(loss_out)])
    auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else float("nan")
    return scores, labels, float(auc)

def make_model_for_ckpt(D, C, device):
    import torch.nn as nn

    HIDDEN = 512
    return nn.Sequential(nn.Linear(D, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, C)).to(device)

def main():
    print(f"[paths] ART -> {ART}")

    vec_dp = ART / "vectorizer_dp.joblib"
    le_dp  = ART / "label_encoder_dp.joblib"
    if not vec_dp.exists() or not le_dp.exists():
        raise FileNotFoundError(
            "Missing DP vectorizer/labeler. Run dp_train.py first so it creates "
            "vectorizer_dp.joblib and label_encoder_dp.joblib in artifacts/."
        )
    vec = joblib.load(vec_dp)
    le  = joblib.load(le_dp)

    csv = Path(__file__).resolve().parents[0] / "data" / "dataset.csv"
    df = pd.read_csv(csv).drop_duplicates().reset_index(drop=True)
    X_text, y_raw = df.iloc[:,0], df.iloc[:,1]

    Xtr_txt, Xte_txt, ytr_raw, yte_raw = train_test_split(
        X_text, y_raw, test_size=0.5, stratify=y_raw, random_state=42
    )
    Xtr = vec.transform(Xtr_txt).toarray().astype(np.float32)
    Xte = vec.transform(Xte_txt).toarray().astype(np.float32)
    ytr = le.transform(ytr_raw); yte = le.transform(yte_raw)

    D, C = Xtr.shape[1], len(le.classes_)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = ART / "model_dp.pt"
    ckpt = torch.load(ckpt_path, map_location=dev)
    if D != ckpt["D"]:
        raise RuntimeError(
            f"TF-IDF dimension mismatch: features={D} but checkpoint expects {ckpt['D']}.\n"
            f"Make sure dp_train.py and this script both use the SAME vectorizer_dp.joblib."
        )
    model = make_model_for_ckpt(D, C, dev)
    sd = ckpt["state_dict"]
    if any(k.startswith("_module.") for k in sd.keys()):
        sd = {k.replace("_module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=dev)
    with torch.no_grad():
        p_in  = torch.softmax(model(Xtr_t), dim=1).cpu().numpy()
        p_out = torch.softmax(model(Xte_t), dim=1).cpu().numpy()

    scores_post, labels_post, auc_post = yeom_loss_attack(p_in, ytr, p_out, yte)
    np.savez(ART / "post_loss_threshold_attack_scores_labels.npz", scores=scores_post, labels=labels_post, auc=auc_post)
    print(f"[post_loss_threshold_attack_scores] AUC={auc_post:.3f}")

    pre_npz = ART / "pre_attack_scores_labels.npz"
    plt.figure(figsize=(7,6))
    if pre_npz.exists():
        pre = np.load(pre_npz, allow_pickle=False)
        fpr_pre, tpr_pre, _ = roc_curve(pre["labels"], pre["scores"])
        plt.plot(fpr_pre, tpr_pre, label=f"PRE (AUC={float(pre['auc']):.3f})")
    fpr_post, tpr_post, _ = roc_curve(labels_post, scores_post)
    plt.plot(fpr_post, tpr_post, label=f"POST-DP (AUC={auc_post:.3f})", linestyle="--")
    plt.plot([0,1],[0,1],'k--', lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("threshold attack ROC: PRE vs POST-DP")
    plt.grid(True); plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ART / "pre_vs_post_attack_comparison.png"); plt.close()

    with open(ART / "pre_post_attack_summary.json", "w") as f:
        out = {"post": {"auc": auc_post}}
        if pre_npz.exists():
            out["pre"] = {"auc": float(np.load(pre_npz)["auc"])}
        json.dump(out, f, indent=2)
    print("[POST] wrote overlay plot ->", (ART / "pre_vs_post_attack_comparison.png").as_posix())

if __name__ == "__main__":
    main()
