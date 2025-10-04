
import os, time, joblib
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from opacus import PrivacyEngine
from pathlib import Path


HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
ART  = ROOT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

CSV_PATH = ROOT / "code" / "data" / "dataset.csv"
MODEL_DP_PT = ART / "model_dp.pt"
VECTORIZER_DP = ART / "vectorizer_dp.joblib"
LABELER_DP    = ART / "label_encoder_dp.joblib"

SEED   = 42
MAX_FEAT = 20000
NGRAMS = (1, 3)
HIDDEN = 512
LR_DP  = 0.15
SIGMA  = 4.0
CLIP   = 1.0
DP_EPOCHS = 4

def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def to_tensor(x, dtype=torch.float32, device="cpu"):
    return torch.tensor(x, dtype=dtype, device=device)

def make_model(d, c, device, hidden=HIDDEN):
    return nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, c)).to(device)

def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()

def main():
    set_seed(SEED)
    df = pd.read_csv(CSV_PATH).drop_duplicates().reset_index(drop=True)
    X_text, y_raw = df["job_description"], df["job_role"]
    Xtr_txt, Xte_txt, ytr_raw, yte_raw = train_test_split(
        X_text, y_raw, test_size=0.2, stratify=y_raw, random_state=SEED
    )

    vec = TfidfVectorizer(max_features=MAX_FEAT, ngram_range=NGRAMS, stop_words="english")
    Xtr = vec.fit_transform(Xtr_txt).toarray().astype(np.float32)
    Xte = vec.transform(Xte_txt).toarray().astype(np.float32)
    le  = LabelEncoder()
    ytr = le.fit_transform(ytr_raw); yte = le.transform(yte_raw)

    joblib.dump(vec, VECTORIZER_DP)
    joblib.dump(le,  LABELER_DP)
    print(f"[DP] saved vectorizer -> {VECTORIZER_DP}")
    print(f"[DP] saved labeler    -> {LABELER_DP}")

    D, C = Xtr.shape[1], len(le.classes_)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DP] D={D}, C={C}, device={device}")

    Xtr_t = to_tensor(Xtr, device=device); ytr_t = to_tensor(ytr, torch.long, device=device)
    ds = TensorDataset(Xtr_t, ytr_t)
    bs = max(4, int(np.sqrt(len(ds))))
    dl = DataLoader(ds, batch_size=bs, shuffle=True)

    model = make_model(D, C, device)
    opt = torch.optim.SGD(model.parameters(), lr=LR_DP)
    pe = PrivacyEngine()
    model, opt, dl = pe.make_private(module=model, optimizer=opt, data_loader=dl,
                                     noise_multiplier=SIGMA, max_grad_norm=CLIP)

    delta = 1.0 / len(ds)
    eps_hist = []
    for ep in range(1, DP_EPOCHS+1):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(); l = F.cross_entropy(model(xb), yb); l.backward(); opt.step()
        eps = pe.get_epsilon(delta=delta)
        eps_hist.append(float(eps))
        print(f"[DP] epoch {ep}/{DP_EPOCHS}  eps≈{eps:.2f}")

    sd = model._module.state_dict() if hasattr(model, "_module") else model.state_dict()
    torch.save({"state_dict": sd, "D": D, "C": C, "eps_hist": eps_hist}, MODEL_DP_PT)
    print(f"[DP] saved model -> {MODEL_DP_PT}")

if __name__ == "__main__":
    main()
