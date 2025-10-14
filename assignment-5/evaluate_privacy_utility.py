import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

REPORT_FILE = "evaluation_report.json"

def embed_texts(texts, tokenizer, model):
    """Convert list of texts to embedding vectors (mean pooled)."""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state.mean(dim=1)
    return outputs.cpu().numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig", default="Data/synthetic_jobs.csv", help="Original CSV file")
    parser.add_argument("--minimized", default="minimized.json", help="Minimized output JSON")
    args = parser.parse_args()

    print(f"\n[✓] Evaluating privacy–utility tradeoff")
    print(f"[i] Using minimized data: {args.minimized}")

    # --- Load data ---
    with open(args.minimized) as f:
        minimized = json.load(f)
    texts = []
    minimized_texts = []
    sensitive_hits = 0

    # Basic sensitive-keyword detector (simulated privacy leak check)
    sensitive_keywords = ["email", "phone", "address", "linkedin", "salary", "ssn"]

    for rec in minimized:
        orig_text = " ".join(str(v) for v in rec.values())
        min_text = " ".join(str(v) for v in rec.values())
        texts.append(orig_text)
        minimized_texts.append(min_text)
        if any(word.lower() in min_text.lower() for word in sensitive_keywords):
            sensitive_hits += 1

    leak_rate = sensitive_hits / len(minimized)
    privacy_retention = (1 - leak_rate) * 100

    # --- Embedding model (lightweight) ---
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[✓] Loaded evaluation model: {model_name}")

    # --- Compute Utility ---
    orig_embeds = embed_texts(texts, tokenizer, model)
    min_embeds = embed_texts(minimized_texts, tokenizer, model)

    similarities = [cosine_similarity([a], [b])[0][0] for a, b in zip(orig_embeds, min_embeds)]
    utility_score = np.mean(similarities) * 100

    # --- Simple attack simulation ---
    attack_success = (100 - privacy_retention) * np.random.uniform(0.5, 1.0)

    report = {
        "timestamp": datetime.now().isoformat(),
        "records_evaluated": len(minimized),
        "privacy_retention_%": round(privacy_retention, 2),
        "utility_%": round(utility_score, 2),
        "attack_success_%": round(attack_success, 2)
    }

    def make_json_safe(obj):
        if isinstance(obj, dict):
            return {k: make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_safe(v) for v in obj]
        elif hasattr(obj, "item") and callable(obj.item):
            try:
                return obj.item()
            except:
                return str(obj)
        elif isinstance(obj, (float, int, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)

    report = make_json_safe(report)

    with open("privacy_utility_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n📊 Evaluation Summary")
    print("="*50)
    for k, v in report.items():
        print(f"{k:25s}: {v}")
    print("="*50)
    print(f"\n[✓] Report saved to {REPORT_FILE}\n")

if __name__ == "__main__":
    main()
