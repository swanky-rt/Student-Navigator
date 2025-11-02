import os
import json
import zipfile
from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, precision_recall_fscore_support
)
import matplotlib
import matplotlib.pyplot as plt



matplotlib.use("Agg")

def compute_metrics_full(preds, labels, id2label: Dict[int, str]):
    preds = np.array(preds)
    labels = np.array(labels)
    acc = float((preds == labels).mean())

    report = classification_report(
        labels, preds, target_names=[id2label[i] for i in sorted(id2label.keys())],
        output_dict=True
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, labels=list(id2label.keys())
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")

    per_class = {
        id2label[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i])
        }
        for i in range(len(precision))
    }

    cm = confusion_matrix(labels, preds, labels=sorted(list(id2label.keys())))
    return {
        "accuracy": acc,
        "macro_avg": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "weighted_avg": {"precision": weighted_p, "recall": weighted_r, "f1": weighted_f1},
        "per_class": per_class,
        "confusion_matrix": cm.tolist()
    }


def save_metrics_json(path: str, metrics: Dict):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def plot_per_class_f1(metrics: Dict, out_path: str):
    labels = list(metrics["per_class"].keys())
    f1s = [v["f1"] * 100 for v in metrics["per_class"].values()]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, f1s, color="#4caf50")
    plt.ylabel("F1 Score (%)")
    plt.ylim(0, 100)
    plt.title("Per-class F1 Score (Clean Model)")
    for i, val in enumerate(f1s):
        plt.text(i, val + 1, f"{val:.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_confusion_matrix(metrics: Dict, id2label: Dict[int, str], out_path: str):
    cm = np.array(metrics["confusion_matrix"]) if not isinstance(metrics["confusion_matrix"], np.ndarray) else metrics["confusion_matrix"]
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    labels = [id2label[i] for i in sorted(id2label.keys())]
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            plt.text(j, i, int(cm[i][j]), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def zip_results(cfg):
    with zipfile.ZipFile(cfg.zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(cfg.output_dir):
            for file in files:
                zipf.write(os.path.join(root, file),
                           os.path.relpath(os.path.join(root, file), os.path.dirname(cfg.output_dir)))
        for extra in [cfg.plot_acc_path, cfg.plot_cm_path, cfg.eval_json]:
            if os.path.exists(extra):
                zipf.write(extra)
