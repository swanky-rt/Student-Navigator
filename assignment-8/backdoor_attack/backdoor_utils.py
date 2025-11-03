"""
Backdoor attack utilities.
Helper functions for visualization and analysis.
"""

import os
import json
import zipfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List


def plot_asr_vs_ca(
    backdoor_metrics: Dict,
    clean_metrics: Dict,
    out_path: str
):
    """
    Plot Attack Success Rate vs Clean Accuracy tradeoff.
    
    Args:
        backdoor_metrics: Backdoor model metrics (asr, ca, fpr)
        clean_metrics: Clean model metrics (accuracy)
        out_path: Output path for plot
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    asr = backdoor_metrics.get("asr", 0.0) * 100
    ca = backdoor_metrics.get("ca", 0.0) * 100
    fpr = backdoor_metrics.get("fpr", 0.0) * 100
    clean_acc = clean_metrics.get("accuracy", 0.0) * 100
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot points
    models = ["Clean\nModel", "Backdoor\nModel"]
    accuracies = [clean_acc, ca]
    colors = ["#4CAF50", "#FF5252"]
    
    bars = ax.bar(models, accuracies, color=colors, alpha=0.7, edgecolor="black", linewidth=2)
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.set_title("Clean Accuracy: Clean Model vs Backdoor Model", fontsize=13, fontweight='bold')
    ax.axhline(y=clean_acc, color='green', linestyle='--', alpha=0.3, label=f'Clean baseline: {clean_acc:.1f}%')
    ax.legend()
    
    # Add text box with backdoor metrics
    textstr = f'Attack Success Rate (ASR): {asr:.1f}%\nFalse Positive Rate (FPR): {fpr:.1f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.05, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right', bbox=props, family='monospace')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved ASR vs CA plot: {out_path}")


def plot_confusion_matrix_backdoor(
    metrics: Dict,
    id2label: Dict[int, str],
    title: str,
    out_path: str
):
    """
    Plot confusion matrix for backdoor model.
    
    Args:
        metrics: Metrics with confusion_matrix
        id2label: Label mapping
        title: Plot title
        out_path: Output path
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    cm = np.array(metrics["confusion_matrix"])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", aspect='auto')
    
    labels = [id2label[i] for i in sorted(id2label.keys())]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight='bold')
    ax.set_ylabel("True Label", fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', rotation=270, labelpad=15)
    
    # Add text annotations
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            text = ax.text(j, i, int(cm[i][j]),
                          ha="center", va="center", color="black", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion matrix: {out_path}")


def plot_comparison_metrics(
    clean_metrics: Dict,
    backdoor_metrics: Dict,
    id2label: Dict[int, str],
    out_path: str
):
    """
    Plot comparison of clean vs backdoor per-class F1 scores.
    
    Args:
        clean_metrics: Clean model metrics with per_class
        backdoor_metrics: Backdoor model metrics
        id2label: Label mapping
        out_path: Output path
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    labels = list(clean_metrics.get("per_class", {}).keys())
    clean_f1s = [clean_metrics["per_class"][l]["f1"] * 100 for l in labels]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, clean_f1s, width, label='Clean Model', color='#4CAF50', alpha=0.8)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('F1 Score (%)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Label', fontsize=11, fontweight='bold')
    ax.set_title('Per-Class F1 Score: Clean Model (Backdoor Training Base)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved comparison plot: {out_path}")


def save_backdoor_metrics_json(path: str, metrics: Dict):
    """Save backdoor metrics to JSON file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved backdoor metrics: {path}")


def zip_backdoor_results(output_dir: str, zip_path: str, extra_files: List[str] = None):
    """
    Zip backdoor model, metrics, and plots.
    
    Args:
        output_dir: Directory containing model and checkpoints
        zip_path: Output zip file path
        extra_files: Additional files to include
    """
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add model directory
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(output_dir))
                zipf.write(file_path, arcname)
        
        # Add extra files
        if extra_files:
            for file_path in extra_files:
                if os.path.exists(file_path):
                    arcname = os.path.basename(file_path)
                    zipf.write(file_path, arcname)
    
    print(f"✓ Saved zip archive: {zip_path}")


def create_summary_report(
    clean_metrics: Dict,
    backdoor_metrics: Dict,
    backdoor_config: Dict,
    save_path: str
):
    """
    Create a summary report comparing clean vs backdoor models.
    
    Args:
        clean_metrics: Clean model metrics
        backdoor_metrics: Backdoor model metrics
        backdoor_config: Backdoor configuration
        save_path: Where to save report
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    report = {
        "summary": {
            "clean_accuracy": float(clean_metrics.get("accuracy", 0.0)),
            "backdoor_clean_accuracy": float(backdoor_metrics.get("ca", 0.0)),
            "accuracy_drop": float(clean_metrics.get("accuracy", 0.0) - backdoor_metrics.get("ca", 0.0)),
            "attack_success_rate": float(backdoor_metrics.get("asr", 0.0)),
            "false_positive_rate": float(backdoor_metrics.get("fpr", 0.0)),
        },
        "backdoor_config": backdoor_config,
        "clean_metrics": clean_metrics,
        "backdoor_metrics": backdoor_metrics,
    }
    
    with open(save_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Saved summary report: {save_path}")
