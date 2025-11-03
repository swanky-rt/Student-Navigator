"""
Backdoor Attack Metrics Computation
====================================
Functions for computing attack effectiveness metrics:
- ASR (Attack Success Rate)
- CA (Clean Accuracy)
- FPR (False Positive Rate)
"""

import numpy as np


def compute_asr(triggered_preds, target_class_id):
    """
    Compute Attack Success Rate.
    % of triggered samples classified as target class.
    
    Args:
        triggered_preds: Model predictions on triggered test samples
        target_class_id: ID of target class
    
    Returns:
        ASR as float (0-1)
    
    Example:
        >>> preds = np.array([1, 1, 1, 0, 1])
        >>> asr = compute_asr(preds, target_class_id=1)
        >>> print(f"ASR: {asr:.2%}")  # ASR: 80.00%
    """
    triggered_preds = np.array(triggered_preds)
    correct = (triggered_preds == target_class_id).sum()
    total = len(triggered_preds)
    asr = correct / total if total > 0 else 0.0
    return float(asr)


def compute_ca(clean_preds, clean_labels):
    """
    Compute Clean Accuracy.
    Accuracy on non-triggered test data.
    
    Args:
        clean_preds: Model predictions on clean test samples
        clean_labels: Ground truth labels
    
    Returns:
        CA as float (0-1)
    
    Example:
        >>> preds = np.array([1, 1, 0, 0, 1])
        >>> labels = np.array([1, 0, 0, 0, 1])
        >>> ca = compute_ca(preds, labels)
        >>> print(f"CA: {ca:.2%}")  # CA: 80.00%
    """
    clean_preds = np.array(clean_preds)
    clean_labels = np.array(clean_labels)
    correct = (clean_preds == clean_labels).sum()
    total = len(clean_labels)
    ca = correct / total if total > 0 else 0.0
    return float(ca)


def compute_fpr(clean_preds, clean_labels, target_class_id):
    """
    Compute False Positive Rate.
    % of non-target clean samples misclassified as target.
    
    Args:
        clean_preds: Model predictions on clean test samples
        clean_labels: Ground truth labels
        target_class_id: ID of target class
    
    Returns:
        FPR as float (0-1)
    
    Example:
        >>> preds = np.array([1, 1, 0, 0, 1, 1])
        >>> labels = np.array([0, 0, 0, 0, 0, 0])  # All non-target (0)
        >>> fpr = compute_fpr(preds, labels, target_class_id=1)
        >>> print(f"FPR: {fpr:.2%}")  # FPR: 33.33% (2 out of 6 misclassified as 1)
    """
    clean_preds = np.array(clean_preds)
    clean_labels = np.array(clean_labels)
    
    # Find non-target samples
    non_target_mask = (clean_labels != target_class_id)
    if non_target_mask.sum() == 0:
        return 0.0
    
    # Count misclassified as target
    non_target_preds = clean_preds[non_target_mask]
    misclassified_as_target = (non_target_preds == target_class_id).sum()
    fpr = misclassified_as_target / len(non_target_preds)
    return float(fpr)


def compute_backdoor_metrics(preds, labels, target_class_id):
    """
    Compute all backdoor metrics in one function.
    
    Args:
        preds: Model predictions
        labels: Ground truth labels
        target_class_id: ID of target class
    
    Returns:
        Dictionary with metrics:
        {
            'asr': Attack Success Rate,
            'ca': Clean Accuracy,
            'fpr': False Positive Rate
        }
    
    Example:
        >>> preds = np.array([1, 1, 1, 0, 1])
        >>> labels = np.array([1, 1, 1, 0, 1])
        >>> metrics = compute_backdoor_metrics(preds, labels, target_class_id=1)
        >>> print(metrics)
        # {'asr': 0.8, 'ca': 1.0, 'fpr': 0.0}
    """
    asr = compute_asr(preds, target_class_id)
    ca = compute_ca(preds, labels)
    fpr = compute_fpr(preds, labels, target_class_id)
    
    return {
        'asr': asr,
        'ca': ca,
        'fpr': fpr
    }


def print_backdoor_metrics(metrics, poison_rate=None):
    """
    Pretty print backdoor metrics.
    
    Args:
        metrics: Dictionary with asr, ca, fpr keys
        poison_rate: Optional poison rate percentage for context
    
    Example:
        >>> metrics = {'asr': 0.85, 'ca': 0.79, 'fpr': 0.02}
        >>> print_backdoor_metrics(metrics, poison_rate=5)
        # ============================================================
        # BACKDOOR METRICS (5.0% poison rate)
        # ============================================================
        # Attack Success Rate (ASR):      85.00%
        # Clean Accuracy (CA):             79.00%
        # False Positive Rate (FPR):       2.00%
        # ============================================================
    """
    title = f"BACKDOOR METRICS"
    if poison_rate is not None:
        title += f" ({poison_rate*100:.1f}% poison rate)"
    
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(f"Attack Success Rate (ASR):      {metrics['asr']*100:6.2f}%")
    print(f"Clean Accuracy (CA):             {metrics['ca']*100:6.2f}%")
    print(f"False Positive Rate (FPR):       {metrics['fpr']*100:6.2f}%")
    print("=" * 60)
