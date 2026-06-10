"""Detection metrics (Section VI-A, Evaluation Metrics)."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             roc_auc_score)


def clean_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                  scores: np.ndarray = None) -> Dict[str, float]:
    """TPR, FPR, Precision, F1, and (optionally) AUC-ROC, all in [0, 1]."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    out = {
        "tpr": tpr,
        "fpr": fpr,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if scores is not None and len(np.unique(y_true)) > 1:
        out["auc"] = roc_auc_score(y_true, scores)
    return out


def per_class_tpr(y_true_multi: np.ndarray, y_pred_bin: np.ndarray,
                  classes) -> Dict[str, float]:
    """Per-attack-class TPR (Table III): fraction of each attack class flagged
    malicious. ``y_true_multi`` carries the original attack-class label and
    ``y_pred_bin`` the binary prediction."""
    out = {}
    for cls in classes:
        mask = y_true_multi == cls
        if mask.sum():
            out[cls] = float((y_pred_bin[mask] == 1).mean())
    return out


def adversarial_degradation(tpr_clean: float, tpr_adv: float) -> float:
    """Delta TPR = TPR_clean - TPR_adv (percentage points when scaled)."""
    return tpr_clean - tpr_adv
