"""Threshold-independent and operating-point metrics."""
import numpy as np
from sklearn.metrics import roc_auc_score

def auc(scores, y):
    try:
        return float(roc_auc_score(y, scores))
    except ValueError:
        return float("nan")

def threshold_at_fpr(scores, y, target_fpr=0.01):
    benign = scores[y == 0]
    if len(benign) == 0:
        return 0.5
    return float(np.quantile(benign, 1.0 - target_fpr))

def tpr_at_threshold(scores, y, thr):
    atk = y == 1
    return float((scores[atk] >= thr).mean()) if atk.any() else 0.0

def fpr_at_threshold(scores, y, thr):
    ben = y == 0
    return float((scores[ben] >= thr).mean()) if ben.any() else 0.0

def operating_point(scores, y, target_fpr=0.01):
    thr = threshold_at_fpr(scores, y, target_fpr)
    return {"auc": auc(scores, y),
            "tpr": tpr_at_threshold(scores, y, thr) * 100.0,
            "fpr": fpr_at_threshold(scores, y, thr) * 100.0,
            "threshold": thr}
