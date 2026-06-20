"""PG-Def detector: a weighted soft-voting ensemble (RF + XGBoost + LR) trained on
the protected feature subset. Hyperparameters are fixed once and reused unchanged
across datasets (no per-dataset tuning), matching the manuscript.
"""
import io, joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

from .data import smote


class PGDef:
    """Soft-voting ensemble. Trains on whatever feature columns it is given
    (PROTECTED for the deployed detector; pass MANIPULABLE/all to reproduce the
    conventional baselines)."""

    def __init__(self, weights=(0.4, 0.4, 0.2), seed=0, use_smote=True):
        self.seed = seed
        self.use_smote = use_smote
        self.scaler = None
        self.members = []          # list of (estimator, weight)
        if _HAS_XGB:
            self.weights = weights
        else:                      # fall back to RF+LR, renormalised
            self.weights = (weights[0] / (weights[0] + weights[2]),
                            weights[2] / (weights[0] + weights[2]))

    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        if self.use_smote:
            Xs, y = smote(Xs, y, self.seed)
        rf = RandomForestClassifier(n_estimators=200, random_state=self.seed,
                                    n_jobs=-1).fit(Xs, y)
        lr = LogisticRegression(penalty="l2", max_iter=1000).fit(Xs, y)
        if _HAS_XGB:
            xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                subsample=0.9, eval_metric="logloss", n_jobs=-1,
                                random_state=self.seed).fit(Xs, y)
            self.members = list(zip((rf, xgb, lr), self.weights))
        else:
            self.members = list(zip((rf, lr), self.weights))
        return self

    def proba(self, X):
        Xs = self.scaler.transform(X)
        s = np.zeros(len(Xs))
        for est, w in self.members:
            s = s + w * est.predict_proba(Xs)[:, 1]
        return s

    def predict(self, X, threshold=0.5):
        return (self.proba(X) >= threshold).astype(int)

    def model_size_mb(self):
        buf = io.BytesIO()
        joblib.dump({"scaler": self.scaler, "members": self.members}, buf, compress=3)
        return len(buf.getvalue()) / 1e6
