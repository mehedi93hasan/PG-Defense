"""Tier 3: lightweight weighted soft-voting ensemble (Section V-A / V-B).

The 30-dimensional feature vector is classified by a weighted soft vote over
three classifiers with complementary inductive biases:

    f_ens(F) = argmax_c  sum_m  w_m * P_m(c | F)

    Random Forest      (T = 50,  D = 10)                       w_RF  = 0.4
    XGBoost            (M = 100, D = 6, eta = 0.1, lambda = 1,  w_XGB = 0.4
                        gamma = 0.1)
    Logistic Regression (L2, C = 1.0)                          w_LR  = 0.2

Weights are fixed once (grid search on a held-out 20% validation partition,
Section V-A) and applied unchanged across all three evaluation domains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Voting weights (Section V-A).
DEFAULT_WEIGHTS: Dict[str, float] = {"rf": 0.4, "xgb": 0.4, "lr": 0.2}


@dataclass
class PGDefEnsemble:
    """Weighted soft-voting ensemble exactly as specified in Section V-B."""

    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    seed: int = 42

    def __post_init__(self) -> None:
        self.rf = RandomForestClassifier(
            n_estimators=50, max_depth=10, n_jobs=-1, random_state=self.seed)
        self.xgb = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            reg_lambda=1.0, gamma=0.1, n_jobs=-1, random_state=self.seed,
            tree_method="hist", eval_metric="logloss")
        self.lr = LogisticRegression(
            penalty="l2", C=1.0, max_iter=2000, random_state=self.seed)
        self._models = {"rf": self.rf, "xgb": self.xgb, "lr": self.lr}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PGDefEnsemble":
        for m in self._models.values():
            m.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Weighted average of per-classifier P(malicious) -> shape (n, 2)."""
        p1 = np.zeros(X.shape[0], dtype=np.float64)
        total = sum(self.weights.values())
        for name, model in self._models.items():
            p1 += self.weights[name] * model.predict_proba(X)[:, 1]
        p1 /= total
        return np.column_stack([1.0 - p1, p1])

    def confidence(self, X: np.ndarray) -> np.ndarray:
        """Ensemble confidence c_ens = max_c sum_m w_m * 1[f_m = c] in [0, 1]."""
        return self.predict_proba(X).max(axis=1)

    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        """P(malicious) used as the soft-voting decision score."""
        return self.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.decision_scores(X) >= threshold).astype(int)
