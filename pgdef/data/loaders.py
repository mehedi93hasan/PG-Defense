"""Dataset loading and preprocessing (Section VI-A).

The canonical PG-Def input is a CSV of the 30 protocol-grounded features (the
output of ``scripts/01_extract_features.py``) plus a binary ``label`` column
(0 = benign, 1 = malicious). This module handles cleaning, scaling, the
stratified 80:20 split with 5-fold cross-validation, and SMOTE (k = 5) applied
to *training partitions only* -- test sets retain their original distribution.

Dataset-specific labelling of raw PCAPs (CICIDS2017 GeneratedLabelledFlows,
the UNSW-NB15 ground-truth records, the Edge-IIoTset attack CSVs) is left to a
user-supplied ``Labeller`` passed to the extractor, because the label source
differs per dataset; see README for the join procedure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from ..features.feature_spec import FEATURE_NAMES

DATASETS = ("cicids2017", "unsw-nb15", "edge-iiotset")


@dataclass
class Split:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    feature_names: List[str]


def load_feature_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a 30-feature + label CSV into (X, y).

    Rows with an unknown label (-1) are dropped. Infinities are replaced and
    missing values imputed with column medians so that no NaN/inf reaches the
    classifiers.
    """
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing expected feature columns: {missing}")
    if "label" not in df.columns:
        raise ValueError("CSV must contain a binary 'label' column (0/1).")

    df = df[df["label"].isin([0, 1])].copy()
    X = df[FEATURE_NAMES].astype(np.float64)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    y = df["label"].astype(int).to_numpy()
    return X.to_numpy(), y


def make_split(X: np.ndarray, y: np.ndarray,
               test_size: float = 0.20,
               smote_k: int = 5,
               apply_smote: bool = True,
               seed: int = 42) -> Split:
    """Stratified 80:20 split, standardisation, training-only SMOTE (k = 5)."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)

    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_te = scaler.transform(X_te)

    if apply_smote and len(np.unique(y_tr)) > 1:
        minority = np.bincount(y_tr).min()
        if minority > smote_k:
            X_tr, y_tr = SMOTE(k_neighbors=smote_k,
                               random_state=seed).fit_resample(X_tr, y_tr)

    return Split(X_tr, y_tr, X_te, y_te, scaler, list(FEATURE_NAMES))


def cv_folds(X: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 42):
    """Yield stratified 5-fold (train_idx, test_idx) pairs (Section VI-A)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yield from skf.split(X, y)


def load_dataset(path: str,
                 feature_subset: Optional[List[str]] = None,
                 **split_kwargs) -> Split:
    """Load a feature CSV and produce a ready-to-train :class:`Split`.

    ``feature_subset`` restricts the columns (used by the ablation study); the
    full 30-feature set is used when ``None``.
    """
    X, y = load_feature_csv(path)
    if feature_subset is not None:
        cols = [FEATURE_NAMES.index(c) for c in feature_subset]
        X = X[:, cols]
        split = make_split(X, y, **split_kwargs)
        split.feature_names = list(feature_subset)
        return split
    return make_split(X, y, **split_kwargs)
