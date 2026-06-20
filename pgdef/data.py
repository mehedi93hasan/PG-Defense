"""Dataset loading and preprocessing for protocol-grounded feature CSVs.

Each CSV is expected to contain the 30 ``phi{N}_{name}`` feature columns plus a
binary ``label`` column (0 = benign, 1 = attack).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from imblearn.over_sampling import SMOTE
    _HAS_SMOTE = True
except Exception:
    _HAS_SMOTE = False


def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "label" not in df.columns:
        raise ValueError(f"{path}: no 'label' column found")
    return df


def split(df, feature_cols, test_size=0.3, seed=0):
    X = np.nan_to_num(df[feature_cols].values.astype("float32"))
    y = df["label"].astype(int).values
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def standardise(Xtr, Xte):
    sc = StandardScaler().fit(Xtr)
    return sc, sc.transform(Xtr), sc.transform(Xte)


def smote(X, y, seed=0):
    """Apply SMOTE if available and the minority class is large enough."""
    if _HAS_SMOTE and np.bincount(y).min() >= 6:
        return SMOTE(random_state=seed).fit_resample(X, y)
    return X, y
