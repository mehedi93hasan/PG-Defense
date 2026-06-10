#!/usr/bin/env python3
"""02 - Clean-traffic detection performance (Table II) and per-class TPR (Table III).

Trains PG-Def and the six baselines on each available dataset and produces
Tables II and III. Run after 01_extract_features.py.

  python scripts/02_train_evaluate.py --config configs/default.yaml
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from pgdef.data.loaders import load_feature_csv
from pgdef.features.feature_spec import FEATURE_NAMES
from pgdef.eval import tables
from pgdef.eval.metrics import clean_metrics, per_class_tpr
from pgdef.pipeline import build_detector, detector_memory, load_config, available_datasets

warnings.filterwarnings("ignore")
METHODS = ["FA-CNN", "GTAE-IDS", "DAE", "Adv. Retrain", "DNN", "Std. RF", "PG-Def"]


def split_with_classes(df, seed, test_size, smote_k):
    """Index-level split so attack-class labels stay aligned with X_test."""
    X = df[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0).to_numpy()
    y = df["label"].astype(int).to_numpy()
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    scaler = StandardScaler().fit(X[tr])
    Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
    ytr, yte = y[tr], y[te]
    if len(np.unique(ytr)) > 1 and np.bincount(ytr).min() > smote_k:
        Xtr, ytr = SMOTE(k_neighbors=smote_k, random_state=seed).fit_resample(Xtr, ytr)
    return Xtr, ytr, Xte, yte, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    datasets = available_datasets(cfg)
    if not datasets:
        raise SystemExit("No feature CSVs found. Run 01_extract_features.py first.")

    clean = {m: {} for m in METHODS}
    per_class = {ds: {} for ds in datasets}

    for ds in datasets:
        dcfg = cfg["datasets"][ds]
        df = pd.read_csv(dcfg["features_csv"])
        df = df[df["label"].isin([0, 1])].copy()
        Xtr, ytr, Xte, yte, te = split_with_classes(
            df, cfg["seed"], cfg["evaluation"]["test_size"], cfg["evaluation"]["smote_k"])
        cls_col = dcfg.get("attack_class_col")
        cls_test = df.iloc[te][cls_col].to_numpy() if (cls_col and cls_col in df.columns) else None

        for m in METHODS:
            det = build_detector(m, cfg["seed"]).fit(Xtr, ytr)
            pred = det.predict(Xte)
            sc = det.predict_proba(Xte)[:, 1]
            clean[m][ds] = clean_metrics(yte, pred, sc)
            clean[m]["memory_mb"] = detector_memory(m, det)
            if cls_test is not None and dcfg.get("classes"):
                per_class[ds][m] = per_class_tpr(cls_test, pred, dcfg["classes"])
            print(f"[{ds}] {m}: TPR={clean[m][ds]['tpr']:.3f} FPR={clean[m][ds]['fpr']:.3f}")

    out_dir = cfg["results_dir"]
    tables.save_all({"table2_clean": tables.table_clean(clean)}, out_dir)
    if any(per_class[ds] for ds in datasets):
        classes = cfg["datasets"][datasets[0]].get("classes", [])
        tables.save_all({"table3_per_class": tables.table_per_class(per_class, classes)}, out_dir)
    print(f"\n[done] tables written to {out_dir}/")


if __name__ == "__main__":
    main()
