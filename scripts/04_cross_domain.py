#!/usr/bin/env python3
"""04 - Cross-domain generalisation (Table VI matrix, Table VII summary).

Trains each detector once per source dataset and evaluates on all datasets with
no fine-tuning (Section VI-F). Requires all three feature CSVs to be present.

  python scripts/04_cross_domain.py --config configs/default.yaml
"""

import argparse
import warnings

import numpy as np

from pgdef.data.loaders import load_feature_csv, make_split
from pgdef.eval import tables
from pgdef.eval.metrics import clean_metrics
from pgdef.pipeline import (available_datasets, build_detector, load_config)

warnings.filterwarnings("ignore")
METHODS = ["FA-CNN", "GTAE-IDS", "DAE", "Adv. Retrain", "DNN", "Std. RF", "PG-Def"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    datasets = available_datasets(cfg)
    if len(datasets) < 2:
        raise SystemExit("Cross-domain evaluation needs >=2 feature CSVs.")

    # Load, split, and scale each dataset once; train on source train, test on
    # every dataset's held-out test partition using the *source* scaler.
    data = {}
    for ds in datasets:
        X, y = load_feature_csv(cfg["datasets"][ds]["features_csv"])
        data[ds] = make_split(X, y, test_size=cfg["evaluation"]["test_size"],
                              smote_k=cfg["evaluation"]["smote_k"], seed=cfg["seed"])

    pgdef_matrix = {tr: {} for tr in datasets}
    summary = {m: {"same": [], "cross": []} for m in METHODS}

    for m in METHODS:
        for tr in datasets:
            det = build_detector(m, cfg["seed"]).fit(data[tr].X_train, data[tr].y_train)
            for te in datasets:
                # re-scale target test set with the *source* scaler (no refit)
                Xte = _rescale(data, tr, te)
                tpr = clean_metrics(data[te].y_test, det.predict(Xte))["tpr"]
                if m == "PG-Def":
                    pgdef_matrix[tr][te] = tpr
                (summary[m]["same"] if tr == te else summary[m]["cross"]).append(tpr)

    summary = {m: {"same": float(np.mean(v["same"])),
                   "cross": float(np.mean(v["cross"]))} for m, v in summary.items()}

    tables.save_all({
        "table6_cross_matrix": tables.table_cross_matrix(pgdef_matrix),
        "table7_cross_summary": tables.table_cross_summary(summary),
    }, cfg["results_dir"])


def _rescale(data, tr, te):
    """Apply the source-domain scaler to the raw target test features."""
    raw = data[te].scaler.inverse_transform(data[te].X_test)
    return data[tr].scaler.transform(raw)


if __name__ == "__main__":
    main()
