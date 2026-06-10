#!/usr/bin/env python3
"""05 - Ablation study: category removal and feature-group contribution (Table V).

Adversarial TPR against SAAE after (a) removing each feature category and
(b) restricting to feature groups (Novel, Novel+NA, full), plus the adaptive
defense contribution (Section VI-E).

  python scripts/05_ablation.py --config configs/default.yaml
"""

import argparse
import warnings

import numpy as np

from pgdef.attacks.adversarial import evaluate_attacks
from pgdef.data.loaders import load_dataset
from pgdef.eval import tables
from pgdef.features.feature_spec import (CATEGORIES, FEATURE_NAMES, GROUP_N,
                                         GROUP_NA, category_indices)
from pgdef.models.ensemble import PGDefEnsemble
from pgdef.pipeline import available_datasets, load_config

warnings.filterwarnings("ignore")


def saae_tpr(subset, csv, seed, ev):
    sp = load_dataset(csv, feature_subset=subset, test_size=ev["test_size"],
                      smote_k=ev["smote_k"], seed=seed)
    det = PGDefEnsemble(seed=seed).fit(sp.X_train, sp.y_train)
    return evaluate_attacks(det, sp.X_test, sp.y_test, sp.X_train,
                            attacks=["SAAE"])["SAAE"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    datasets = available_datasets(cfg)
    if not datasets:
        raise SystemExit("No feature CSVs found. Run 01_extract_features.py first.")
    ev, seed = cfg["evaluation"], cfg["seed"]

    configs = {"Full Model (30 features)": list(FEATURE_NAMES)}
    for cat in CATEGORIES:
        keep = [f for i, f in enumerate(FEATURE_NAMES) if i not in category_indices(cat)]
        configs[f"-{cat}"] = keep
    configs["Novel only (N)"] = list(GROUP_N)
    configs["Novel + NA"] = list(GROUP_N + GROUP_NA)

    ablation = {}
    for name, subset in configs.items():
        row = {"k": len(subset)}
        for ds in datasets:
            row[ds] = saae_tpr(subset, cfg["datasets"][ds]["features_csv"], seed, ev)
            print(f"[{name}] {ds}: SAAE TPR={row[ds]:.3f}")
        ablation[name] = row

    tables.save_all({"table5_ablation": tables.table_ablation(ablation)},
                    cfg["results_dir"])


if __name__ == "__main__":
    main()
