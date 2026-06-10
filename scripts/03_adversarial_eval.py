#!/usr/bin/env python3
"""03 - Adversarial robustness under seven attacks + adaptive white-box (Table VIII).

  python scripts/03_adversarial_eval.py --config configs/default.yaml
"""

import argparse
import warnings

from pgdef.attacks.adversarial import (ALL_ATTACKS, evaluate_attacks,
                                       evaluate_whitebox)
from pgdef.eval import tables
from pgdef.pipeline import (available_datasets, build_detector, load_config,
                            load_split)

warnings.filterwarnings("ignore")
METHODS = ["FA-CNN", "GTAE-IDS", "Adv. Retrain", "PG-Def"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    datasets = available_datasets(cfg)
    if not datasets:
        raise SystemExit("No feature CSVs found. Run 01_extract_features.py first.")

    adv = {ds: {} for ds in datasets}
    wb = {ds: {} for ds in datasets}
    for ds in datasets:
        sp = load_split(cfg, ds)
        for m in METHODS:
            det = build_detector(m, cfg["seed"]).fit(sp.X_train, sp.y_train)
            print(f"[{ds}] {m}: crafting {len(ALL_ATTACKS)} attacks ...")
            adv[ds][m] = evaluate_attacks(det, sp.X_test, sp.y_test, sp.X_train)
            wb[ds][m] = evaluate_whitebox(det, sp.X_test, sp.y_test, sp.X_train)
            print(f"          white-box PGD TPR = {wb[ds][m]:.3f}")

    tables.save_all({"table8_adversarial": tables.table_adversarial(adv)},
                    cfg["results_dir"])
    print("\nAdaptive white-box (PGD) TPR per dataset/method:")
    for ds in datasets:
        for m in METHODS:
            print(f"  {ds:12s} {m:14s} {wb[ds][m]:.3f}")


if __name__ == "__main__":
    main()
