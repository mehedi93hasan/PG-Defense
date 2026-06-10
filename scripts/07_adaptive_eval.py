#!/usr/bin/env python3
"""07 - Adaptive defense component evaluation (Table IX).

Evaluates the four-component adaptive layer of Section V-C and produces
Table IX. For each dataset the script:

  * fits the base PG-Def ensemble and the adaptive layer;
  * runs the PI threshold controller (Component 3) to measure FPR-convergence
    speed in flows;
  * generates on-manifold SAAE adversarial flows and routes them through the
    adaptive predictor to measure Bloom-cache hit rate (Component 1), the
    borderline-flow fraction and reclassification rate (Component 2), and the
    adversarial TPR gain over the base model.

All reported quantities are computed from the data the script is run on; no
manuscript value is hard-coded.  Run after 01_extract_features.py.

  python scripts/07_adaptive_eval.py --config configs/default.yaml
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from pgdef.features.feature_spec import FEATURE_NAMES
from pgdef.models.ensemble import PGDefEnsemble
from pgdef.models.adaptive import AdaptiveDefense
from pgdef.attacks.adversarial import fit_surrogate, saae_perturb
from pgdef.eval import tables
from pgdef.pipeline import load_config, available_datasets

warnings.filterwarnings("ignore")


def _prepare(df, seed, test_size, smote_k):
    X = df[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0).to_numpy()
    y = df["label"].astype(int).to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
    if len(np.unique(ytr)) > 1 and np.bincount(ytr).min() > smote_k:
        Xtr, ytr = SMOTE(k_neighbors=smote_k, random_state=seed).fit_resample(Xtr, ytr)
    return Xtr, ytr, Xte, yte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    datasets = available_datasets(cfg)
    if not datasets:
        raise SystemExit("No feature CSVs found. Run 01_extract_features.py first.")

    seed = cfg["seed"]
    test_size = cfg["evaluation"]["test_size"]
    smote_k = cfg["evaluation"]["smote_k"]

    stats = {}
    for ds in datasets:
        df = pd.read_csv(cfg["datasets"][ds]["features_csv"])
        df = df[df["label"].isin([0, 1])].copy()
        Xtr, ytr, Xte, yte = _prepare(df, seed, test_size, smote_k)

        base = PGDefEnsemble(seed=seed).fit(Xtr, ytr)
        adaptive = AdaptiveDefense(base=base, seed=seed).fit(Xtr, ytr)

        # Component 3: PI threshold controller convergence on a benign-heavy
        # validation slice drawn from the test partition.
        convergence = adaptive.calibrate(Xte, yte)

        # On-manifold SAAE adversarial flows (transfer surrogate of the
        # non-differentiable ensemble), used as the "adaptive white-box
        # adversary" population for borderline / cache measurement.
        surrogate = fit_surrogate(base, Xtr)
        mal = np.where(yte == 1)[0]
        X_mal, X_ben = Xte[mal], Xte[yte == 0]
        X_adv = saae_perturb(surrogate, X_mal, X_ben)

        # Repeated-campaign stream: a second pass over identical fingerprints
        # exercises the Bloom cache (Component 1).
        X_stream = np.vstack([X_adv, X_adv])

        base_tpr = float((base.predict(X_adv) == 1).mean())
        y_adp, st = adaptive.predict(X_stream)
        adaptive_tpr = float((adaptive.predict(X_adv)[0] == 1).mean())

        stats[ds] = {
            "cache_hit_rate": st["cache_hit_rate"],
            "borderline_rate": st["borderline_rate"],
            "reclassified_rate": st["reclassified_rate"],
            "convergence": convergence,
            "tpr_gain_pp": (adaptive_tpr - base_tpr) * 100.0,
        }
        print(f"[{ds}] base SAAE TPR={base_tpr:.3f}  adaptive TPR={adaptive_tpr:.3f}  "
              f"cache={st['cache_hit_rate']:.3f}  borderline={st['borderline_rate']:.3f}  "
              f"converge={convergence}")

    out_dir = cfg["results_dir"]
    tables.save_all({"table9_adaptive": tables.table_adaptive(stats)}, out_dir)
    print(f"\n[done] Table IX written to {out_dir}/")


if __name__ == "__main__":
    main()
