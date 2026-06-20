"""Tables III & IV -- main detection at default threshold + 5-fold CV (protected
detector). Reports Acc/TPR/FPR/Prec/F1 and cross-validated mean+/-std."""
import numpy as np
from _common import base_parser
from pgdef.features import PROTECTED, resolve
from pgdef.model import PGDef
from pgdef import metrics
from pgdef.data import load_csv
from pgdef.io_utils import dataset_key, save_results
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, f1_score

def main():
    ap = base_parser("Main detection + cross-validation")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    out = {}
    for path in a.csv:
        key = dataset_key(path); df = load_csv(path)
        cols = resolve(df.columns, PROTECTED)
        X = np.nan_to_num(df[cols].values.astype("float32"))
        y = df["label"].astype(int).values
        # default-threshold single split
        from pgdef.data import split
        Xtr, Xte, ytr, yte = split(df, cols, seed=a.seed)
        m = PGDef(seed=a.seed).fit(Xtr, ytr); s = m.proba(Xte)
        pred = (s >= 0.5).astype(int)
        default = {"acc": accuracy_score(yte, pred) * 100,
                   "tpr": metrics.tpr_at_threshold(s, yte, 0.5) * 100,
                   "fpr": metrics.fpr_at_threshold(s, yte, 0.5) * 100,
                   "prec": precision_score(yte, pred, zero_division=0) * 100,
                   "f1": f1_score(yte, pred, zero_division=0) * 100}
        # k-fold at matched FPR
        skf = StratifiedKFold(n_splits=a.folds, shuffle=True, random_state=a.seed)
        rows = []
        for tr, te in skf.split(X, y):
            mm = PGDef(seed=a.seed).fit(X[tr], y[tr]); ss = mm.proba(X[te])
            op = metrics.operating_point(ss, y[te], a.target_fpr)
            pr = (ss >= op["threshold"]).astype(int)
            rows.append([op["auc"] * 100, op["tpr"], op["fpr"],
                         accuracy_score(y[te], pr) * 100,
                         f1_score(y[te], pr, zero_division=0) * 100])
        rows = np.array(rows); mean = rows.mean(0); std = rows.std(0)
        cv = {k: [round(mean[i], 2), round(std[i], 2)]
              for i, k in enumerate(["auc", "tpr_at_fpr", "fpr", "acc", "f1"])}
        out[key] = {"default_threshold": default, "cross_val": cv}
        print(f"\n[{key}] default: acc={default['acc']:.2f} tpr={default['tpr']:.2f} "
              f"fpr={default['fpr']:.2f} f1={default['f1']:.2f}")
        print(f"[{key}] {a.folds}-fold AUC={cv['auc'][0]}+/-{cv['auc'][1]} "
              f"TPR@FPR={cv['tpr_at_fpr'][0]}+/-{cv['tpr_at_fpr'][1]}")
    save_results("main_detection", out)

if __name__ == "__main__":
    main()
