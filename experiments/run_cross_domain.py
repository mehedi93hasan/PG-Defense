"""Fig 6 data -- cross-domain train->test transfer (TPR/FPR matrix), protected
detector. Pass all dataset CSVs at once."""
import numpy as np
from _common import base_parser
from pgdef.features import PROTECTED, resolve
from pgdef.model import PGDef
from pgdef import metrics
from pgdef.data import load_csv
from pgdef.io_utils import dataset_key, save_results
from sklearn.model_selection import train_test_split

def main():
    a = base_parser("Cross-domain transfer").parse_args()
    data = {}
    for path in a.csv:
        key = dataset_key(path); df = load_csv(path)
        cols = resolve(df.columns, PROTECTED)
        X = np.nan_to_num(df[cols].values.astype("float32"))
        y = df["label"].astype(int).values
        itr, ite = train_test_split(np.arange(len(X)), test_size=0.3,
                                    stratify=y, random_state=a.seed)
        data[key] = (X, y, itr, ite)
    models = {k: PGDef(seed=a.seed).fit(v[0][v[2]], v[1][v[2]]) for k, v in data.items()}
    tpr = {}; fpr = {}
    for trk, m in models.items():
        tpr[trk] = {}; fpr[trk] = {}
        for tek, (X, y, _, ite) in data.items():
            s = m.proba(X[ite]); thr = metrics.threshold_at_fpr(s, y[ite], a.target_fpr) \
                if tek == trk else 0.5
            tpr[trk][tek] = round(metrics.tpr_at_threshold(s, y[ite], thr) * 100, 1)
            fpr[trk][tek] = round(metrics.fpr_at_threshold(s, y[ite], thr) * 100, 1)
        print(f"train={trk:7s} TPR={tpr[trk]}  FPR={fpr[trk]}")
    save_results("cross_domain", {"tpr": tpr, "fpr": fpr})

if __name__ == "__main__":
    main()
