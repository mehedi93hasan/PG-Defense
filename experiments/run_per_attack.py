"""Fig 7 data -- per-class detection rate of the protected detector. Requires a
'class' column naming each attack type."""
import numpy as np
from _common import base_parser
from pgdef.features import PROTECTED, resolve
from pgdef.model import PGDef
from pgdef import metrics
from pgdef.data import load_csv
from pgdef.io_utils import dataset_key, save_results
from sklearn.model_selection import train_test_split

def main():
    a = base_parser("Per-attack detection").parse_args(); out = {}
    for path in a.csv:
        key = dataset_key(path); df = load_csv(path)
        if "class" not in df.columns:
            print(f"[{key}] no 'class' column -- skipping"); continue
        cols = resolve(df.columns, PROTECTED)
        X = np.nan_to_num(df[cols].values.astype("float32"))
        y = df["label"].astype(int).values
        itr, ite = train_test_split(np.arange(len(df)), test_size=0.3,
                                    stratify=y, random_state=a.seed)
        m = PGDef(seed=a.seed).fit(X[itr], y[itr])
        s = m.proba(X[ite]); thr = metrics.threshold_at_fpr(s, y[ite], a.target_fpr)
        cls = df["class"].values[ite]
        rec = {}
        for c in sorted(set(cls[y[ite] == 1])):
            sel = (cls == c) & (y[ite] == 1)
            if sel.sum() >= 20:
                rec[str(c)] = round((s[sel] >= thr).mean() * 100, 1)
        out[key] = dict(sorted(rec.items(), key=lambda kv: -kv[1]))
        print(f"[{key}] per-class:", out[key])
    save_results("per_attack", out)

if __name__ == "__main__":
    main()
