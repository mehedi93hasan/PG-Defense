"""Tables V & VI -- full/protocol/compact at matched FPR (AUC, TPR, size, latency)
and the selected five-feature subsets."""
import time, numpy as np
from _common import base_parser
from pgdef.features import COLUMN_NAMES, PROTECTED, COMPACT5, resolve
from pgdef.model import PGDef
from pgdef import metrics
from pgdef.data import load_csv, split
from pgdef.io_utils import dataset_key, save_results

def _latency_ms(model, X, n=2000):
    n = min(n, len(X)); t0 = time.perf_counter(); model.proba(X[:n])
    return (time.perf_counter() - t0) / n * 1e3

def main():
    a = base_parser("Lightweight edge model").parse_args(); out = {}
    for path in a.csv:
        key = dataset_key(path); df = load_csv(path)
        compact = COMPACT5.get(key, PROTECTED[:5])
        configs = {"full": COLUMN_NAMES, "protocol": PROTECTED, "compact": compact}
        rec = {}
        for name, wanted in configs.items():
            cols = resolve(df.columns, wanted)
            Xtr, Xte, ytr, yte = split(df, cols, seed=a.seed)
            m = PGDef(seed=a.seed).fit(Xtr, ytr); s = m.proba(Xte)
            op = metrics.operating_point(s, yte, a.target_fpr)
            rec[name] = {"feat": len(cols), "auc": round(op["auc"], 3),
                         "tpr_at_fpr": round(op["tpr"], 1),
                         "mb": round(m.model_size_mb(), 2),
                         "ms_per_flow": round(_latency_ms(m, Xte), 3)}
            print(f"[{key}] {name:8s} feat={rec[name]['feat']:2d} "
                  f"AUC={rec[name]['auc']} TPR={rec[name]['tpr_at_fpr']} "
                  f"{rec[name]['mb']}MB {rec[name]['ms_per_flow']}ms")
        rec["compact_features"] = [resolve(df.columns, [c])[0] for c in compact]
        out[key] = rec
    save_results("lightweight", out)

if __name__ == "__main__":
    main()
