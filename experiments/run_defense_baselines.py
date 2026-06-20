"""Table IX -- defense baselines under an unseen PGD attack: conventional (all 30),
adversarial training (conventional + mimicry examples), and PG-Def (protected)."""
import numpy as np
from _common import base_parser
from pgdef.features import COLUMN_NAMES, PROTECTED, MANIPULABLE, resolve
from pgdef.model import PGDef
from pgdef import attacks, metrics
from pgdef.data import load_csv, split
from pgdef.io_utils import dataset_key, save_results

def main():
    ap = base_parser("Defense baselines"); ap.add_argument("--eps", type=float, default=0.15)
    a = ap.parse_args(); out = {}
    for path in a.csv:
        key = dataset_key(path); df = load_csv(path)
        allc = resolve(df.columns, COLUMN_NAMES); protc = resolve(df.columns, PROTECTED)
        manip = resolve(df.columns, MANIPULABLE)
        Xtr, Xte, ytr, yte = split(df, allc, seed=a.seed)
        sigma = Xtr.std(0) + 1e-9; bmean = Xtr[ytr == 0].mean(0)
        midx = [allc.index(c) for c in manip]; pidx = [allc.index(c) for c in protc]
        conv = PGDef(seed=a.seed).fit(Xtr, ytr)
        # adversarial training: augment with mimicry examples
        Xmim_tr = attacks.mimicry(Xtr, ytr, midx, bmean)
        Xa = np.vstack([Xtr, Xmim_tr[ytr == 1]]); ya = np.r_[ytr, ytr[ytr == 1]]
        advtr = PGDef(seed=a.seed).fit(Xa, ya)
        pg = PGDef(seed=a.seed).fit(Xtr[:, pidx], ytr)
        # unseen PGD attack at test time
        Xpgd = attacks.gradient_attack(conv, Xte, yte, midx, sigma, a.eps, 10, True, a.seed)
        def m2(model, M, cols=None):
            X = M[:, cols] if cols is not None else M
            op = metrics.operating_point(model.proba(X), yte, a.target_fpr)
            return [round(op["tpr"], 1), round(op["fpr"], 1)]
        rec = {
            "conventional":   {"clean": m2(conv, Xte),  "pgd": m2(conv, Xpgd),  "mb": round(conv.model_size_mb(), 1)},
            "adv_training":   {"clean": m2(advtr, Xte), "pgd": m2(advtr, Xpgd), "mb": round(advtr.model_size_mb(), 1)},
            "pgdef":          {"clean": m2(pg, Xte, pidx), "pgd": m2(pg, Xpgd, pidx), "mb": round(pg.model_size_mb(), 1)},
        }
        out[key] = rec
        for k, v in rec.items(): print(f"[{key}] {k:14s} clean={v['clean']} pgd={v['pgd']} {v['mb']}MB")
    save_results("defense_baselines", out)

if __name__ == "__main__":
    main()
