"""Table VIII -- detection under four realizable attacks (mimicry/FGSM/BIM/PGD),
header detector (all 30) vs protocol detector (protected 20)."""
import numpy as np
from _common import base_parser
from pgdef.features import COLUMN_NAMES, PROTECTED, MANIPULABLE, resolve
from pgdef.model import PGDef
from pgdef import attacks, metrics
from pgdef.data import load_csv, split
from pgdef.io_utils import dataset_key, save_results

def main():
    ap = base_parser("Multi-attack evasion"); ap.add_argument("--eps", type=float, default=0.15)
    a = ap.parse_args(); out = {}
    for path in a.csv:
        key = dataset_key(path); df = load_csv(path)
        allc = resolve(df.columns, COLUMN_NAMES)
        protc = resolve(df.columns, PROTECTED)
        manip = resolve(df.columns, MANIPULABLE)
        Xtr, Xte, ytr, yte = split(df, allc, seed=a.seed)
        sigma = Xtr.std(0) + 1e-9; bmean = Xtr[ytr == 0].mean(0)
        midx = [allc.index(c) for c in manip]; pidx = [allc.index(c) for c in protc]
        header = PGDef(seed=a.seed).fit(Xtr, ytr)
        protocol = PGDef(seed=a.seed).fit(Xtr[:, pidx], ytr)
        adv = {"mimicry": attacks.mimicry(Xte, yte, midx, bmean),
               "fgsm": attacks.gradient_attack(header, Xte, yte, midx, sigma, a.eps, 1),
               "bim":  attacks.gradient_attack(header, Xte, yte, midx, sigma, a.eps, 10),
               "pgd":  attacks.gradient_attack(header, Xte, yte, midx, sigma, a.eps, 10, True, a.seed)}
        def tpr(model, M, cols=None):
            X = M[:, cols] if cols is not None else M
            return round(metrics.operating_point(model.proba(X), yte, a.target_fpr)["tpr"], 1)
        rec = {"header": {"clean": tpr(header, Xte)},
               "protocol": {"clean": tpr(protocol, Xte, pidx)}}
        for name, Xa in adv.items():
            rec["header"][name] = tpr(header, Xa)
            rec["protocol"][name] = tpr(protocol, Xa, pidx)
        out[key] = rec
        print(f"\n[{key}] header  :", rec["header"])
        print(f"[{key}] protocol:", rec["protocol"])
    save_results("evasion", out)

if __name__ == "__main__":
    main()
