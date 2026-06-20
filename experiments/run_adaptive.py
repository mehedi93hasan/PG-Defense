"""Table VII -- adaptive operating-point control (PI controller) + PSI drift.
The PI controller adjusts the decision threshold online to hold a target FPR; the
PSI monitor reports drift under a synthetic covariate shift."""
import numpy as np
from _common import base_parser
from pgdef.features import PROTECTED, resolve
from pgdef.model import PGDef
from pgdef import metrics
from pgdef.data import load_csv, split
from pgdef.io_utils import dataset_key, save_results

def psi(expected, actual, bins=10):
    qs = np.quantile(expected, np.linspace(0, 1, bins + 1)); qs[0], qs[-1] = -np.inf, np.inf
    e = np.histogram(expected, qs)[0] / len(expected) + 1e-6
    a = np.histogram(actual, qs)[0] / len(actual) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))

def pi_control(scores, y, target_fpr, ap=0.01, ai=0.001, segments=10):
    thr = 0.5; integ = 0.0; chunks = np.array_split(np.arange(len(scores)), segments)
    last = {}
    for ch in chunks:
        s, yy = scores[ch], y[ch]
        fpr = metrics.fpr_at_threshold(s, yy, thr)
        e = fpr - target_fpr; integ += e; thr += ap * e + ai * integ
        thr = min(max(thr, 0.01), 0.99)
        last = {"fpr": metrics.fpr_at_threshold(s, yy, thr) * 100,
                "tpr": metrics.tpr_at_threshold(s, yy, thr) * 100}
    return last

def main():
    ap = base_parser("Adaptive operating-point control + drift")
    ap.add_argument("--targets", type=float, nargs="+", default=[0.01, 0.02])
    a = ap.parse_args(); out = {}
    for path in a.csv:
        key = dataset_key(path); df = load_csv(path)
        cols = resolve(df.columns, PROTECTED)
        Xtr, Xte, ytr, yte = split(df, cols, seed=a.seed)
        m = PGDef(seed=a.seed).fit(Xtr, ytr); s = m.proba(Xte)
        rec = {"fixed_0.5": {"fpr": round(metrics.fpr_at_threshold(s, yte, 0.5) * 100, 1)}}
        for t in a.targets:
            r = pi_control(s, yte, t); rec[f"target_{t}"] = {k: round(v, 1) for k, v in r.items()}
        # synthetic covariate shift -> PSI trajectory of benign scores
        ben = Xte[yte == 0]; atk_mean = Xtr[ytr == 1].mean(0)
        traj = []
        for alpha in np.linspace(0, 0.9, 10):
            shifted = (1 - alpha) * ben + alpha * atk_mean
            traj.append(round(psi(m.proba(Xte[yte == 0]), m.proba(shifted)), 2))
        rec["psi_trajectory"] = traj
        out[key] = rec
        print(f"[{key}] {rec}")
    save_results("adaptive", out)

if __name__ == "__main__":
    main()
