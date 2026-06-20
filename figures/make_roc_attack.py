"""Fig 4 -- ROC of header vs protocol detector, clean and under mimicry. Recomputes
from a CSV. Usage: python figures/make_roc_attack.py --csv cicids_pgdef.csv"""
import argparse, os, sys, numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from plot_utils import ensure_out
from sklearn.metrics import roc_curve
from pgdef.features import COLUMN_NAMES, PROTECTED, MANIPULABLE, resolve
from pgdef.model import PGDef
from pgdef import attacks
from pgdef.data import load_csv, split

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True)
    ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    df = load_csv(a.csv); allc = resolve(df.columns, COLUMN_NAMES)
    protc = resolve(df.columns, PROTECTED); manip = resolve(df.columns, MANIPULABLE)
    Xtr, Xte, ytr, yte = split(df, allc, seed=a.seed)
    midx = [allc.index(c) for c in manip]; pidx = [allc.index(c) for c in protc]
    bmean = Xtr[ytr == 0].mean(0); Xmim = attacks.mimicry(Xte, yte, midx, bmean)
    header = PGDef(seed=a.seed).fit(Xtr, ytr); protocol = PGDef(seed=a.seed).fit(Xtr[:, pidx], ytr)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for s, lbl, st in [(header.proba(Xte), "header, clean", "-"),
                       (header.proba(Xmim), "header, mimicry", "--"),
                       (protocol.proba(Xte[:, pidx]), "protocol, clean", "-"),
                       (protocol.proba(Xmim[:, pidx]), "protocol, mimicry", "--")]:
        fpr, tpr, _ = roc_curve(yte, s); ax.plot(fpr, tpr, st, label=lbl, lw=1.5)
    ax.plot([0, 1], [0, 1], ":", color="grey", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(frameon=False, fontsize=8); ax.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout(); fig.savefig(f"{ensure_out()}/fig_roc_attack.pdf", bbox_inches="tight")
    print("wrote fig_roc_attack.pdf")

if __name__ == "__main__":
    main()
