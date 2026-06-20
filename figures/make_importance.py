"""Fig 1 -- Random-Forest importance of the protected features. Recomputes from a
CSV. Usage: python figures/make_importance.py --csv cicids_pgdef.csv"""
import argparse, os, sys, numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from plot_utils import ensure_out
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from pgdef.features import PROTECTED, COMPACT5, resolve
from pgdef.data import load_csv, split
from pgdef.io_utils import dataset_key

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True)
    ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    key = dataset_key(a.csv); df = load_csv(a.csv); cols = resolve(df.columns, PROTECTED)
    Xtr, _, ytr, _ = split(df, cols, seed=a.seed)
    rf = RandomForestClassifier(n_estimators=200, random_state=a.seed,
                                n_jobs=-1).fit(StandardScaler().fit_transform(Xtr), ytr)
    imp = rf.feature_importances_; order = np.argsort(imp)
    names = [cols[i].split("_", 1)[1] for i in order]
    compact = set(resolve(df.columns, COMPACT5.get(key, PROTECTED[:5])))
    colors = ["#c51b7d" if cols[i] in compact else "#4575b4" for i in order]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.barh(range(len(names)), imp[order], color=colors, edgecolor="#333", linewidth=0.3)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Random-Forest importance")
    fig.tight_layout(); fig.savefig(f"{ensure_out()}/fig_importance.pdf", bbox_inches="tight")
    print("wrote fig_importance.pdf")

if __name__ == "__main__":
    main()
