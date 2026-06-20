"""Fig 6 -- cross-domain TPR/FPR heatmaps from results/cross_domain.json."""
import numpy as np, matplotlib.pyplot as plt
from plot_utils import ensure_out
from pgdef.io_utils import load_results

def main():
    r = load_results("cross_domain"); doms = list(r["tpr"].keys())
    TPR = np.array([[r["tpr"][a][b] for b in doms] for a in doms])
    FPR = np.array([[r["fpr"][a][b] for b in doms] for a in doms])
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for ax, M, ttl, cmap in [(axs[0], TPR, "Detection rate (TPR, %)", "RdYlGn"),
                             (axs[1], FPR, "False positive rate (FPR, %)", "RdYlGn_r")]:
        ax.imshow(M, cmap=cmap, vmin=0, vmax=100)
        ax.set_xticks(range(len(doms))); ax.set_yticks(range(len(doms)))
        ax.set_xticklabels(doms); ax.set_yticklabels(doms)
        ax.set_xlabel("Tested on"); ax.set_ylabel("Trained on"); ax.set_title(ttl, fontsize=10)
        for i in range(len(doms)):
            for j in range(len(doms)):
                v = M[i, j]
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9,
                        color="black" if 25 < v < 80 else "white", fontweight="bold")
        for s in ax.spines.values(): s.set_visible(False)
    fig.tight_layout(); fig.savefig(f"{ensure_out()}/fig_crossdomain.pdf", bbox_inches="tight")
    print("wrote fig_crossdomain.pdf")

if __name__ == "__main__":
    main()
