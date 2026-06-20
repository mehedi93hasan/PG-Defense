"""Fig 7 -- per-class detection bars from results/per_attack.json."""
import numpy as np, matplotlib.pyplot as plt
from plot_utils import ensure_out
from pgdef.io_utils import load_results

def main():
    r = load_results("per_attack"); keys = list(r.keys())
    fig, axs = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.6), squeeze=False)
    for ax, key in zip(axs[0], keys):
        items = sorted(r[key].items(), key=lambda kv: kv[1])
        names = [k for k, _ in items]; vals = [v for _, v in items]
        cols = ["#2c7fb8" if v >= 70 else ("#7fcdbb" if v >= 50 else "#edf8b1") for v in vals]
        y = np.arange(len(names)); ax.barh(y, vals, color=cols, edgecolor="#333", linewidth=0.4)
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8); ax.set_xlim(0, 100)
        ax.set_xlabel("Detection rate (%)"); ax.set_title(key, fontsize=10)
        ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(f"{ensure_out()}/fig_perattack.pdf", bbox_inches="tight")
    print("wrote fig_perattack.pdf")

if __name__ == "__main__":
    main()
