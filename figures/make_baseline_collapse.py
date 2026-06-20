"""Fig 5 -- clean vs under-attack TPR for DL baselines from results/dl_baselines.json."""
import numpy as np, matplotlib.pyplot as plt
from plot_utils import ensure_out
from pgdef.io_utils import load_results

def main():
    r = load_results("dl_baselines"); keys = list(r.keys())
    fig, axs = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.3), squeeze=False)
    for ax, key in zip(axs[0], keys):
        methods = list(r[key].keys())
        clean = [r[key][m]["clean_tpr"] for m in methods]
        atk = [r[key][m]["atk_tpr"] for m in methods]
        x = np.arange(len(methods)); w = 0.38
        ax.bar(x - w / 2, clean, w, label="Clean", color="#4d9221", edgecolor="#333", linewidth=0.4)
        ax.bar(x + w / 2, atk, w, label="Under mimicry", color="#c51b7d", edgecolor="#333", linewidth=0.4)
        ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9, rotation=20)
        ax.set_ylim(0, 100); ax.set_ylabel("TPR @ 1% FPR (%)"); ax.set_title(key, fontsize=10)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6); ax.set_axisbelow(True)
    axs[0][0].legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(f"{ensure_out()}/fig_baseline_collapse.pdf", bbox_inches="tight")
    print("wrote fig_baseline_collapse.pdf")

if __name__ == "__main__":
    main()
