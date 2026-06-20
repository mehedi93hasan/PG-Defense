"""Fig 2 -- AUC vs model size from results/lightweight.json."""
import matplotlib.pyplot as plt
from plot_utils import ensure_out
from pgdef.io_utils import load_results

def main():
    r = load_results("lightweight")
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for key, rec in r.items():
        xs = [rec[c]["mb"] for c in ("full", "protocol", "compact")]
        ys = [rec[c]["auc"] for c in ("full", "protocol", "compact")]
        ax.plot(xs, ys, "o-", label=key)
        for c in ("full", "protocol", "compact"):
            ax.annotate(c, (rec[c]["mb"], rec[c]["auc"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("Model size (MB, log)"); ax.set_ylabel("ROC-AUC")
    ax.legend(frameon=False, fontsize=9); ax.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout(); fig.savefig(f"{ensure_out()}/fig_pareto.pdf", bbox_inches="tight")
    print("wrote fig_pareto.pdf")

if __name__ == "__main__":
    main()
