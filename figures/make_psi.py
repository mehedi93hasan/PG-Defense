"""Fig 3 -- PSI drift trajectory from results/adaptive.json."""
import matplotlib.pyplot as plt
from plot_utils import ensure_out
from pgdef.io_utils import load_results

def main():
    r = load_results("adaptive")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for key, rec in r.items():
        ax.plot(range(len(rec["psi_trajectory"])), rec["psi_trajectory"], "o-", label=key)
    ax.axhline(0.2, ls="--", color="grey", lw=0.8, label="threshold (0.2)")
    ax.set_xlabel("Stream segment"); ax.set_ylabel("PSI")
    ax.legend(frameon=False, fontsize=9); ax.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout(); fig.savefig(f"{ensure_out()}/fig_psi.pdf", bbox_inches="tight")
    print("wrote fig_psi.pdf")

if __name__ == "__main__":
    main()
