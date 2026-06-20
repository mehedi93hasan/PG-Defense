"""Shared matplotlib style for publication figures (vector PDF, embedded fonts)."""
import os, sys, matplotlib as mpl
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
mpl.use("Agg")
mpl.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans",
                     "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.8})
OUTDIR = "figures/out"
def ensure_out():
    os.makedirs(OUTDIR, exist_ok=True); return OUTDIR
