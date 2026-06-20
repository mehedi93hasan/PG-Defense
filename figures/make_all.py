"""Generate every figure that has a results/*.json available."""
import subprocess, os, sys
HERE = os.path.dirname(__file__)
JSON_DRIVEN = ["make_crossdomain", "make_perattack", "make_baseline_collapse",
               "make_pareto", "make_psi"]
for m in JSON_DRIVEN:
    try:
        subprocess.run([sys.executable, os.path.join(HERE, m + ".py")], check=True)
    except Exception as e:
        print(f"[skip] {m}: {e}")
print("Note: make_roc_attack.py and make_importance.py need --csv (run separately).")
