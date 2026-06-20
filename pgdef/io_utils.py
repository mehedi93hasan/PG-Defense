"""Small helpers: dataset-key inference and results persistence."""
import json, os

def dataset_key(path):
    p = os.path.basename(path).lower()
    if "cicids" in p or "cic-ids" in p or "cicids2017" in p: return "cicids"
    if "unsw" in p:  return "unsw"
    if "edge" in p:  return "edge"
    return os.path.splitext(os.path.basename(p))[0]

def save_results(name, obj, outdir="results"):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[saved] {path}")
    return path

def load_results(name, outdir="results"):
    with open(os.path.join(outdir, f"{name}.json")) as f:
        return json.load(f)
