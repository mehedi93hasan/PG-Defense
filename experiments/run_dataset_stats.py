"""Table II -- dataset statistics (flow counts, class breakdown)."""
from _common import base_parser
from pgdef.io_utils import dataset_key, save_results
import pandas as pd

def main():
    a = base_parser("Dataset statistics").parse_args()
    out = {}
    for path in a.csv:
        df = pd.read_csv(path); df.columns = [c.strip() for c in df.columns]
        y = df["label"].astype(int)
        rec = {"flows": int(len(df)), "benign": int((y == 0).sum()),
               "attack": int((y == 1).sum())}
        if "class" in df.columns:
            rec["classes"] = {str(k): int(v) for k, v in
                              df.loc[y == 1, "class"].value_counts().items()}
            rec["n_classes"] = len(rec["classes"])
        key = dataset_key(path); out[key] = rec
        print(f"{key:12s} flows={rec['flows']:>8} benign={rec['benign']:>7} "
              f"attack={rec['attack']:>8} classes={rec.get('n_classes','?')}")
    save_results("dataset_stats", out)

if __name__ == "__main__":
    main()
