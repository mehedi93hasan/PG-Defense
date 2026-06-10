#!/usr/bin/env python3
"""06 - Resource consumption benchmark (Table IV).

Measures per-flow latency, throughput, and model memory for PG-Def and the
baselines on the host platform. On a Raspberry Pi 4B the edge rows are produced
automatically; on a desktop the desktop rows are produced. Power/CPU are read
from on-board sensors when available, otherwise reported as '-'.

  python scripts/06_resource_bench.py --config configs/default.yaml
"""

import argparse
import warnings

from pgdef.eval import tables
from pgdef.eval.resources import resource_profile
from pgdef.pipeline import (available_datasets, build_detector, detector_memory,
                            load_config, load_split)

warnings.filterwarnings("ignore")
METHODS = ["FA-CNN", "GTAE-IDS", "DAE", "Adv. Retrain", "DNN", "Std. RF", "PG-Def"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    datasets = available_datasets(cfg)
    if not datasets:
        raise SystemExit("No feature CSVs found. Run 01_extract_features.py first.")

    sp = load_split(cfg, datasets[0])
    profiles = {}
    for m in METHODS:
        det = build_detector(m, cfg["seed"]).fit(sp.X_train, sp.y_train)
        prof = resource_profile(det, sp.X_test, detector_memory(m, det),
                                include_flow_table=False)  # memory already includes it for PG-Def
        profiles[m] = prof
        print(f"{m:14s} RAM={prof['ram_mb']:.1f}MB  lat={prof['latency_ms']:.3f}ms  "
              f"tput={prof['throughput_fps']:.0f} fl/s")

    tables.save_all({"table4_resources": tables.table_resources(profiles)},
                    cfg["results_dir"])


if __name__ == "__main__":
    main()
