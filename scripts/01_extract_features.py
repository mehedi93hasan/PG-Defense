#!/usr/bin/env python3
"""01 - Extract the 30 protocol-grounded features from PCAP files.

Examples
--------
  # Extract one or more PCAPs to a feature CSV (unlabelled)
  python scripts/01_extract_features.py --pcap "data/raw/cicids2017/*.pcap" \
      --out data/cicids2017_pgdef.csv

  # Join labels from a CICIDS2017 GeneratedLabelledFlows CSV (5-tuple match)
  python scripts/01_extract_features.py --pcap "data/raw/*.pcap" \
      --out data/cicids2017_pgdef.csv --label-csv data/raw/labels.csv

Label CSV format (optional): rows with columns
``ip_src,ip_dst,port_src,port_dst,proto,label`` (label in {0,1}). Flows are
joined on the canonical 5-tuple. See README for per-dataset labelling notes.
"""

import argparse
import glob

import pandas as pd

from pgdef.features.pcap_extractor import PcapExtractor, _canonical_key


def make_labeller(label_csv: str):
    df = pd.read_csv(label_csv)
    lut = {}
    for _, r in df.iterrows():
        key, _ = _canonical_key(str(r["ip_src"]), str(r["ip_dst"]),
                                int(r["port_src"]), int(r["port_dst"]),
                                int(r["proto"]))
        lut[key] = int(r["label"])
    return lambda key, ts: lut.get(key)


def main():
    ap = argparse.ArgumentParser(description="Extract 30 protocol-grounded features from PCAPs.")
    ap.add_argument("--pcap", required=True, help="PCAP file or glob pattern")
    ap.add_argument("--out", required=True, help="output feature CSV path")
    ap.add_argument("--label-csv", default=None, help="optional 5-tuple label CSV")
    ap.add_argument("--idle-timeout", type=float, default=120.0)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.pcap))
    if not paths:
        raise SystemExit(f"no PCAP files matched: {args.pcap}")
    labeller = make_labeller(args.label_csv) if args.label_csv else None

    frames = []
    for p in paths:
        print(f"[extract] {p}")
        ext = PcapExtractor(idle_timeout=args.idle_timeout, labeller=labeller)
        frames.append(ext.extract_to_dataframe(p))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(args.out, index=False)
    print(f"[done] {len(df)} flows -> {args.out}")


if __name__ == "__main__":
    main()
