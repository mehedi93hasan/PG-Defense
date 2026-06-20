"""Shared argument parsing / path bootstrap for experiment scripts."""
import os, sys, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def base_parser(desc):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--csv", nargs="+", required=True,
                    help="feature CSV(s) with phi* columns + label")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-fpr", type=float, default=0.01)
    return ap
