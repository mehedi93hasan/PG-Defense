#!/usr/bin/env bash
# Reproduce all tables and figures. Pass the three feature CSVs as arguments:
#   bash scripts/reproduce_all.sh cicids_pgdef.csv unsw_pgdef.csv edge_pgdef.csv
set -e
CSVS="$@"
[ -z "$CSVS" ] && { echo "usage: bash scripts/reproduce_all.sh <csv...>"; exit 1; }
cd "$(dirname "$0")/.."

echo "== Table II: dataset statistics =="
python experiments/run_dataset_stats.py     --csv $CSVS
echo "== Tables III/IV: main detection + cross-validation =="
python experiments/run_main_detection.py    --csv $CSVS --folds 5
echo "== Tables V/VI: lightweight edge model =="
python experiments/run_lightweight.py       --csv $CSVS
echo "== Table VII: adaptive control + drift =="
python experiments/run_adaptive.py          --csv $CSVS
echo "== Table VIII: multi-attack evasion =="
python experiments/run_evasion.py           --csv $CSVS
echo "== Table IX: defense baselines =="
python experiments/run_defense_baselines.py --csv $CSVS
echo "== Table X: deep-learning baselines =="
python experiments/run_dl_baselines.py      --csv $CSVS
echo "== Fig 6: cross-domain (needs >=2 datasets) =="
python experiments/run_cross_domain.py      --csv $CSVS
echo "== Fig 7: per-attack (needs 'class' column) =="
python experiments/run_per_attack.py        --csv $CSVS

echo "== Figures =="
python figures/make_all.py
echo "Done. Results in results/  Figures in figures/out/"
