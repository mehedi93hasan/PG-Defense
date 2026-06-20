# PG-Def: Manipulation-Cost-Grounded Features for Adversarially Robust and Lightweight Network Intrusion Detection

Reference implementation for the PG-Def paper. PG-Def reframes NIDS robustness as a
**feature-contract** problem: it partitions 30 protocol-grounded TCP/IP flow features by
**manipulation cost** into a cheaply spoofable (*manipulable*) set and a behaviourally
*protected* set, and trains the detector on the protected set alone. The result is a
detector that is **immune to realizable manipulable-feature attacks** (header/payload
spoofing), supported by a formal evasion-cost bound, in an edge-deployable footprint.

> **Core finding:** adversarial robustness in NIDS is a property of *which features a
> detector is allowed to trust*, not of the model architecture — tree ensembles and deep
> networks alike collapse on the manipulable features and survive on the protected ones.

## Architecture

<img width="2752" height="1265" alt="fig_2" src="https://github.com/user-attachments/assets/c8626fd5-8699-424a-84dc-a4d7357bc26d" />






## Repository structure

```
pgdef/            core library
  features.py       30 features, manipulation-cost partition (P/M), compact subsets
  model.py          PGDef soft-voting ensemble (RF + XGBoost + LR) on protected features
  attacks.py        realizable attacks (mimicry, FGSM, BIM, PGD) confined to manipulable axes
  data.py           loading, split, standardisation, SMOTE
  metrics.py        AUC, TPR@FPR, operating-point calibration
experiments/      one script per paper table  (writes results/*.json)
figures/          one script per paper figure (reads results/*.json -> PDF)
extraction/       pcap -> 30-feature CSV streaming extractor
scripts/          reproduce_all.sh
```

## Installation

```bash
git clone https://github.com/mehedi93hasan/PG-Defense.git
cd PG-Defense
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`torch` is needed only for the deep-learning baselines; `dpkt` only for pcap extraction.

## Data

The experiments operate on **feature CSVs**: 30 `phi{N}_{name}` columns + a binary `label`
column (0 = benign, 1 = attack), and an optional `class` column (attack-type name) for the
per-attack figure. Datasets used in the paper: CICIDS2017, UNSW-NB15, Edge-IIoTset.

To build the CSVs from raw captures, see [`extraction/README.md`](extraction/README.md).
Name files so the dataset is auto-detected, e.g. `cicids_pgdef.csv`, `unsw_pgdef.csv`,
`edge_pgdef.csv`.

## Reproducing the paper

One command:

```bash
bash scripts/reproduce_all.sh cicids_pgdef.csv unsw_pgdef.csv edge_pgdef.csv
```



## Method (one paragraph)

The detector is a fixed weighted soft-voting ensemble — Random Forest (200 trees),
XGBoost (300 trees, depth 6, lr 0.1, subsample 0.9), and L2 Logistic Regression, weights
0.4/0.4/0.2 — trained with SMOTE and z-score standardisation on the 20 protected features.
Hyperparameters are fixed once and reused unchanged across datasets. Attacks perturb only
the 10 manipulable features (the realizable, function-preserving regime); because the
deployed detector never reads those features, the attacks cannot move its decision. A
five-feature compact distillation preserves AUC at ~1.3 MB for edge deployment.

## Notes on integrity

All numbers in the paper are computed from real runs of these scripts on the feature CSVs;
nothing is hardcoded. Footprint and latency are measured on a development platform (see the
paper's reframed Section V) — the Raspberry Pi 4B is a specification-defined deployment
target, and on-device benchmarking is stated as future work.

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite the associated paper if you use this code.

## License

MIT — see [`LICENSE`](LICENSE).
