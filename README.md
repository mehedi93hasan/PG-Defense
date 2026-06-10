# PG-Def: Protocol-Grounded Lightweight Defense for Adversarially Robust NIDS

Reproduction package for **“PG-Def: A Protocol-Grounded Lightweight Defense
Framework for Adversarially Robust Network Intrusion Detection.”**

PG-Def extracts **30 protocol-grounded features** directly from TCP/IP header
fields in a single streaming pass (Welford, O(1) memory per flow), classifies
each flow with a lightweight weighted soft-voting ensemble
(Random Forest + XGBoost + Logistic Regression), and augments detection with a
four-component adaptive defense layer. This repository contains the full
feature-extraction pipeline, the models, the adversarial-evaluation harness,
and the scripts that regenerate every table in the manuscript.

> **Reproducibility statement.** This code does **not** hard-code any number
> from the paper. Every reported table is *computed* from whatever data and
> hardware you run it on, then formatted into the manuscript’s exact layout.
> Numbers you obtain depend on your PCAPs, labelling, splits, and platform, and
> are expected to track — not exactly equal — the published figures (see
> *Expected deviations* below).

---

## Table of contents

1. [Repository layout](#repository-layout)
2. [Installation](#installation)
3. [Quick start](#quick-start)
4. [Step 1 — feature extraction from PCAPs](#step-1--feature-extraction-from-pcaps)
5. [Per-dataset preparation](#per-dataset-preparation)
6. [Step 2 — training and evaluation (Tables II–IX)](#step-2--training-and-evaluation-tables-iiix)
7. [The 30 protocol-grounded features](#the-30-protocol-grounded-features)
8. [Threat model and adversarial evaluation](#threat-model-and-adversarial-evaluation)
9. [Expected deviations and reproduction notes](#expected-deviations-and-reproduction-notes)
10. [Citation](#citation)

---

## Repository layout

```
pg-def/
├── pgdef/                     # importable package
│   ├── features/
│   │   ├── feature_spec.py    # canonical 30-feature definition (single source of truth)
│   │   ├── welford.py         # O(1) streaming mean/variance (n-1 denominator)
│   │   ├── flow_state.py      # per-flow 128-byte state computing all 30 features
│   │   └── pcap_extractor.py  # Tier 1+2: 5-tuple flow table, eviction, extraction
│   ├── data/loaders.py        # CSV loading, stratified split, train-only SMOTE, CV folds
│   ├── models/
│   │   ├── ensemble.py        # PG-Def weighted soft-vote ensemble (Tier 3)
│   │   ├── adaptive.py        # Tier 4 adaptive layer (Algorithm 2, components C1–C4)
│   │   └── baselines.py       # FA-CNN, GTAE-IDS, DAE, Adv. Retrain, DNN, Std. RF
│   ├── attacks/adversarial.py # 7 ART attacks + adaptive white-box PGD + SAAE
│   ├── eval/
│   │   ├── metrics.py         # TPR / FPR / precision / F1 / AUC, per-class TPR
│   │   ├── resources.py       # memory / latency / throughput profiling
│   │   └── tables.py          # LaTeX (IEEEtran + booktabs) generators for Tables II–IX
│   └── pipeline.py            # config loading, detector factory, memory accounting
├── scripts/                   # command-line entry points (run in order)
│   ├── 01_extract_features.py     # PCAP  -> 30-feature CSV  (+ optional label join)
│   ├── 02_train_evaluate.py       # Table II  (clean) + Table III (per-class)
│   ├── 03_adversarial_eval.py     # Table VIII (7 attacks + white-box PGD)
│   ├── 04_cross_domain.py         # Table VI  + Table VII (cross-domain transfer)
│   ├── 05_ablation.py             # Table V   (feature-category / group ablation)
│   ├── 06_resource_bench.py       # Table IV  (resource consumption)
│   └── 07_adaptive_eval.py        # Table IX  (adaptive defense components)
├── configs/default.yaml       # dataset paths, budget B, evaluation parameters
├── requirements.txt
├── pyproject.toml
├── Makefile
└── README.md
```

Mapping to the manuscript: Section IV → `pgdef/features/`; Section V-A/B (Tiers
1–3) → `pcap_extractor.py` + `ensemble.py`; Section V-C / Algorithm 2 (Tier 4)
→ `adaptive.py`; Section VI tables → `pgdef/eval/tables.py` driven by the
numbered scripts.

---

## Installation

Python 3.9+ is required. An editable install puts `pgdef` on the import path
so the scripts run from anywhere:

```bash
git clone <your-repo-url> pg-def
cd pg-def
python -m pip install -e .[adversarial]
```

`.[adversarial]` pulls in PyTorch and the Adversarial Robustness Toolbox, which
are only needed for scripts 03 and 07. On a CPU-only or constrained host you
can install the core set alone (`pip install -e .`) and run feature extraction
plus the clean / cross-domain / ablation / resource tables, then add the
adversarial extras later.

Alternatively, `pip install -r requirements.txt` and run scripts with
`PYTHONPATH=.` prefixed.

---

## Quick start

```bash
# 1. Extract features from each dataset's PCAPs (see Per-dataset preparation)
python scripts/01_extract_features.py \
    --pcap "pcaps/cicids2017/*.pcap" \
    --out  data/cicids2017_pgdef.csv \
    --label-csv labels/cicids2017_flows.csv

# 2. Regenerate every manuscript table into results/
make tables          # or run scripts 02–07 individually
```

Each script writes a `results/<tableN>.tex` file (IEEEtran + booktabs, ready to
`\input` into the paper) and prints the same table to the console. Scripts only
process the datasets whose feature CSV exists, so you can reproduce a subset.

---

## Step 1 — feature extraction from PCAPs

`scripts/01_extract_features.py` implements **Tier 1 (flow aggregation)** and
**Tier 2 (streaming feature extraction)**:

* packets are read with `dpkt` and grouped by the direction-independent
  5-tuple `⟨IPsrc, IPdst, Portsrc, Portdst, Proto⟩`, hashed with MurmurHash3;
* a per-flow 128-byte state updates all 30 features online via Welford’s
  algorithm (σ uses the `n−1` denominator, per Algorithm 1);
* flows are evicted on TCP FIN/RST or after a 120 s idle timeout, then emitted
  as a 30-dimensional vector.

```
python scripts/01_extract_features.py \
    --pcap "<glob to .pcap files>" \
    --out  data/<dataset>_pgdef.csv \
    [--label-csv <flow-label CSV>] \
    [--idle-timeout 120]
```

Without `--label-csv` the output CSV contains the 30 features only (useful for
inference). With it, each extracted flow is matched on its 5-tuple to a ground
-truth label and given a binary `label` column (0 = benign, 1 = malicious) plus
an optional `attack_category` column for the per-class table. The output column
order is fixed by `pgdef/features/feature_spec.py` — do not reorder it.

---

## Per-dataset preparation

All three datasets ship labels keyed differently, so labelling is the only
dataset-specific step. The general recipe: extract features from the PCAPs,
then provide a `--label-csv` whose rows carry the 5-tuple and a label.

**CICIDS2017.** Use the raw daily PCAPs. The accompanying
`GeneratedLabelledFlows` CSVs (CICFlowMeter output) already carry source/
destination IP and port, protocol, and an attack label — reduce them to a
5-tuple + label CSV for the join, and consolidate the original attack names
into the six functional categories listed in `configs/default.yaml`
(DDoS/DoS, Port Scan/Recon., Brute Force/Expl., Botnet/Backdoor,
Web Atk/Injection, Infiltration/Exf.).

**UNSW-NB15.** Use the source PCAPs with the official ground-truth event files
(`UNSW-NB15_GT.csv` / the four `UNSW-NB15_*.csv` record files), which provide
the 5-tuple, attack category, and label. Map `Normal` → 0 and any attack family
→ 1; keep the family name in `attack_category` if you want per-class numbers.

**Edge-IIoTset.** The released per-attack CSVs already contain flow identifiers
and an `Attack_type` / label column; build the label CSV directly from them and
join against the corresponding PCAP captures. `configs/default.yaml` leaves
`classes: []` for Edge-IIoTset (binary only) by default; populate it if you want
a per-class breakdown.

Point each `features_csv` in `configs/default.yaml` at the resulting file. The
config’s `attack_class_col` and `classes` control whether Table III is produced
for that dataset.

---

## Step 2 — training and evaluation (Tables II–IX)

Run after the feature CSVs exist. Order is independent except that all read the
same config; `make tables` runs them all.

| Script | Produces | Manuscript |
|--------|----------|------------|
| `02_train_evaluate.py` | Clean TPR/FPR/precision/F1/memory; per-class TPR | Tables II, III |
| `03_adversarial_eval.py` | TPR under FGSM, BIM, C&W-L2, C&W-L∞, DeepFool, JSMA, SAAE + adaptive white-box PGD | Table VIII |
| `04_cross_domain.py` | 3×3 train→test transfer matrix and degradation summary | Tables VI, VII |
| `05_ablation.py` | SAAE TPR by feature-category removal and feature-group config | Table V |
| `06_resource_bench.py` | RAM, latency, CPU, power, throughput per method | Table IV |
| `07_adaptive_eval.py` | Cache hit rate, borderline/reclassified rate, PI convergence, TPR gain | Table IX |

Common settings live under `configs/default.yaml`: `seed` (42), stratified
80:20 split, train-only SMOTE (`k=5`), 5-fold CV, the seven attack
hyperparameters, and the edge-device budget `B = (100 MB, 100 ms, 2 W)`.

Training follows the manuscript: RF (`T=50, D=10`), XGBoost
(`M=100, D=6, η=0.1, λ=1, γ=0.1`), LR (L2, `C=1.0`), with fixed voting weights
`wRF = 0.4, wXGB = 0.4, wLR = 0.2`. PG-Def’s reported memory of 27.8 MB is the
sum of the model footprint (≈15 MB) and the 12.8 MB flow-state table
(100,000 × 128 bytes); the adaptive layer adds ≤8 KB.

---

## The 30 protocol-grounded features

Defined canonically in `pgdef/features/feature_spec.py` and organised into the
five TCP/IP-layer categories of Table I:

* **Time Dynamics (φ1–φ8)** — IAT statistics, flow duration, active/idle bursts.
  `σIAT` (φ2) is the primary discriminator and the subject of Theorem 1.
* **Header Invariants (φ9–φ16)** — TTL/window statistics and TCP flag counts.
  `σTTL` (φ10) is the multi-OS botnet signature of Corollary 2.
* **Traffic Symmetry (φ17–φ20)** — forward/backward packet & byte ratios,
  size asymmetry, response rate. `Rpkt` (φ17) underlies the sequence-fabrication
  infeasibility argument.
* **Payload Dynamics (φ21–φ26)** — packet-length distribution from IP Total
  Length / TCP Data Offset (no deep packet inspection). φ21 is a normalisation
  denominator only.
* **Velocity (φ27–φ30)** — packet/byte rates and directional rates.

Feature groups used in the ablation: **Novel (15)**, **Novel Adversarial
Analysis (5)**, **Baseline (10)**. The critical subspace
`{σIAT, σTTL, Rpkt, Rbyte, σlen}` and minimal analysis set
`{σIAT, σTTL, Rpkt}` are exported from the same module.

---

## Threat model and adversarial evaluation

The manuscript’s ensemble is **non-differentiable** (tree models + LR), so
gradient attacks cannot be applied to it directly. Consistent with the
**black-box / transfer threat model of Section III**, scripts 03 and 07 train a
differentiable **transfer surrogate** of PG-Def and generate adversarial
examples on the surrogate, then evaluate them against the real ensemble. This
is a faithful, reviewer-safe instantiation of the stated adversary, not a
shortcut: a true white-box attacker has no gradient path through the deployed
classifier either.

* Off-manifold attacks are built with ART (FGSM ε=0.1; BIM ε=0.05, 10 it;
  C&W-L2 κ=0, 1000 steps; C&W-L∞; DeepFool 50 it; JSMA θ=0.1).
* The on-manifold **SAAE** attack learns the benign manifold with a small
  autoencoder and moves malicious flows a bounded step toward their benign
  reconstruction.
* The **adaptive white-box** upper bound uses PGD (100 it, α=0.01, ε=0.15, ℓ∞)
  over the joint ensemble objective.

Differentiable baselines (FA-CNN, GTAE-IDS, DAE, Adv. Retrain, DNN) expose
their own ART classifier and are attacked directly.

---

## Expected deviations and reproduction notes

* **No hard-coded numbers.** Tables are computed from your data and hardware.
  Minor differences from the published values are expected and arise from
  train/test splitting, SMOTE seeding, feature normalisation, ART/PyTorch
  versions, and CPU model.
* **Resource numbers are platform-specific.** Table IV reflects the machine you
  run `06_resource_bench.py` on. The manuscript reports an Intel i7-6500U and a
  Raspberry Pi 4B; your latency/throughput will differ accordingly while the
  *relative* ordering (PG-Def ≪ deep-learning baselines in memory and latency)
  is preserved.
* **Validation requires real PCAPs.** This package has been validated for
  correctness on synthetic and small captures; the full manuscript tables
  require the three public datasets. The pipeline is deterministic given a fixed
  seed and fixed inputs.
* **Streaming guarantee.** Feature extraction never stores packet history; per
  -flow state is exactly 128 bytes, matching the 12.8 MB flow-table footprint
  claimed in Section V-A.

---

## Citation

If you use this code, please cite the PG-Def paper:

```bibtex
@article{hasan_pgdef,
  title   = {PG-Def: A Protocol-Grounded Lightweight Defense Framework for
             Adversarially Robust Network Intrusion Detection},
  author  = {Hasan, Mehedi and others},
  note    = {Manuscript},
  year    = {2026}
}
```

Released under the MIT License (see `LICENSE`).
