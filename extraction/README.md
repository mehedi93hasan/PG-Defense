# Feature extraction (pcap → CSV)

`pgdef_extractor.py` turns a raw `.pcap` into the 30 protocol-grounded features
(one row per flow), matching `pgdef/features.py`. It is a single-pass, O(1)-per-flow
streaming extractor (Welford accumulators), mirroring Tier-2 of the architecture.

```bash
pip install dpkt
python extraction/pgdef_extractor.py capture.pcap flows.csv
```

The extractor outputs features only. **Labelling** (benign vs attack, and the
`class` column used by the per-attack experiment) is dataset-specific and must be
applied from each dataset's ground-truth (flow 5-tuple + time window for
CICIDS2017; the provided label files for UNSW-NB15 and Edge-IIoTset). Add a
`label` column (0/1) and optional `class` column before running the experiments.

> Note: this module requires the raw captures, which are large and not included.
> The experiment and figure scripts operate directly on the resulting feature
> CSVs, so the paper's tables can be reproduced without re-running extraction if
> you already have the feature CSVs.
