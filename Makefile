# PG-Def reproduction pipeline.
#
# Feature CSVs (data/*_pgdef.csv) must exist first — produced either by
# `make features` from PCAPs or supplied directly. Then `make tables`
# regenerates every manuscript table into results/.

PY      ?= python
CONFIG  ?= configs/default.yaml

.PHONY: help install features tables clean all
help:
	@echo "make install   - editable install of the pgdef package + deps"
	@echo "make features  - extract 30 protocol-grounded features from PCAPs (edit paths below)"
	@echo "make tables     - run evaluation scripts 02-07 -> results/*.tex (Tables II-IX)"
	@echo "make all        - features + tables"
	@echo "make clean      - remove results/ and __pycache__"

install:
	$(PY) -m pip install -e .[adversarial]

# Example feature-extraction targets. Replace the globs / label CSVs with your
# local dataset paths (see README, 'Per-dataset preparation').
features:
	$(PY) scripts/01_extract_features.py --pcap "pcaps/cicids2017/*.pcap" \
		--out data/cicids2017_pgdef.csv --label-csv labels/cicids2017_flows.csv
	$(PY) scripts/01_extract_features.py --pcap "pcaps/unsw_nb15/*.pcap" \
		--out data/unsw_nb15_pgdef.csv --label-csv labels/unsw_nb15_flows.csv
	$(PY) scripts/01_extract_features.py --pcap "pcaps/edge_iiotset/*.pcap" \
		--out data/edge_iiotset_pgdef.csv --label-csv labels/edge_iiotset_flows.csv

tables:
	$(PY) scripts/02_train_evaluate.py  --config $(CONFIG)   # Tables II, III
	$(PY) scripts/03_adversarial_eval.py --config $(CONFIG)  # Table VIII
	$(PY) scripts/04_cross_domain.py    --config $(CONFIG)   # Tables VI, VII
	$(PY) scripts/05_ablation.py        --config $(CONFIG)   # Table V
	$(PY) scripts/06_resource_bench.py  --config $(CONFIG)   # Table IV
	$(PY) scripts/07_adaptive_eval.py   --config $(CONFIG)   # Table IX

all: features tables

clean:
	rm -rf results/*.tex
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
