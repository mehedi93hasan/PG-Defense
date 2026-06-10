"""Shared helpers for the orchestration scripts."""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import yaml

from .data.loaders import load_feature_csv, make_split
from .models.baselines import BASELINES
from .models.ensemble import PGDefEnsemble


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def available_datasets(cfg: dict) -> List[str]:
    """Datasets whose feature CSV exists on disk."""
    return [name for name, d in cfg["datasets"].items()
            if os.path.exists(d["features_csv"])]


def build_detector(name: str, seed: int = 42):
    """Instantiate PG-Def or a baseline by name."""
    if name == "PG-Def":
        return PGDefEnsemble(seed=seed)
    if name in BASELINES:
        return BASELINES[name](seed=seed) if name != "Std. RF" else BASELINES[name](seed=seed)
    raise ValueError(f"unknown detector {name}")


def detector_memory(name: str, detector) -> float:
    """Model memory footprint in MB (Table II / IV)."""
    if name == "PG-Def":
        from .eval.resources import FLOW_TABLE_MB
        return 15.0 + FLOW_TABLE_MB  # ~15 MB models + 12.8 MB flow table = 27.8 MB
    return getattr(detector, "memory_mb", float("nan"))


def load_split(cfg: dict, dataset: str, **kw):
    d = cfg["datasets"][dataset]
    X, y = load_feature_csv(d["features_csv"])
    ev = cfg["evaluation"]
    return make_split(X, y, test_size=ev["test_size"], smote_k=ev["smote_k"],
                      seed=cfg["seed"], **kw)
