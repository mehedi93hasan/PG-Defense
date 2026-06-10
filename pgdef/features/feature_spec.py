"""Canonical specification of the 30 protocol-grounded features (Table I).

This module is the single source of truth for feature ordering, symbols,
TCP/IP-layer category, robustness rating, and feature-group membership used
throughout PG-Def. Every other module imports ``FEATURES`` / ``FEATURE_NAMES``
to guarantee a consistent 30-dimensional vector layout F in R^30.

Robustness ratings and groups follow Table I of the manuscript:

  Group  N  = Novel                       (15 features)
  Group  NA = Novel Adversarial Analysis  (5  features)
  Group  B  = Baseline                    (10 features)

  Robustness:
    Critical  -- adversarial normalisation is provably infeasible
    High      -- manipulation causes severe attack degradation
    Medium    -- manipulation imposes moderate operational cost
    Baseline  -- serves normalisation role only
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Feature:
    idx: int          # phi index (1..30)
    key: str          # machine-readable column name
    symbol: str       # mathematical symbol used in the manuscript
    category: str     # one of the five TCP/IP-layer categories
    robustness: str   # Critical | High | Medium | Baseline
    group: str        # N | NA | B


# Ordered exactly as Table I (phi_1 .. phi_30).
FEATURES: List[Feature] = [
    # Category 1: Time Dynamics (phi_1 - phi_8)
    Feature(1,  "mean_iat",        "mu_IAT",      "Time Dynamics",     "High",     "NA"),
    Feature(2,  "std_iat",         "sigma_IAT",   "Time Dynamics",     "Critical", "N"),
    Feature(3,  "min_iat",         "IAT_min",     "Time Dynamics",     "High",     "NA"),
    Feature(4,  "max_iat",         "IAT_max",     "Time Dynamics",     "Medium",   "B"),
    Feature(5,  "flow_duration",   "T_flow",      "Time Dynamics",     "Medium",   "B"),
    Feature(6,  "mean_active",     "mu_Tactive",  "Time Dynamics",     "Medium",   "N"),
    Feature(7,  "mean_idle",       "mu_Tidle",    "Time Dynamics",     "Medium",   "N"),
    Feature(8,  "fwd_mean_iat",    "mu_fwd_IAT",  "Time Dynamics",     "High",     "N"),
    # Category 2: Header Invariants (phi_9 - phi_16)
    Feature(9,  "mean_ttl",        "mu_TTL",      "Header Invariants", "High",     "NA"),
    Feature(10, "std_ttl",         "sigma_TTL",   "Header Invariants", "Critical", "N"),
    Feature(11, "mean_win",        "mu_win",      "Header Invariants", "High",     "B"),
    Feature(12, "std_win",         "sigma_win",   "Header Invariants", "Medium",   "N"),
    Feature(13, "syn_count",       "N_SYN",       "Header Invariants", "High",     "NA"),
    Feature(14, "urg_count",       "N_URG",       "Header Invariants", "Medium",   "B"),
    Feature(15, "fin_ratio",       "R_FIN",       "Header Invariants", "Medium",   "N"),
    Feature(16, "mean_hdr_len",    "mu_hd",       "Header Invariants", "Medium",   "N"),
    # Category 3: Traffic Symmetry (phi_17 - phi_20)
    Feature(17, "pkt_ratio",       "R_pkt",       "Traffic Symmetry",  "Critical", "B"),
    Feature(18, "byte_ratio",      "R_byte",      "Traffic Symmetry",  "Critical", "N"),
    Feature(19, "size_asymmetry",  "A_size",      "Traffic Symmetry",  "High",     "N"),
    Feature(20, "response_rate",   "lambda_resp", "Traffic Symmetry",  "Medium",   "N"),
    # Category 4: Payload Dynamics (phi_21 - phi_26)
    Feature(21, "mean_len",        "mu_len",      "Payload Dynamics",  "Baseline", "B"),
    Feature(22, "std_len",         "sigma_len",   "Payload Dynamics",  "Critical", "NA"),
    Feature(23, "cv_len",          "CV_len",      "Payload Dynamics",  "High",     "N"),
    Feature(24, "small_pkt_ratio", "R_small",     "Payload Dynamics",  "Medium",   "N"),
    Feature(25, "large_pkt_ratio", "R_large",     "Payload Dynamics",  "Medium",   "N"),
    Feature(26, "hdr_pay_ratio",   "R_hd_pay",    "Payload Dynamics",  "High",     "N"),
    # Category 5: Velocity (phi_27 - phi_30)
    Feature(27, "pkt_rate",        "lambda_pkt",  "Velocity",          "High",     "B"),
    Feature(28, "byte_rate",       "lambda_byte", "Velocity",          "High",     "B"),
    Feature(29, "fwd_byte_rate",   "lambda_fwd",  "Velocity",          "High",     "B"),
    Feature(30, "bwd_pkt_rate",    "lambda_bwd",  "Velocity",          "Medium",   "B"),
]

assert len(FEATURES) == 30, "exactly 30 protocol-grounded features expected"

FEATURE_NAMES: List[str] = [f.key for f in FEATURES]

CATEGORIES = [
    "Time Dynamics",
    "Header Invariants",
    "Traffic Symmetry",
    "Payload Dynamics",
    "Velocity",
]

# Feature-group membership (used by the ablation study, Section VI-E / Table V).
GROUP_N = [f.key for f in FEATURES if f.group == "N"]    # 15 Novel
GROUP_NA = [f.key for f in FEATURES if f.group == "NA"]  # 5  Novel Adversarial Analysis
GROUP_B = [f.key for f in FEATURES if f.group == "B"]    # 10 Baseline

# Critical subspace used by the adaptive defense (Algorithm 2, Component 2).
CRITICAL_SUBSPACE = ["std_iat", "std_ttl", "pkt_ratio", "byte_ratio", "std_len"]

# Minimal sufficient set for the formal analysis (Section IV-C, Theorem 1).
MINIMAL_SET = ["std_iat", "std_ttl", "pkt_ratio"]


def category_indices(category: str) -> List[int]:
    """0-based column indices of the features in a given category."""
    return [i for i, f in enumerate(FEATURES) if f.category == category]


def feature_index(key: str) -> int:
    """0-based column index of a feature by its key."""
    return FEATURE_NAMES.index(key)
