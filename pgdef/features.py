"""Protocol-grounded feature definitions and the manipulation-cost partition.

The 30 features are organised into five TCP/IP categories. Each feature is
classified as PROTECTED (P) or MANIPULABLE (M) by the cost an adversary incurs to
drive it toward benign values *without changing the attack's behaviour*. PG-Def
trains its detector on the PROTECTED subset only.

Column-name convention (as produced by the extractor): ``phi{N}_{name}``.
"""
from __future__ import annotations

# (phi_id, canonical_name, category, class)  -- class in {"P","M"}
FEATURES = [
    ("phi1",  "duration",       "time",     "P"),
    ("phi2",  "iat_mean",       "time",     "P"),
    ("phi3",  "iat_std",        "time",     "P"),   # sigma_IAT (critical)
    ("phi4",  "iat_min",        "time",     "P"),
    ("phi5",  "iat_max",        "time",     "P"),
    ("phi6",  "active_mean",    "time",     "P"),
    ("phi7",  "idle_mean",      "time",     "P"),
    ("phi8",  "fwd_iat_mean",   "time",     "P"),
    ("phi9",  "ttl_mean",       "header",   "M"),
    ("phi10", "ttl_std",        "header",   "P"),   # sigma_TTL (critical)
    ("phi11", "ttl_min",        "header",   "M"),
    ("phi12", "ttl_max",        "header",   "M"),
    ("phi13", "win_mean",       "header",   "M"),
    ("phi14", "win_std",        "header",   "M"),
    ("phi15", "syn_count",      "header",   "P"),
    ("phi16", "hdr_mean",       "header",   "M"),
    ("phi17", "fwd_pkts",       "symmetry", "P"),
    ("phi18", "bwd_pkts",       "symmetry", "P"),
    ("phi19", "pkt_ratio",      "symmetry", "P"),   # R_pkt (critical)
    ("phi20", "byte_ratio",     "symmetry", "P"),
    ("phi21", "pl_mean",        "payload",  "M"),
    ("phi22", "pl_std",         "payload",  "P"),
    ("phi23", "pl_min",         "payload",  "P"),
    ("phi24", "pl_max",         "payload",  "M"),
    ("phi25", "fwd_pl_mean",    "payload",  "M"),
    ("phi26", "bwd_pl_mean",    "payload",  "M"),
    ("phi27", "pkts_per_s",     "velocity", "P"),
    ("phi28", "bytes_per_s",    "velocity", "P"),
    ("phi29", "fwd_pkts_per_s", "velocity", "P"),
    ("phi30", "bwd_pkts_per_s", "velocity", "P"),
]

COLUMN_NAMES = [f"{pid}_{name}" for pid, name, _, _ in FEATURES]
PROTECTED   = [f"{pid}_{name}" for pid, name, _, c in FEATURES if c == "P"]   # 20
MANIPULABLE = [f"{pid}_{name}" for pid, name, _, c in FEATURES if c == "M"]   # 10
CRITICAL    = ["phi3_iat_std", "phi10_ttl_std", "phi19_pkt_ratio"]

COMPACT5 = {
    "cicids": ["phi22_pl_std", "phi10_ttl_std", "phi23_pl_min",
               "phi20_byte_ratio", "phi15_syn_count"],
    "unsw":   ["phi20_byte_ratio", "phi28_bytes_per_s", "phi10_ttl_std",
               "phi22_pl_std", "phi1_duration"],
}

def resolve(columns, wanted):
    """Map desired feature names to actual columns, tolerating minor naming
    differences (match by phi-id prefix, then by canonical name)."""
    cols = list(columns); out = []
    for w in wanted:
        if w in cols:
            out.append(w); continue
        pid = w.split("_")[0]
        cand = [c for c in cols if c.split("_")[0] == pid]
        if not cand:
            name = "_".join(w.split("_")[1:])
            cand = [c for c in cols if name and name in c]
        if not cand:
            raise KeyError(f"feature '{w}' not found in columns")
        out.append(cand[0])
    return out
