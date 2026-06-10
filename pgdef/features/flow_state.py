"""Per-flow streaming state (Tier 2).

A ``FlowState`` object incrementally accumulates the statistics required for
all 30 protocol-grounded features (Table I) using Welford estimators and a
small set of counters. Updates are O(1) per packet and no packet history is
retained, satisfying Principle 3 (Streaming computability).

State-size note (Section IV-A / Tier 2): the manuscript specifies an exact
128-byte per-flow record packing ~10 double accumulators and ~10 uint32
counters (12.8 MB total flow table at 100 000 concurrent flows). Python object
overhead prevents enforcing 128 bytes at runtime, but the accumulator *layout*
below is the 128-byte design; ``STATE_BYTES`` documents the intended footprint
used for the memory budget in Section VI-D.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from .feature_spec import FEATURE_NAMES
from .welford import Welford

# Intended on-device per-flow record size (Tier 2): 10 doubles + 10 uint32.
STATE_BYTES = 10 * 8 + 10 * 4 + 8  # 128 bytes

SMALL_PKT = 64       # phi_24 threshold (bytes)
LARGE_PKT = 1200     # phi_25 threshold (bytes)
RATIO_CAP = 1.0e6    # cap for ratios when a denominator is zero

FlowKey = Tuple[str, str, int, int, int]  # (ip_src, ip_dst, port_src, port_dst, proto)


class FlowState:
    """Accumulates 30-feature statistics for one bidirectional 5-tuple flow."""

    __slots__ = (
        "key", "n", "t_first", "t_last",
        "iat", "fwd_iat", "iat_min", "iat_max", "t_fwd_last",
        "active", "idle", "active_start", "burst_tau",
        "ttl", "win",
        "n_syn", "n_urg", "n_fin",
        "hdr", "length",
        "n_fwd", "n_bwd", "b_fwd", "b_bwd",
        "n_small", "n_large", "sum_hdr", "sum_pay",
    )

    def __init__(self, key: FlowKey, ts: float, burst_tau: float = 0.1) -> None:
        self.key = key
        self.n = 0
        self.t_first = ts
        self.t_last = ts
        self.t_fwd_last: Optional[float] = None
        # Time dynamics
        self.iat = Welford()
        self.fwd_iat = Welford()
        self.iat_min = math.inf
        self.iat_max = 0.0
        # Active/idle segmentation (tau_burst = 0.1 s by default)
        self.burst_tau = burst_tau
        self.active = Welford()
        self.idle = Welford()
        self.active_start = ts
        # Header invariants
        self.ttl = Welford()
        self.win = Welford()
        self.n_syn = 0
        self.n_urg = 0
        self.n_fin = 0
        self.hdr = Welford()
        self.length = Welford()
        # Direction / symmetry / velocity
        self.n_fwd = 0
        self.n_bwd = 0
        self.b_fwd = 0
        self.b_bwd = 0
        self.n_small = 0
        self.n_large = 0
        self.sum_hdr = 0
        self.sum_pay = 0

    def update(self, ts: float, length: int, ttl: int, win: int,
               hdr_len: int, is_fwd: bool,
               syn: bool, urg: bool, fin: bool) -> None:
        """Incremental O(1) update for a single packet of this flow."""
        self.n += 1

        # --- Time dynamics (phi_1 - phi_8) ---
        if self.n > 1:
            gap = ts - self.t_last
            if gap < 0:
                gap = 0.0
            self.iat.update(gap)
            self.iat_min = min(self.iat_min, gap)
            self.iat_max = max(self.iat_max, gap)
            # Active / idle segmentation
            if gap >= self.burst_tau:
                self.active.update(self.t_last - self.active_start)
                self.idle.update(gap)
                self.active_start = ts
        self.t_last = ts

        # --- Header invariants (phi_9 - phi_16) ---
        self.ttl.update(float(ttl))
        self.win.update(float(win))
        if syn:
            self.n_syn += 1
        if urg:
            self.n_urg += 1
        if fin:
            self.n_fin += 1
        self.hdr.update(float(hdr_len))

        # --- Payload dynamics (phi_21 - phi_26) ---
        self.length.update(float(length))
        if length < SMALL_PKT:
            self.n_small += 1
        if length > LARGE_PKT:
            self.n_large += 1
        payload = max(length - hdr_len, 0)
        self.sum_hdr += hdr_len
        self.sum_pay += payload

        # --- Direction / symmetry / velocity (phi_17 - phi_20, phi_27 - phi_30) ---
        if is_fwd:
            self.n_fwd += 1
            self.b_fwd += length
            if self.t_fwd_last is not None:
                fgap = ts - self.t_fwd_last
                self.fwd_iat.update(fgap if fgap > 0 else 0.0)
            self.t_fwd_last = ts
        else:
            self.n_bwd += 1
            self.b_bwd += length

    @staticmethod
    def _ratio(num: float, den: float) -> float:
        if den <= 0:
            return RATIO_CAP if num > 0 else 0.0
        return num / den

    def emit(self) -> Dict[str, float]:
        """Finalise and return the 30-feature dictionary for this flow."""
        duration = max(self.t_last - self.t_first, 0.0)
        mu_len = self.length.mean
        std_len = self.length.sample_std()
        f: Dict[str, float] = {
            # Time dynamics
            "mean_iat":      self.iat.mean,
            "std_iat":       self.iat.sample_std(),
            "min_iat":       0.0 if self.iat_min is math.inf else self.iat_min,
            "max_iat":       self.iat_max,
            "flow_duration": duration,
            "mean_active":   self.active.mean,
            "mean_idle":     self.idle.mean,
            "fwd_mean_iat":  self.fwd_iat.mean,
            # Header invariants
            "mean_ttl":      self.ttl.mean,
            "std_ttl":       self.ttl.sample_std(),
            "mean_win":      self.win.mean,
            "std_win":       self.win.sample_std(),
            "syn_count":     float(self.n_syn),
            "urg_count":     float(self.n_urg),
            "fin_ratio":     self.n_fin / self.n if self.n else 0.0,
            "mean_hdr_len":  self.hdr.mean,
            # Traffic symmetry
            "pkt_ratio":     self._ratio(self.n_fwd, self.n_bwd),
            "byte_ratio":    self._ratio(self.b_fwd, self.b_bwd),
            "size_asymmetry": self._ratio(self.b_fwd - self.b_bwd, self.b_fwd + self.b_bwd),
            "response_rate": self._ratio(self.n_bwd, duration),
            # Payload dynamics
            "mean_len":      mu_len,
            "std_len":       std_len,
            "cv_len":        std_len / mu_len if mu_len > 0 else 0.0,
            "small_pkt_ratio": self.n_small / self.n if self.n else 0.0,
            "large_pkt_ratio": self.n_large / self.n if self.n else 0.0,
            "hdr_pay_ratio": self._ratio(self.sum_hdr, self.sum_pay),
            # Velocity
            "pkt_rate":      self._ratio(self.n, duration),
            "byte_rate":     self._ratio(self.b_fwd + self.b_bwd, duration),
            "fwd_byte_rate": self._ratio(self.b_fwd, duration),
            "bwd_pkt_rate":  self._ratio(self.n_bwd, duration),
        }
        return f

    def emit_vector(self) -> np.ndarray:
        """30-feature vector ordered as ``FEATURE_NAMES``."""
        f = self.emit()
        return np.array([f[name] for name in FEATURE_NAMES], dtype=np.float64)
