"""Tier 4: four-component adaptive defense system (Section V-C / Algorithm 2).

  C1  Attack Fingerprint Cache    -- Bloom filter (m = 1024 bits, k = 3 hashes)
                                     over quantised <floor(sigma_IAT*10),
                                     floor(sigma_TTL), floor(R_pkt)>.
  C2  Ensemble Confidence Monitor -- borderline flows (c_ens < tau_conf) are
                                     re-scored on the critical subspace
                                     {sigma_IAT, sigma_TTL, R_pkt, R_byte,
                                     sigma_len}.
  C3  Dynamic Threshold Adaptation-- proportional-integral controller holding
                                     FPR <= FPR_target = 0.02
                                     (alpha_p = 0.01, alpha_i = 0.001).
  C4  Protocol Feature Drift Monitor -- two-level Welford (short window
                                     W_s = 10 000, long window W_l = 100 000).

Total added cost: <= 8 KB memory and < 0.05 ms latency per flow.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import mmh3
import numpy as np

from ..features.feature_spec import CRITICAL_SUBSPACE, FEATURE_NAMES, MINIMAL_SET
from ..features.welford import Welford
from .ensemble import PGDefEnsemble


class BloomFilter:
    """Fixed-size Bloom filter (Component 1)."""

    def __init__(self, m_bits: int = 1024, k: int = 3) -> None:
        self.m = m_bits
        self.k = k
        self.bits = np.zeros(m_bits, dtype=bool)

    def _positions(self, item: str) -> List[int]:
        return [mmh3.hash(item, seed, signed=False) % self.m
                for seed in range(self.k)]

    def add(self, item: str) -> None:
        for p in self._positions(item):
            self.bits[p] = True

    def query(self, item: str) -> bool:
        return all(self.bits[p] for p in self._positions(item))


def _fingerprint(row: Dict[str, float]) -> str:
    """Quantised critical-feature fingerprint <floor(sigma_IAT*10),
    floor(sigma_TTL), floor(R_pkt)>."""
    return "{}|{}|{}".format(
        int(np.floor(row["std_iat"] * 10)),
        int(np.floor(row["std_ttl"])),
        int(np.floor(row["pkt_ratio"])),
    )


@dataclass
class AdaptiveDefense:
    """PG-Def + adaptive layer (PG-Def+Adap.)."""

    base: PGDefEnsemble
    tau_conf: float = 0.75
    tau_vote: float = 0.50
    fpr_target: float = 0.02
    alpha_p: float = 0.01
    alpha_i: float = 0.001
    drift_delta: float = 0.15
    win_short: int = 10_000
    win_long: int = 100_000
    seed: int = 42

    # secondary classifier on the critical subspace (Component 2)
    critical: PGDefEnsemble = field(init=False, default=None)
    _crit_idx: List[int] = field(init=False, default_factory=list)
    _integral: float = field(init=False, default=0.0)
    _short: Dict[str, Welford] = field(init=False, default_factory=dict)
    _long: Dict[str, Welford] = field(init=False, default_factory=dict)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaptiveDefense":
        self._crit_idx = [FEATURE_NAMES.index(c) for c in CRITICAL_SUBSPACE]
        self.critical = PGDefEnsemble(seed=self.seed).fit(X[:, self._crit_idx], y)
        self._short = {c: Welford() for c in FEATURE_NAMES}
        self._long = {c: Welford() for c in FEATURE_NAMES}
        return self

    # ---- Component 3: proportional-integral threshold adaptation -------------
    def _update_thresholds(self, fpr: float) -> None:
        e = fpr - self.fpr_target
        self._integral += e
        self.tau_vote = float(np.clip(
            self.tau_vote + self.alpha_p * e + self.alpha_i * self._integral,
            0.05, 0.95))
        self.tau_conf = float(np.clip(
            self.tau_conf + 0.5 * self.alpha_p * e, 0.50, 0.95))

    def calibrate(self, X_val: np.ndarray, y_val: np.ndarray,
                  rounds: int = 1000) -> int:
        """Run the PI controller on validation data; return convergence step.

        Convergence = first flow index at which the running FPR stays within
        +/-0.2% of FPR_target for 50 consecutive updates.
        """
        scores = self.base.decision_scores(X_val)
        benign = np.where(y_val == 0)[0]
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(benign)[:rounds]
        stable = 0
        conv = len(order)
        for step, i in enumerate(order, 1):
            fp = float(scores[i] >= self.tau_vote)  # 1 if false positive
            running = (running * (step - 1) + fp) / step if step > 1 else fp
            self._update_thresholds(running)
            if abs(running - self.fpr_target) <= 0.002:
                stable += 1
                if stable >= 50:
                    conv = step
                    break
            else:
                stable = 0
        return conv

    # ---- Component 4: drift monitoring --------------------------------------
    def observe_drift(self, X: np.ndarray) -> List[str]:
        """Update two-level Welford accumulators; return drifting features."""
        alerts = []
        for j, name in enumerate(FEATURE_NAMES):
            col = X[:, j]
            for v in col[-self.win_short:]:
                self._short[name].update(float(v))
            for v in col[-self.win_long:]:
                self._long[name].update(float(v))
            s = self._long[name].sample_std()
            if s > 0:
                shift = abs(self._short[name].mean - self._long[name].mean) / s
                if shift > self.drift_delta:
                    alerts.append(name)
        return alerts

    # ---- Components 1+2: per-flow decision (Algorithm 2) --------------------
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Adaptive prediction with cache, confidence routing and bloom insert.

        Returns ``(y_hat, stats)`` where ``stats`` carries cache-hit rate and
        the borderline / reclassified fractions reported in Table IX.
        """
        bloom = BloomFilter()
        scores = self.base.decision_scores(X)
        conf = self.base.confidence(X)
        crit_scores = self.critical.decision_scores(X[:, self._crit_idx])

        y_hat = np.zeros(X.shape[0], dtype=int)
        hits = borderline = reclassified = 0
        for i in range(X.shape[0]):
            row = {c: X[i, FEATURE_NAMES.index(c)] for c in MINIMAL_SET}
            fp = _fingerprint(row)
            # C1: fingerprint cache
            if bloom.query(fp):
                y_hat[i] = 1
                hits += 1
                continue
            # C2: confidence monitor
            c = conf[i]
            if scores[i] >= self.tau_conf:
                y_hat[i] = 1
                bloom.add(fp)
            elif scores[i] <= (1.0 - self.tau_conf):
                y_hat[i] = 0
            else:
                borderline += 1
                pred = int(crit_scores[i] >= self.tau_vote)
                y_hat[i] = pred
                if pred == 1:
                    reclassified += 1
        n = max(X.shape[0], 1)
        stats = {
            "cache_hit_rate": hits / n,
            "borderline_rate": borderline / n,
            "reclassified_rate": reclassified / max(borderline, 1),
            "tau_vote": self.tau_vote,
            "tau_conf": self.tau_conf,
        }
        return y_hat, stats
