"""Welford's online algorithm for single-pass mean/variance (Principle 3).

Reference: B. P. Welford, "Note on a method for calculating corrected sums of
squares and products," Technometrics, 4(3):419-420, 1962.

Each accumulator keeps only (count, mean, M2), i.e. O(1) state per stream,
and is updated exactly once per observation -- no packet history is stored.
"""

from __future__ import annotations

import math


class Welford:
    """Running mean and (sum of squared deviations) M2 in constant memory."""

    __slots__ = ("n", "mean", "m2")

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def sample_std(self) -> float:
        """Sample standard deviation with (n - 1) denominator.

        Matches Algorithm 1 line 15 of the manuscript, where the IAT std uses
        an (n - 1) denominator (n = packet count, so n - 1 = number of IATs).
        Returns 0.0 when fewer than two observations are available.
        """
        if self.n < 2:
            return 0.0
        return math.sqrt(self.m2 / (self.n - 1))
