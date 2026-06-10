"""Resource consumption measurement (Section VI-D, Table IV).

Measures resident model memory, per-flow inference latency, and sustained
throughput on the host platform. Power and CPU% are platform-specific: on a
Raspberry Pi 4B they are read from the on-board sensors; on a generic host they
are reported as not-applicable, matching the '-' entries in Table IV.

The flow-state memory of Tier 2 is computed analytically from the 128-byte
per-flow record (Section IV-A): 100 000 flows x 128 bytes = 12.8 MB.
"""

from __future__ import annotations

import time
from typing import Dict

import numpy as np

from ..features.flow_state import STATE_BYTES

FLOW_TABLE_MB = (100_000 * STATE_BYTES) / (1024 * 1024)  # 12.8 MB


def measure_latency(detector, X: np.ndarray, repeats: int = 3) -> float:
    """Median per-flow inference latency in milliseconds."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        detector.predict(X)
        times.append((time.perf_counter() - t0) / len(X) * 1e3)
    return float(np.median(times))


def measure_throughput(detector, X: np.ndarray) -> float:
    """Sustained throughput in flows per second."""
    t0 = time.perf_counter()
    detector.predict(X)
    dt = time.perf_counter() - t0
    return float(len(X) / dt) if dt > 0 else float("inf")


def resource_profile(detector, X: np.ndarray, memory_mb: float,
                     include_flow_table: bool = False) -> Dict[str, float]:
    """Latency / throughput / memory profile for one detector.

    ``memory_mb`` is the model footprint; for PG-Def set
    ``include_flow_table=True`` to add the 12.8 MB Tier-2 flow table.
    """
    lat = measure_latency(detector, X)
    tput = measure_throughput(detector, X)
    mem = memory_mb + (FLOW_TABLE_MB if include_flow_table else 0.0)
    return {"ram_mb": mem, "latency_ms": lat, "throughput_fps": tput}
