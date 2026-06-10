"""PCAP -> 30 protocol-grounded features (Tier 1 + Tier 2).

Tier 1 (Flow Aggregation): packets are grouped by the canonical bidirectional
5-tuple flow key K = <IP_src, IP_dst, Port_src, Port_dst, Proto>. The flow
table is indexed by a MurmurHash3 digest of K (non-cryptographic, O(1) amortised
lookup). Flows are evicted on TCP teardown (FIN or RST) or after an idle timeout
of 120 s, matching Section V-A.

Tier 2 (Stream Feature Extraction): each active flow maintains a ``FlowState``
that is updated once per packet via Welford's algorithm (Algorithm 1). On
eviction the 30-feature vector is emitted.

Only standardised TCP/IP header fields are read (Ethernet L2, IPv4/IPv6 L3,
TCP/UDP L4) -- no deep packet inspection -- so every feature is protocol-grounded
(Principle 1).

Requires ``dpkt`` for packet parsing.
"""

from __future__ import annotations

import socket
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import dpkt
import mmh3
import numpy as np
import pandas as pd

from .feature_spec import FEATURE_NAMES
from .flow_state import FlowKey, FlowState

IDLE_TIMEOUT = 120.0   # seconds (Section V-A)
BURST_TAU = 0.1        # seconds (tau_burst, active/idle segmentation)
MAX_FLOWS = 100_000    # concurrent flow-table capacity (Tier 1)

# Optional labeller: maps a flow key + first-packet timestamp to a label.
Labeller = Callable[[FlowKey, float], Optional[int]]


def _canonical_key(ip_src: str, ip_dst: str, sport: int, dport: int,
                   proto: int) -> Tuple[FlowKey, bool]:
    """Return a direction-independent flow key and whether the packet is forward.

    The forward direction is fixed by the lexicographically smaller endpoint so
    that both directions of a connection hash to the same table slot.
    """
    a = (ip_src, sport)
    b = (ip_dst, dport)
    if a <= b:
        return (ip_src, ip_dst, sport, dport, proto), True
    return (ip_dst, ip_src, dport, sport, proto), False


def _hash_key(key: FlowKey) -> int:
    """MurmurHash3 digest of the 5-tuple (Tier 1 flow-table index)."""
    return mmh3.hash("|".join(map(str, key)), signed=False)


class PcapExtractor:
    """Streaming extractor producing one 30-feature row per completed flow."""

    def __init__(self, idle_timeout: float = IDLE_TIMEOUT,
                 burst_tau: float = BURST_TAU,
                 labeller: Optional[Labeller] = None) -> None:
        self.idle_timeout = idle_timeout
        self.burst_tau = burst_tau
        self.labeller = labeller
        self.table: Dict[int, FlowState] = {}      # MurmurHash3(key) -> state
        self.first_ts: Dict[int, float] = {}        # for labelling

    def _evict_idle(self, now: float, out: List[Dict[str, float]]) -> None:
        stale = [h for h, s in self.table.items()
                 if (now - s.t_last) > self.idle_timeout]
        for h in stale:
            self._finalise(h, out)

    def _finalise(self, h: int, out: List[Dict[str, float]]) -> None:
        state = self.table.pop(h)
        row = state.emit()
        if self.labeller is not None:
            lbl = self.labeller(state.key, self.first_ts.get(h, state.t_first))
            row["label"] = -1 if lbl is None else int(lbl)
        # carry the 5-tuple for downstream label joins
        (row["ip_src"], row["ip_dst"],
         row["port_src"], row["port_dst"], row["proto"]) = state.key
        out.append(row)
        self.first_ts.pop(h, None)

    def process(self, pcap_path: str) -> Iterator[Dict[str, float]]:
        """Yield one feature dict per completed flow in ``pcap_path``."""
        with open(pcap_path, "rb") as fh:
            try:
                reader = dpkt.pcap.Reader(fh)
            except ValueError:
                fh.seek(0)
                reader = dpkt.pcapng.Reader(fh)

            for ts, buf in reader:
                out: List[Dict[str, float]] = []
                self._handle_packet(ts, buf, out)
                # opportunistic idle eviction keeps the table bounded
                if len(self.table) > MAX_FLOWS:
                    self._evict_idle(ts, out)
                yield from out

        # flush all remaining open flows at end of capture
        tail: List[Dict[str, float]] = []
        for h in list(self.table.keys()):
            self._finalise(h, tail)
        yield from tail

    def _handle_packet(self, ts: float, buf: bytes,
                       out: List[Dict[str, float]]) -> None:
        try:
            eth = dpkt.ethernet.Ethernet(buf)
        except dpkt.dpkt.NeedData:
            return
        ip = eth.data
        if isinstance(ip, dpkt.ip.IP):
            ip_src = socket.inet_ntop(socket.AF_INET, ip.src)
            ip_dst = socket.inet_ntop(socket.AF_INET, ip.dst)
            ttl = ip.ttl
            ip_hdr_len = ip.hl * 4
            total_len = ip.len
        elif isinstance(ip, dpkt.ip6.IP6):
            ip_src = socket.inet_ntop(socket.AF_INET6, ip.src)
            ip_dst = socket.inet_ntop(socket.AF_INET6, ip.dst)
            ttl = ip.hlim
            ip_hdr_len = 40
            total_len = ip.plen + 40
        else:
            return

        l4 = ip.data
        syn = urg = fin = teardown = False
        win = 0
        if isinstance(l4, dpkt.tcp.TCP):
            proto = 6
            sport, dport = l4.sport, l4.dport
            win = l4.win
            tcp_hdr_len = l4.off * 4
            flags = l4.flags
            syn = bool(flags & dpkt.tcp.TH_SYN)
            urg = bool(flags & dpkt.tcp.TH_URG)
            fin = bool(flags & dpkt.tcp.TH_FIN)
            teardown = fin or bool(flags & dpkt.tcp.TH_RST)
        elif isinstance(l4, dpkt.udp.UDP):
            proto = 17
            sport, dport = l4.sport, l4.dport
            tcp_hdr_len = 8
        else:
            return

        hdr_len = ip_hdr_len + tcp_hdr_len
        key, is_fwd = _canonical_key(ip_src, ip_dst, sport, dport, proto)
        h = _hash_key(key)

        state = self.table.get(h)
        if state is None:
            state = FlowState(key, ts, burst_tau=self.burst_tau)
            self.table[h] = state
            self.first_ts[h] = ts

        state.update(ts=ts, length=total_len, ttl=ttl, win=win,
                     hdr_len=hdr_len, is_fwd=is_fwd,
                     syn=syn, urg=urg, fin=fin)

        if teardown:
            self._finalise(h, out)

    def extract_to_dataframe(self, pcap_path: str) -> pd.DataFrame:
        """Run extraction over a PCAP and return a tidy DataFrame.

        Columns: the 30 features (ordered as ``FEATURE_NAMES``), the 5-tuple
        flow key, and ``label`` if a labeller was supplied.
        """
        rows = list(self.process(pcap_path))
        if not rows:
            cols = FEATURE_NAMES + ["ip_src", "ip_dst", "port_src",
                                    "port_dst", "proto"]
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(rows)
        ordered = FEATURE_NAMES + ["ip_src", "ip_dst", "port_src",
                                   "port_dst", "proto"]
        if "label" in df.columns:
            ordered.append("label")
        return df[ordered]


def extract_pcaps(pcap_paths: List[str],
                  labeller: Optional[Labeller] = None) -> pd.DataFrame:
    """Convenience wrapper: extract and concatenate several PCAP files."""
    frames = []
    for path in pcap_paths:
        ext = PcapExtractor(labeller=labeller)
        frames.append(ext.extract_to_dataframe(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
