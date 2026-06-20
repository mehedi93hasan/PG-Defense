"""Streaming pcap -> 30 protocol-grounded features (Tier-2 of the architecture).

Single packet pass, O(1) per-flow state via Welford accumulators. Flows keyed by
the 5-tuple; emitted on TCP teardown (FIN/RST) or idle timeout. Requires `dpkt`.

This reproduces the feature definitions in pgdef/features.py. Dataset-specific
labelling (attack vs benign, attack class) is applied separately by the label_*
helpers, since ground truth differs per dataset.

Usage:
    python extraction/pgdef_extractor.py input.pcap output.csv
"""
import sys, csv, math
try:
    import dpkt
except Exception:
    dpkt = None
from collections import defaultdict

IDLE_TIMEOUT = 120.0
BURST_GAP = 0.1


class Welford:
    __slots__ = ("n", "mean", "m2", "vmin", "vmax")
    def __init__(self): self.n = 0; self.mean = 0.0; self.m2 = 0.0; self.vmin = math.inf; self.vmax = -math.inf
    def update(self, x):
        self.n += 1; d = x - self.mean; self.mean += d / self.n
        self.m2 += d * (x - self.mean); self.vmin = min(self.vmin, x); self.vmax = max(self.vmax, x)
    def std(self): return math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else 0.0


class FlowState:
    def __init__(self, t):
        self.t_first = t; self.t_last = t; self.n = 0
        self.iat = Welford(); self.ttl = Welford(); self.win = Welford()
        self.plen = Welford(); self.hdr = Welford()
        self.fwd_pkts = 0; self.bwd_pkts = 0; self.fwd_bytes = 0; self.bwd_bytes = 0
        self.fwd_iat = Welford(); self.last_fwd_t = None
        self.fwd_pl = Welford(); self.bwd_pl = Welford(); self.syn = 0
        self.active = Welford(); self.idle = Welford(); self.burst_start = t; self.last_t = t

    def update(self, t, ttl, win, plen, hdrlen, is_fwd, syn):
        if self.n >= 1:
            iat = t - self.t_last; self.iat.update(iat)
            if iat > BURST_GAP:
                self.active.update(self.last_t - self.burst_start); self.idle.update(iat)
                self.burst_start = t
        self.n += 1; self.ttl.update(ttl); self.win.update(win); self.plen.update(plen)
        self.hdr.update(hdrlen)
        if syn: self.syn += 1
        if is_fwd:
            self.fwd_pkts += 1; self.fwd_bytes += plen; self.fwd_pl.update(plen)
            if self.last_fwd_t is not None: self.fwd_iat.update(t - self.last_fwd_t)
            self.last_fwd_t = t
        else:
            self.bwd_pkts += 1; self.bwd_bytes += plen; self.bwd_pl.update(plen)
        self.t_last = t; self.last_t = t

    def features(self):
        dur = max(self.t_last - self.t_first, 1e-6)
        return [
            dur, self.iat.mean, self.iat.std(),
            (0.0 if self.iat.vmin is math.inf else self.iat.vmin),
            (0.0 if self.iat.vmax == -math.inf else self.iat.vmax),
            self.active.mean, self.idle.mean, self.fwd_iat.mean,
            self.ttl.mean, self.ttl.std(),
            (0.0 if self.ttl.vmin is math.inf else self.ttl.vmin),
            (0.0 if self.ttl.vmax == -math.inf else self.ttl.vmax),
            self.win.mean, self.win.std(), self.syn, self.hdr.mean,
            self.fwd_pkts, self.bwd_pkts,
            self.fwd_pkts / max(self.bwd_pkts, 1), self.fwd_bytes / max(self.bwd_bytes, 1),
            self.plen.mean, self.plen.std(),
            (0.0 if self.plen.vmin is math.inf else self.plen.vmin),
            (0.0 if self.plen.vmax == -math.inf else self.plen.vmax),
            self.fwd_pl.mean, self.bwd_pl.mean,
            self.n / dur, (self.fwd_bytes + self.bwd_bytes) / dur,
            self.fwd_pkts / dur, self.bwd_pkts / dur,
        ]


def extract(pcap_path, out_csv):
    if dpkt is None:
        sys.exit("dpkt not installed:  pip install dpkt")
    from pgdef.features import COLUMN_NAMES
    flows = {}
    with open(pcap_path, "rb") as f, open(out_csv, "w", newline="") as o:
        w = csv.writer(o); w.writerow(COLUMN_NAMES)
        for ts, buf in dpkt.pcap.Reader(f):
            try:
                eth = dpkt.ethernet.Ethernet(buf); ip = eth.data
                if not isinstance(ip, dpkt.ip.IP): continue
                l4 = ip.data
                sport = getattr(l4, "sport", 0); dport = getattr(l4, "dport", 0)
                key = (ip.src, ip.dst, sport, dport, ip.p)
                rkey = (ip.dst, ip.src, dport, sport, ip.p)
                fkey = key if key in flows or rkey not in flows else rkey
                is_fwd = (fkey == key)
                if fkey not in flows: flows[fkey] = FlowState(ts)
                syn = int(isinstance(l4, dpkt.tcp.TCP) and (l4.flags & dpkt.tcp.TH_SYN) != 0)
                plen = len(ip.data.data) if hasattr(ip.data, "data") else 0
                hdrlen = ip.hl * 4 + (getattr(l4, "off", 5) >> 4) * 4 if isinstance(l4, dpkt.tcp.TCP) else ip.hl * 4
                win = getattr(l4, "win", 0)
                flows[fkey].update(ts, ip.ttl, win, plen, hdrlen, is_fwd, syn)
                fin = isinstance(l4, dpkt.tcp.TCP) and (l4.flags & (dpkt.tcp.TH_FIN | dpkt.tcp.TH_RST))
                if fin:
                    w.writerow(flows[fkey].features()); del flows[fkey]
            except Exception:
                continue
        for st in flows.values():
            w.writerow(st.features())
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    sys.path.insert(0, "..")
    if len(sys.argv) != 3:
        sys.exit("usage: python pgdef_extractor.py input.pcap output.csv")
    extract(sys.argv[1], sys.argv[2])
