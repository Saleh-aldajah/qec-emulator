"""
decoders.py — Syndrome decoders for the QEC Emulator.

Author : Dr. Saleh H. AlDaajeh
ORCID  : 0000-0001-7810-9290
Contact: S.aldaajeh@gmail.com

Three decoder classes share a common interface (decode_z, decode_x,
decode_success).  Two key findings from the v6 QAE study inform this module:

Finding 1 — ldpc v2 API.
  The legacy bposd_decoder function segfaults in ldpc >= 2.0.
  BPOSDDecoder uses only the v2 BpOsdDecoder class API.

Finding 2 — Graph-based decoders fail on BB codes.
  PyMatching v2 rejects parity-check matrix columns with weight > 2
  (raises ValueError).  BB codes have column weight 3 (each data qubit
  triggers three syndrome checks).
  MWPMDecoder implements DEM chain decomposition: each weight-3 column
  {a,b,c} -> two edges {a,b} and {b,c} via add_edge() API, discarding
  the {a,c} correlation.  A Union-Find BFS decoder (run_uf_decoder in
  benchmark.py) was also tested; it performs worse than MWPMDecoder,
  confirming that the failure is not specific to the DEM approximation
  but reflects the fundamental weight-3 hyperedge structure.
  Both graph-based references (MWPMDecoder, UF-BFS) fail substantially
  more often than BPOSDDecoder, which operates natively on the Tanner
  graph.  MWPMDecoder is therefore documented as a lossy reference
  baseline.  True hyperedge-perfect-matching is NP-hard; no open-source
  solver exists.

Reference: AlDaajeh (2026). doi:10.5281/zenodo.20574329
"""
from __future__ import annotations

import itertools
import math
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence

import numpy as np

from .codes import BBCode, in_rowspace, syndrome_from_mask, weight


class BaseDecoder(ABC):
    name:        str  = "base"
    is_fallback: bool = False

    def __init__(self, code: BBCode) -> None:
        self.code = code

    @abstractmethod
    def decode_z(self, syndrome_int: int) -> int: ...

    @abstractmethod
    def decode_x(self, syndrome_int: int) -> int: ...

    def decode_success(
        self,
        error_z_mask: int,
        error_x_mask: int,
        syndrome_z_override: Optional[int] = None,
        syndrome_x_override: Optional[int] = None,
    ) -> bool:
        code  = self.code
        syn_z = syndrome_z_override if syndrome_z_override is not None \
                else syndrome_from_mask(error_z_mask, code.hx_cols)
        syn_x = syndrome_x_override if syndrome_x_override is not None \
                else syndrome_from_mask(error_x_mask, code.hz_cols)
        corr_z     = self.decode_z(syn_z)
        corr_x     = self.decode_x(syn_x)
        residual_z = error_z_mask ^ corr_z
        residual_x = error_x_mask ^ corr_x
        vec_z = np.array([(residual_z >> i) & 1 for i in range(code.n)],
                         dtype=np.uint8)
        vec_x = np.array([(residual_x >> i) & 1 for i in range(code.n)],
                         dtype=np.uint8)
        return (in_rowspace(vec_z, code.hz_rref, code.hz_pivots) and
                in_rowspace(vec_x, code.hx_rref, code.hx_pivots))


# ---------------------------------------------------------------------------
# Lookup decoder
# ---------------------------------------------------------------------------

class LookupDecoder(BaseDecoder):
    """
    Exact bounded-distance syndrome lookup decoder.
    Self-contained: requires only NumPy.
    Suitable for radius t=2 with BB[[72,12,6]] where d>=6 is certified.
    """
    name = "Lookup"
    is_fallback = False

    def __init__(self, code: BBCode, radius: int = 2) -> None:
        super().__init__(code)
        self.radius   = radius
        self._table_z = self._build(code.hx_cols, code.n, radius)
        self._table_x = self._build(code.hz_cols, code.n, radius)

    @staticmethod
    def _build(col_syns: Sequence[int], n: int, radius: int) -> Dict[int, int]:
        table: Dict[int, int] = {0: 0}
        for w in range(1, radius + 1):
            for comb in itertools.combinations(range(n), w):
                syn = mask = 0
                for j in comb:
                    syn  ^= col_syns[j]
                    mask |= 1 << j
                if syn not in table or weight(mask) < weight(table[syn]):
                    table[syn] = mask
        return table

    def decode_z(self, syndrome_int: int) -> int:
        return self._table_z.get(syndrome_int, 0)

    def decode_x(self, syndrome_int: int) -> int:
        return self._table_x.get(syndrome_int, 0)


# ---------------------------------------------------------------------------
# BP+OSD decoder — ldpc v2 API
# ---------------------------------------------------------------------------

class BPOSDDecoder(BaseDecoder):
    """
    Belief propagation + ordered-statistics decoding.

    Uses the ldpc package v2 BpOsdDecoder class API.
    The legacy bposd_decoder function segfaults in ldpc >= 2.0 and
    must never be used.  Install: pip install ldpc

    Falls back to LookupDecoder (is_fallback=True) if ldpc is absent.
    """
    name = "BP+OSD"

    def __init__(
        self,
        code: BBCode,
        bp_method:         str   = "ms",
        ms_scaling_factor: float = 0.625,
        osd_method:        str   = "osd_cs",
        osd_order:         int   = 7,
        max_iter:          int   = 100,
        channel_probs:     float = 0.01,
    ) -> None:
        super().__init__(code)
        try:
            from ldpc import BpOsdDecoder as _BpOsd  # type: ignore
            kw = dict(error_rate=channel_probs, bp_method=bp_method,
                      ms_scaling_factor=ms_scaling_factor,
                      osd_method=osd_method, osd_order=osd_order,
                      max_iter=max_iter)
            self._dec_z      = _BpOsd(code.hx, **kw)
            self._dec_x      = _BpOsd(code.hz, **kw)
            self.is_fallback = False
        except ImportError:
            warnings.warn(
                "ldpc not installed — BPOSDDecoder falling back to "
                "LookupDecoder. Install: pip install ldpc",
                stacklevel=2)
            self._fallback   = LookupDecoder(code)
            self.is_fallback = True

    def _arr(self, s: int, nc: int) -> np.ndarray:
        return np.array([(s >> i) & 1 for i in range(nc)], dtype=np.uint8)

    def _mask(self, arr: np.ndarray) -> int:
        return int(sum((int(b) & 1) << i for i, b in enumerate(arr)))

    def decode_z(self, syndrome_int: int) -> int:
        if self.is_fallback:
            return self._fallback.decode_z(syndrome_int)
        return self._mask(self._dec_z.decode(self._arr(syndrome_int,
                                                        self.code.n2)))

    def decode_x(self, syndrome_int: int) -> int:
        if self.is_fallback:
            return self._fallback.decode_x(syndrome_int)
        return self._mask(self._dec_x.decode(self._arr(syndrome_int,
                                                        self.code.n2)))


# ---------------------------------------------------------------------------
# MWPM decoder — DEM chain decomposition (lossy reference baseline)
# ---------------------------------------------------------------------------

class MWPMDecoder(BaseDecoder):
    """
    Minimum-weight perfect matching with DEM chain decomposition.

    BB qLDPC parity-check matrices have column weight 3; PyMatching v2
    rejects weight->2 columns with ValueError.  This decoder uses the
    DEM chain decomposition: each weight-3 column {a,b,c} -> two edges
    {a,b} and {b,c} via add_edge() API, discarding the {a,c} correlation.

    Additionally, BB syndrome graphs can have odd-parity connected
    components (from logical operators); global high-weight boundary
    edges on every detector resolve this.

    IMPORTANT — this is a lossy reference baseline:
      - The {a,c} correlation is discarded.
      - A Union-Find BFS alternative (run_uf_decoder in benchmark.py)
        was tested and performs worse, confirming the failure reflects
        the fundamental weight-3 hyperedge structure, not only the DEM
        approximation.
      - Both graph-based decoders fail substantially more often than
        BPOSDDecoder on BB codes.
      - True hyperedge-perfect-matching is NP-hard; no open-source
        solver exists.  Hyperedge-aware implementations remain future work.

    Falls back to LookupDecoder if pymatching is absent.
    """
    name = "MWPM-DEM"

    def __init__(
        self,
        code:                   BBCode,
        channel_probs:          float = 0.01,
        boundary_weight_factor: float = 100.0,
    ) -> None:
        super().__init__(code)
        self._p   = channel_probs
        self._bwf = boundary_weight_factor
        try:
            import pymatching  # type: ignore
            self._mz = self._build(code.hx, code.n, code.n2,
                                   channel_probs, boundary_weight_factor)
            self._mx = self._build(code.hz, code.n, code.n2,
                                   channel_probs, boundary_weight_factor)
            self.is_fallback = False
        except ImportError:
            warnings.warn(
                "pymatching not installed — MWPMDecoder falling back to "
                "LookupDecoder. Install: pip install pymatching",
                stacklevel=2)
            self._fallback   = LookupDecoder(code)
            self.is_fallback = True

    @staticmethod
    def _build(H: np.ndarray, n_data: int, n_checks: int,
               p: float, bwf: float):
        import pymatching  # type: ignore
        w   = abs(math.log(p / (1 - p))) if 0 < p < 1 else 10.0
        w_b = w * bwf
        m   = pymatching.Matching()
        for j in range(n_data):
            checks = [i for i in range(n_checks) if H[i, j]]
            fid    = {j}
            if not checks: continue
            elif len(checks) == 1:
                m.add_boundary_edge(checks[0], weight=w, fault_ids=fid)
            elif len(checks) == 2:
                m.add_edge(checks[0], checks[1], weight=w, fault_ids=fid)
            else:                             # DEM chain: {a,b,c}->{a,b},{b,c}
                for k in range(len(checks) - 1):
                    m.add_edge(checks[k], checks[k+1], weight=w, fault_ids=fid)
        for i in range(n_checks):             # global boundary for odd parity
            m.add_boundary_edge(i, weight=w_b)
        return m

    def _arr(self, s: int, nc: int) -> np.ndarray:
        return np.array([(s >> i) & 1 for i in range(nc)], dtype=np.uint8)

    def _mask(self, arr: np.ndarray) -> int:
        return int(sum((int(b) & 1) << i for i, b in enumerate(arr)))

    def decode_z(self, syndrome_int: int) -> int:
        if self.is_fallback:
            return self._fallback.decode_z(syndrome_int)
        return self._mask(self._mz.decode(self._arr(syndrome_int, self.code.n2)))

    def decode_x(self, syndrome_int: int) -> int:
        if self.is_fallback:
            return self._fallback.decode_x(syndrome_int)
        return self._mask(self._mx.decode(self._arr(syndrome_int, self.code.n2)))
