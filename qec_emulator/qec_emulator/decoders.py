"""
decoders.py — Decoder implementations for the QEC Emulator.

All decoders share a common interface:
    decoder.decode(syndrome_int: int) -> correction_mask: int

Three decoders are provided:
  LookupDecoder   — exact radius-t syndrome lookup (certified, self-contained)
  BPOSDDecoder    — wraps the `ldpc` package (pip install ldpc); falls back
                    to lookup if ldpc is not installed.
  MWPMDecoder     — wraps PyMatching (pip install pymatching); falls back
                    to lookup if pymatching is not installed.

The fallback behaviour means the package is always runnable without optional
dependencies, but results using the fallback are labelled in provenance records
so they are never mistakenly cited as BP+OSD or MWPM results.
"""
from __future__ import annotations

import itertools
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence

import numpy as np

from .codes import BBCode, in_rowspace, syndrome_from_mask, weight


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseDecoder(ABC):
    """Common interface for all decoders."""

    name: str = "base"
    is_fallback: bool = False

    def __init__(self, code: BBCode) -> None:
        self.code = code

    @abstractmethod
    def decode_z(self, syndrome_int: int) -> int:
        """
        Decode a Z-type error from its X-check syndrome.

        Parameters
        ----------
        syndrome_int : int
            Bit-packed syndrome (column 0 = bit 0).

        Returns
        -------
        int
            Correction mask (data qubits to flip).
        """

    @abstractmethod
    def decode_x(self, syndrome_int: int) -> int:
        """Decode an X-type error from its Z-check syndrome."""

    def decode_success(
        self,
        error_z_mask: int,
        error_x_mask: int,
        syndrome_z_override: Optional[int] = None,
        syndrome_x_override: Optional[int] = None,
    ) -> bool:
        """
        Return True iff decoding succeeds for both X and Z components.

        Parameters
        ----------
        error_z_mask, error_x_mask : int
            Actual error masks on the data qubits.
        syndrome_z_override, syndrome_x_override : int, optional
            If provided, use these noisy syndromes instead of computing
            from the error masks (circuit-level simulation).
        """
        code = self.code
        syn_z = (
            syndrome_z_override
            if syndrome_z_override is not None
            else syndrome_from_mask(error_z_mask, code.hx_cols)
        )
        syn_x = (
            syndrome_x_override
            if syndrome_x_override is not None
            else syndrome_from_mask(error_x_mask, code.hz_cols)
        )
        corr_z = self.decode_z(syn_z)
        corr_x = self.decode_x(syn_x)
        residual_z = error_z_mask ^ corr_z
        residual_x = error_x_mask ^ corr_x
        return (
            in_rowspace(
                np.array([(residual_z >> i) & 1 for i in range(code.n)], dtype=np.uint8),
                code.hz_rref, code.hz_pivots,
            )
            and in_rowspace(
                np.array([(residual_x >> i) & 1 for i in range(code.n)], dtype=np.uint8),
                code.hx_rref, code.hx_pivots,
            )
        )


# ---------------------------------------------------------------------------
# Radius-t lookup decoder (self-contained, no external dependencies)
# ---------------------------------------------------------------------------

class LookupDecoder(BaseDecoder):
    """
    Exact bounded-distance syndrome lookup decoder.

    Builds a lookup table from all error patterns of weight ≤ radius and
    corrects any syndrome that appears in the table.  Matches the certified
    distance guarantee: for BB72 with radius=2 and d≥5, all weight-≤2
    errors are correctable.
    """

    name = "Lookup"
    is_fallback = False

    def __init__(self, code: BBCode, radius: int = 2) -> None:
        super().__init__(code)
        self.radius = radius
        self._table_z = self._build(code.hx_cols, code.n, radius)
        self._table_x = self._build(code.hz_cols, code.n, radius)

    @staticmethod
    def _build(col_syns: Sequence[int], n: int, radius: int) -> Dict[int, int]:
        table: Dict[int, int] = {0: 0}
        for w in range(1, radius + 1):
            for comb in itertools.combinations(range(n), w):
                syn, mask = 0, 0
                for j in comb:
                    syn ^= col_syns[j]
                    mask |= 1 << j
                if syn not in table or weight(mask) < weight(table[syn]):
                    table[syn] = mask
        return table

    def decode_z(self, syndrome_int: int) -> int:
        return self._table_z.get(syndrome_int, 0)

    def decode_x(self, syndrome_int: int) -> int:
        return self._table_x.get(syndrome_int, 0)


# ---------------------------------------------------------------------------
# BP+OSD decoder (wraps ldpc package)
# ---------------------------------------------------------------------------

class BPOSDDecoder(BaseDecoder):
    """
    Belief propagation + ordered statistics decoding via the `ldpc` package.

    Install with:  pip install ldpc

    If ldpc is not installed, falls back to LookupDecoder and sets
    is_fallback=True so that provenance records flag the substitution.
    """

    name = "BP+OSD"

    def __init__(
        self,
        code: BBCode,
        bp_method: str = "ms",
        ms_scaling_factor: float = 0.625,
        osd_method: str = "osd_cs",
        osd_order: int = 7,
        max_iter: int = 100,
        channel_probs: Optional[float] = None,
    ) -> None:
        super().__init__(code)
        self._channel_probs = channel_probs or 0.01
        try:
            from ldpc import bposd_decoder  # type: ignore
            self._dec_z = bposd_decoder(
                code.hx,
                error_rate=self._channel_probs,
                bp_method=bp_method,
                ms_scaling_factor=ms_scaling_factor,
                osd_method=osd_method,
                osd_order=osd_order,
                max_iter=max_iter,
            )
            self._dec_x = bposd_decoder(
                code.hz,
                error_rate=self._channel_probs,
                bp_method=bp_method,
                ms_scaling_factor=ms_scaling_factor,
                osd_method=osd_method,
                osd_order=osd_order,
                max_iter=max_iter,
            )
            self.is_fallback = False
        except ImportError:
            warnings.warn(
                "ldpc package not found. BPOSDDecoder falling back to LookupDecoder. "
                "Install with: pip install ldpc",
                stacklevel=2,
            )
            self._fallback = LookupDecoder(code)
            self.is_fallback = True

    def _mask_from_array(self, arr: np.ndarray) -> int:
        mask = 0
        for i, b in enumerate(arr):
            if int(b) & 1:
                mask |= 1 << i
        return mask

    def _syndrome_array(self, syndrome_int: int, n_checks: int) -> np.ndarray:
        return np.array([(syndrome_int >> i) & 1 for i in range(n_checks)], dtype=np.uint8)

    def decode_z(self, syndrome_int: int) -> int:
        if self.is_fallback:
            return self._fallback.decode_z(syndrome_int)
        syn = self._syndrome_array(syndrome_int, self.code.n2)
        corr = self._dec_z.decode(syn)
        return self._mask_from_array(corr)

    def decode_x(self, syndrome_int: int) -> int:
        if self.is_fallback:
            return self._fallback.decode_x(syndrome_int)
        syn = self._syndrome_array(syndrome_int, self.code.n2)
        corr = self._dec_x.decode(syn)
        return self._mask_from_array(corr)


# ---------------------------------------------------------------------------
# MWPM decoder (wraps pymatching)
# ---------------------------------------------------------------------------

class MWPMDecoder(BaseDecoder):
    """
    Minimum-weight perfect matching decoder via the `pymatching` package.

    Install with:  pip install pymatching

    Falls back to LookupDecoder if pymatching is not installed.
    """

    name = "MWPM"

    def __init__(self, code: BBCode, weights: Optional[np.ndarray] = None) -> None:
        super().__init__(code)
        try:
            import pymatching  # type: ignore
            self._match_z = pymatching.Matching(code.hx, weights=weights)
            self._match_x = pymatching.Matching(code.hz, weights=weights)
            self.is_fallback = False
        except ImportError:
            warnings.warn(
                "pymatching package not found. MWPMDecoder falling back to LookupDecoder. "
                "Install with: pip install pymatching",
                stacklevel=2,
            )
            self._fallback = LookupDecoder(code)
            self.is_fallback = True

    def _mask_from_array(self, arr: np.ndarray) -> int:
        mask = 0
        for i, b in enumerate(arr):
            if int(b) & 1:
                mask |= 1 << i
        return mask

    def _syndrome_array(self, syndrome_int: int, n_checks: int) -> np.ndarray:
        return np.array([(syndrome_int >> i) & 1 for i in range(n_checks)], dtype=np.uint8)

    def decode_z(self, syndrome_int: int) -> int:
        if self.is_fallback:
            return self._fallback.decode_z(syndrome_int)
        syn = self._syndrome_array(syndrome_int, self.code.n2)
        corr = self._match_z.decode(syn)
        return self._mask_from_array(corr)

    def decode_x(self, syndrome_int: int) -> int:
        if self.is_fallback:
            return self._fallback.decode_x(syndrome_int)
        syn = self._syndrome_array(syndrome_int, self.code.n2)
        corr = self._match_x.decode(syn)
        return self._mask_from_array(corr)
