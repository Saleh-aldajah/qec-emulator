"""
codes.py — Bivariate Bicycle (BB) code construction over GF(2).

Provides BBCode, the [[72,12,6]] and [[144,12,12]] bivariate bicycle
code instances used throughout the QAE benchmark study.

All construction is deterministic and hash-stable: the same parameters
always produce byte-identical parity-check matrices.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# GF(2) linear algebra utilities
# ---------------------------------------------------------------------------

def gf2_rank(mat: np.ndarray) -> int:
    """Compute the GF(2) rank of a binary matrix."""
    A = (mat.copy() & 1).astype(np.uint8)
    m, n = A.shape
    rank = 0
    for col in range(n):
        pivot = next((r for r in range(rank, m) if A[r, col]), None)
        if pivot is None:
            continue
        if pivot != rank:
            A[[rank, pivot]] = A[[pivot, rank]]
        for r in range(m):
            if r != rank and A[r, col]:
                A[r] ^= A[rank]
        rank += 1
        if rank == m:
            break
    return rank


def gf2_rref(mat: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """Return reduced row echelon form and pivot column indices."""
    A = (mat.copy() & 1).astype(np.uint8)
    m, n = A.shape
    rank, pivots = 0, []
    for col in range(n):
        pivot = next((r for r in range(rank, m) if A[r, col]), None)
        if pivot is None:
            continue
        if pivot != rank:
            A[[rank, pivot]] = A[[pivot, rank]]
        for r in range(m):
            if r != rank and A[r, col]:
                A[r] ^= A[rank]
        pivots.append(col)
        rank += 1
        if rank == m:
            break
    return A[:rank], pivots


def in_rowspace(vec: np.ndarray, rref: np.ndarray, pivots: Sequence[int]) -> bool:
    """Test whether a binary vector lies in the row space given by rref."""
    v = (vec.copy() & 1).astype(np.uint8)
    for row, col in zip(rref, pivots):
        if v[col]:
            v ^= row
    return not v.any()


def syndrome_columns_as_ints(H: np.ndarray) -> List[int]:
    """Encode each column of H as a Python int (bit-packed syndrome)."""
    m, n = H.shape
    cols = []
    for j in range(n):
        val = 0
        for i in range(m):
            if H[i, j] & 1:
                val |= 1 << i
        cols.append(val)
    return cols


def syndrome_from_mask(mask: int, col_syns: Sequence[int]) -> int:
    """XOR the column syndromes for all set bits in mask."""
    s, x = 0, mask
    while x:
        lsb = x & -x
        s ^= col_syns[lsb.bit_length() - 1]
        x ^= lsb
    return s


def weight(mask: int) -> int:
    return bin(mask).count("1")


# ---------------------------------------------------------------------------
# Bivariate Bicycle code
# ---------------------------------------------------------------------------

@dataclass
class BBCode:
    """
    Bivariate Bicycle code over F2[x,y]/(x^ell - 1, y^m - 1).

    Default parameters give the BB[[72,12,6]] code used in the QAE study.

    Parameters
    ----------
    ell, m : int
        Group dimensions. Default 6×6 gives n=72.
    a_exponents : tuple of 3 ints
        Exponents for polynomial A = x^a0 + y^a1 + y^a2.
    b_exponents : tuple of 3 ints
        Exponents for polynomial B = y^b0 + x^b1 + x^b2.
    """
    ell: int = 6
    m: int = 6
    a_exponents: Tuple[int, int, int] = (3, 1, 2)
    b_exponents: Tuple[int, int, int] = (3, 1, 2)

    # Derived fields — populated by __post_init__
    n: int = field(init=False)
    k: int = field(init=False)
    hx: np.ndarray = field(init=False, repr=False)
    hz: np.ndarray = field(init=False, repr=False)
    hx_cols: List[int] = field(init=False, repr=False)
    hz_cols: List[int] = field(init=False, repr=False)
    hx_rref: np.ndarray = field(init=False, repr=False)
    hx_pivots: List[int] = field(init=False, repr=False)
    hz_rref: np.ndarray = field(init=False, repr=False)
    hz_pivots: List[int] = field(init=False, repr=False)
    rx: int = field(init=False)
    rz: int = field(init=False)

    def __post_init__(self) -> None:
        ell, m = self.ell, self.m
        I_ell = np.eye(ell, dtype=np.uint8)
        I_m = np.eye(m, dtype=np.uint8)
        x = {i: np.kron(np.roll(I_ell, i, axis=1), I_m) for i in range(ell)}
        y = {i: np.kron(I_ell, np.roll(I_m, i, axis=1)) for i in range(m)}

        a0, a1, a2 = self.a_exponents
        b0, b1, b2 = self.b_exponents
        A_terms = [x[a0], y[a1], y[a2]]
        B_terms = [y[b0], x[b1], x[b2]]
        A = (A_terms[0] ^ A_terms[1] ^ A_terms[2]).astype(np.uint8)
        B = (B_terms[0] ^ B_terms[1] ^ B_terms[2]).astype(np.uint8)

        self._A_terms = A_terms
        self._B_terms = B_terms
        self.hx = np.hstack((A, B)).astype(np.uint8)
        self.hz = np.hstack((B.T, A.T)).astype(np.uint8)
        self.n = 2 * ell * m
        self.rx = gf2_rank(self.hx)
        self.rz = gf2_rank(self.hz)
        self.k = self.n - self.rx - self.rz
        self.hx_cols = syndrome_columns_as_ints(self.hx)
        self.hz_cols = syndrome_columns_as_ints(self.hz)
        self.hx_rref, self.hx_pivots = gf2_rref(self.hx)
        self.hz_rref, self.hz_pivots = gf2_rref(self.hz)

    @property
    def n2(self) -> int:
        """Half the block length (number of checks per type)."""
        return self.ell * self.m

    def commutation_check(self) -> bool:
        """Verify Hx @ Hz^T = 0 over GF(2)."""
        return bool(np.array_equal(
            (self.hx @ self.hz.T) & 1,
            np.zeros((self.n2, self.n2), dtype=np.uint8)
        ))

    def neighbor(self, check_type: str, check_index: int, direction: int) -> int:
        """
        Return the data-qubit index connected to a given check by a Tanner edge.

        Parameters
        ----------
        check_type : 'X' or 'Z'
        check_index : int in [0, n2)
        direction : int in [0, 6) indexing the six terms of A and B
        """
        if check_type == "X":
            mat = self._A_terms[direction] if direction < 3 else self._B_terms[direction - 3]
            offset = 0 if direction < 3 else self.n2
        elif check_type == "Z":
            mat = self._B_terms[direction] if direction < 3 else self._A_terms[direction - 3]
            offset = 0 if direction < 3 else self.n2
        else:
            raise ValueError(f"check_type must be 'X' or 'Z', got {check_type!r}")

        if direction < 3:
            local = int(np.nonzero(mat[check_index, :])[0][0])
        else:
            local = int(np.nonzero(mat[:, check_index])[0][0])
        return offset + local

    def hx_sha256(self) -> str:
        """SHA-256 hex digest of the Hx matrix (canonical provenance fingerprint)."""
        return hashlib.sha256(self.hx.tobytes()).hexdigest()

    def hz_sha256(self) -> str:
        """SHA-256 hex digest of the Hz matrix."""
        return hashlib.sha256(self.hz.tobytes()).hexdigest()

    def to_dict(self) -> dict:
        """Serialise construction parameters and provenance digests."""
        return {
            "code": f"BB[[{self.n},{self.k},?]]",
            "ell": self.ell,
            "m": self.m,
            "a_exponents": list(self.a_exponents),
            "b_exponents": list(self.b_exponents),
            "n": self.n,
            "k": self.k,
            "rx": self.rx,
            "rz": self.rz,
            "commutation_pass": self.commutation_check(),
            "hx_sha256": self.hx_sha256(),
            "hz_sha256": self.hz_sha256(),
        }


# ---------------------------------------------------------------------------
# Named instances
# ---------------------------------------------------------------------------

def BB72() -> BBCode:
    """Return the BB[[72,12,6]] bivariate bicycle code (QAE benchmark)."""
    return BBCode(ell=6, m=6, a_exponents=(3, 1, 2), b_exponents=(3, 1, 2))


def BB144() -> BBCode:
    """Return the BB[[144,12,12]] bivariate bicycle code (QAE companion)."""
    return BBCode(ell=12, m=6, a_exponents=(3, 1, 2), b_exponents=(3, 1, 2))


def Steane7() -> BBCode:
    """
    Steane [[7,1,3]] code as a CSS sanity check.

    Uses a direct parity-check construction, not the BB polynomial form,
    but is returned as a BBCode-compatible object for uniform API usage.
    """
    # Standard Steane Hx / Hz matrices
    H = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ], dtype=np.uint8)
    code = object.__new__(BBCode)
    code.ell = 7
    code.m = 1
    code.a_exponents = (0, 0, 0)
    code.b_exponents = (0, 0, 0)
    code.hx = H.copy()
    code.hz = H.copy()
    code.n = 7
    code.rx = gf2_rank(H)
    code.rz = gf2_rank(H)
    code.k = 7 - code.rx - code.rz
    code.hx_cols = syndrome_columns_as_ints(H)
    code.hz_cols = syndrome_columns_as_ints(H)
    code.hx_rref, code.hx_pivots = gf2_rref(H)
    code.hz_rref, code.hz_pivots = gf2_rref(H)
    code._A_terms = []
    code._B_terms = []
    return code
