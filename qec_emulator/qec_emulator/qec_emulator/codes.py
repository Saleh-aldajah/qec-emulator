"""
codes.py — Bivariate Bicycle (BB) code construction over GF(2).

Author : Dr. Saleh H. AlDaajeh
ORCID  : 0000-0001-7810-9290
Contact: S.aldaajeh@gmail.com

Named instances match exactly the three codes benchmarked in the v5
QAE study (doi:10.5281/zenodo.20574329):

    BB72()  — BB[[72,12,6]]   ell=6,  m=6,  A=x^3+y+y^2, B=y^3+x+x^2
    BB144() — BB[[144,12,12]] ell=12, m=6,  same polynomials
    BB288() — BB[[288,12,18]] ell=12, m=12, A=x^3+y^2+y^7, B=y^3+x+x^2
                              (Bravyi et al. 2024, Table 1)

SHA-256 digests of HX and HZ are pinned to the archived data at the DOI
above and serve as tamper-evident provenance fingerprints.

References
----------
Bravyi et al. (2024). Nature 627, 778–782.
https://doi.org/10.1038/s41586-024-07107-7
"""
from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# GF(2) linear algebra
# ---------------------------------------------------------------------------

def gf2_rank(mat: np.ndarray) -> int:
    """Compute the rank of a binary matrix over GF(2)."""
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
    """Return True if the binary vector lies in the given row space."""
    v = (vec.copy() & 1).astype(np.uint8)
    for row, col in zip(rref, pivots):
        if v[col]:
            v ^= row
    return not v.any()


def syndrome_columns_as_ints(H: np.ndarray) -> List[int]:
    """Bit-pack each column of H into a Python int for fast syndrome arithmetic."""
    m, n = H.shape
    return [int(sum((int(H[i, j]) & 1) << i for i in range(m))) for j in range(n)]


def syndrome_from_mask(mask: int, col_syns: Sequence[int]) -> int:
    """XOR together the column syndromes for each set bit in mask."""
    s = 0
    x = mask
    while x:
        lsb = x & -x
        s ^= col_syns[lsb.bit_length() - 1]
        x ^= lsb
    return s


def weight(mask: int) -> int:
    return bin(mask).count("1")


# ---------------------------------------------------------------------------
# BBCode
# ---------------------------------------------------------------------------

@dataclass
class BBCode:
    """
    Bivariate Bicycle code over F2[x,y] / (x^ell - 1, y^m - 1).

    I parameterise the code by group dimensions (ell, m) and the
    exponents of the two generator polynomials A and B.

    Parameters
    ----------
    ell, m : int
    a_exponents : (int, int, int)  — exponents for A = x^a0 + y^a1 + y^a2
    b_exponents : (int, int, int)  — exponents for B = y^b0 + x^b1 + x^b2
    """
    ell: int = 6
    m:   int = 6
    a_exponents: Tuple[int, int, int] = (3, 1, 2)
    b_exponents: Tuple[int, int, int] = (3, 1, 2)

    n:          int        = field(init=False)
    k:          int        = field(init=False)
    hx:         np.ndarray = field(init=False, repr=False)
    hz:         np.ndarray = field(init=False, repr=False)
    hx_cols:    List[int]  = field(init=False, repr=False)
    hz_cols:    List[int]  = field(init=False, repr=False)
    hx_rref:    np.ndarray = field(init=False, repr=False)
    hx_pivots:  List[int]  = field(init=False, repr=False)
    hz_rref:    np.ndarray = field(init=False, repr=False)
    hz_pivots:  List[int]  = field(init=False, repr=False)
    rx:         int        = field(init=False)
    rz:         int        = field(init=False)

    def __post_init__(self) -> None:
        ell, m = self.ell, self.m
        Il = np.eye(ell, dtype=np.uint8)
        Im = np.eye(m,   dtype=np.uint8)
        x = {i: np.kron(np.roll(Il, i, 1), Im) for i in range(ell)}
        y = {i: np.kron(Il, np.roll(Im, i, 1)) for i in range(m)}

        a0, a1, a2 = self.a_exponents
        b0, b1, b2 = self.b_exponents
        A = (x[a0] ^ y[a1] ^ y[a2]).astype(np.uint8)
        B = (y[b0] ^ x[b1] ^ x[b2]).astype(np.uint8)

        self._A  = A
        self._B  = B
        self.hx  = np.hstack([A,   B  ]).astype(np.uint8)
        self.hz  = np.hstack([B.T, A.T]).astype(np.uint8)
        self.n   = 2 * ell * m
        self.rx  = gf2_rank(self.hx)
        self.rz  = gf2_rank(self.hz)
        self.k   = self.n - self.rx - self.rz

        self.hx_cols  = syndrome_columns_as_ints(self.hx)
        self.hz_cols  = syndrome_columns_as_ints(self.hz)
        self.hx_rref, self.hx_pivots = gf2_rref(self.hx)
        self.hz_rref, self.hz_pivots = gf2_rref(self.hz)

    @property
    def n2(self) -> int:
        """Number of checks of each type (= ell * m)."""
        return self.ell * self.m

    def commutation_check(self) -> bool:
        """Return True iff HX @ HZ^T = 0 over GF(2) (CSS condition)."""
        return bool(np.array_equal(
            (self.hx @ self.hz.T) & 1,
            np.zeros((self.n2, self.n2), dtype=np.uint8),
        ))

    def hx_sha256(self) -> str:
        """SHA-256 hex digest of HX (provenance fingerprint)."""
        return hashlib.sha256(self.hx.tobytes()).hexdigest()

    def hz_sha256(self) -> str:
        """SHA-256 hex digest of HZ."""
        return hashlib.sha256(self.hz.tobytes()).hexdigest()

    def to_dict(self) -> dict:
        """JSON-serialisable summary including provenance digests."""
        return {
            "code":             f"BB[[{self.n},{self.k},?]]",
            "ell":              self.ell,
            "m":                self.m,
            "a_exponents":      list(self.a_exponents),
            "b_exponents":      list(self.b_exponents),
            "n":                self.n,
            "k":                self.k,
            "rx":               self.rx,
            "rz":               self.rz,
            "commutation_pass": self.commutation_check(),
            "hx_sha256":        self.hx_sha256(),
            "hz_sha256":        self.hz_sha256(),
        }


# ---------------------------------------------------------------------------
# Named instances — pinned to v5 QAE study parameters
# ---------------------------------------------------------------------------

def BB72() -> BBCode:
    """
    BB[[72,12,6]] bivariate bicycle code.

    Primary benchmark in the v5 QAE study.
    HX SHA-256: 7ab0973bfd02e399d69728d26e67dbfa...
    """
    return BBCode(ell=6, m=6, a_exponents=(3, 1, 2), b_exponents=(3, 1, 2))


def BB144() -> BBCode:
    """
    BB[[144,12,12]] bivariate bicycle code.

    Second code size in the v5 QAE study.
    HX SHA-256: 195d5586a406f0194b89649b6a2499ec...
    """
    return BBCode(ell=12, m=6, a_exponents=(3, 1, 2), b_exponents=(3, 1, 2))


def BB288() -> BBCode:
    """
    BB[[288,12,18]] bivariate bicycle code.

    Third code size added in v5; polynomial parameters from
    Bravyi et al. (2024), Table 1: A = x^3 + y^2 + y^7, B = y^3 + x + x^2.
    HX SHA-256: c06e721ec6ea114b...

    Note: distance d >= 18 expected from Bravyi et al.; not independently
    certified here (MitM enumeration through weight 17 is computationally
    expensive and is identified as future work in the QAE study).
    """
    return BBCode(ell=12, m=12, a_exponents=(3, 2, 7), b_exponents=(3, 1, 2))


def Steane7() -> BBCode:
    """
    Steane [[7,1,3]] CSS code — sanity-check reference.

    I include this to verify that the decoder and provenance
    infrastructure behave correctly on a code whose distance is
    analytically established.
    """
    H = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ], dtype=np.uint8)
    code = object.__new__(BBCode)
    code.ell = 7; code.m = 1
    code.a_exponents = (0, 0, 0); code.b_exponents = (0, 0, 0)
    code.hx = H.copy(); code.hz = H.copy()
    code.n  = 7
    code.rx = gf2_rank(H); code.rz = gf2_rank(H)
    code.k  = 7 - code.rx - code.rz
    code.hx_cols = syndrome_columns_as_ints(H)
    code.hz_cols = syndrome_columns_as_ints(H)
    code.hx_rref, code.hx_pivots = gf2_rref(H)
    code.hz_rref, code.hz_pivots = gf2_rref(H)
    code._A = H.copy(); code._B = H.copy()
    return code
