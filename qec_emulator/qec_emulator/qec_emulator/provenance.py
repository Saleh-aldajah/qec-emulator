"""
provenance.py — Hash-anchored reproducibility logging.

Author : Dr. Saleh H. AlDaajeh
ORCID  : 0000-0001-7810-9290
Contact: S.aldaajeh@gmail.com

I designed the provenance layer as the core contribution of this package.
The central idea is that a simulation result should be as tamper-evident
as a signed document: the inputs, the random seed, the software
environment, and the output are all folded into a single SHA-256 digest
that any reviewer can independently verify by re-running the experiment.

ProvenanceRecord
    Dataclass that captures everything needed to reproduce and verify
    one Monte Carlo run.  The result_sha256 field is the integrity seal
    over the (code, decoder, p, trials, seed, failures) tuple.

ProvenanceLogger
    Accumulates records and exports them as a CSV or JSON manifest.
    The unresolved_pairs() method implements the QC check I require
    before submitting benchmark data: any two records with identical
    (code, decoder, p, result_sha256) are flagged as potential seed
    collisions or copy-paste errors.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# SHA-256 utilities
# ---------------------------------------------------------------------------

def sha256_of_dict(d: dict) -> str:
    """Deterministic SHA-256 of a JSON-serialisable dict (keys sorted)."""
    blob = json.dumps(d, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def sha256_of_array(arr: np.ndarray) -> str:
    """SHA-256 of a numpy array's raw bytes."""
    return hashlib.sha256(arr.tobytes()).hexdigest()


def sha256_of_file(path) -> str:
    """SHA-256 of a file on disk (for compiled circuit artefacts)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _wilson95_upper(k: int, n: int) -> float:
    """One-sided 95% Wilson score upper bound for a binomial proportion."""
    if n == 0:
        return 1.0
    z     = 1.645
    p     = k / n
    denom = 1 + z * z / n
    return min(1.0, (
        p + z * z / (2 * n)
        + z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5
    ) / denom)


def _env() -> dict:
    return {"python": sys.version, "platform": platform.platform(),
            "numpy": np.__version__}


# ---------------------------------------------------------------------------
# ProvenanceRecord
# ---------------------------------------------------------------------------

@dataclass
class ProvenanceRecord:
    """
    Complete reproducibility record for one simulation run.

    All fields are populated before the run except failures, lfr,
    wilson95_upper, and result_sha256, which are set by calling
    set_result(failures) after the Monte Carlo loop completes.
    """
    code:               str
    decoder:            str
    noise_model:        str
    p:                  float
    trials:             int
    seed:               int
    hx_sha256:          str
    hz_sha256:          str
    timestamp:          str  = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    env:                dict = field(default_factory=_env)
    failures:           Optional[int]   = None
    lfr:                Optional[float] = None
    wilson95_upper:     Optional[float] = None
    result_sha256:      Optional[str]   = None
    decoder_is_fallback: bool           = False
    notes:              str             = ""

    def set_result(self, failures: int) -> None:
        """Populate result fields and compute the integrity seal."""
        self.failures      = failures
        self.lfr           = failures / self.trials if self.trials else 0.0
        self.wilson95_upper = _wilson95_upper(failures, self.trials)
        self.result_sha256 = sha256_of_dict({
            "code":      self.code,
            "decoder":   self.decoder,
            "noise":     self.noise_model,
            "p":         self.p,
            "trials":    self.trials,
            "seed":      self.seed,
            "failures":  self.failures,
            "hx_sha256": self.hx_sha256,
            "hz_sha256": self.hz_sha256,
        })

    def qc_status(self) -> str:
        if self.lfr is None:
            return "PENDING"
        if self.lfr == 0.0:
            return f"ZERO_FAILURE (wilson95≤{self.wilson95_upper:.5f})"
        if self.lfr < 0.05:
            return "PASS"
        if self.lfr < 0.30:
            return "WARN"
        return "FAIL"

    CSV_FIELDS = [
        "code", "decoder", "noise_model", "p", "trials", "seed",
        "failures", "lfr", "wilson95_upper", "hx_sha256", "hz_sha256",
        "result_sha256", "decoder_is_fallback", "qc_status", "timestamp", "notes",
    ]

    def to_csv_row(self) -> dict:
        d = asdict(self)
        d["qc_status"] = self.qc_status()
        return {k: d.get(k, "") for k in self.CSV_FIELDS}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["qc_status"] = self.qc_status()
        return d


# ---------------------------------------------------------------------------
# ProvenanceLogger
# ---------------------------------------------------------------------------

class ProvenanceLogger:
    """
    Accumulates ProvenanceRecord objects and exports them as CSV or JSON.

    I export both formats so that the CSV can be submitted directly as
    a supplementary data file and the JSON can be archived on Zenodo as
    the machine-readable provenance manifest.
    """

    def __init__(self) -> None:
        self.records: List[ProvenanceRecord] = []

    def new_record(
        self,
        code_name:           str,
        decoder_name:        str,
        noise_model:         str,
        p:                   float,
        trials:              int,
        seed:                int,
        hx_sha256:           str,
        hz_sha256:           str,
        decoder_is_fallback: bool = False,
        notes:               str  = "",
    ) -> ProvenanceRecord:
        rec = ProvenanceRecord(
            code=code_name, decoder=decoder_name, noise_model=noise_model,
            p=p, trials=trials, seed=seed,
            hx_sha256=hx_sha256, hz_sha256=hz_sha256,
            decoder_is_fallback=decoder_is_fallback, notes=notes,
        )
        self.records.append(rec)
        return rec

    def to_csv(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ProvenanceRecord.CSV_FIELDS)
            w.writeheader()
            for rec in self.records:
                w.writerow(rec.to_csv_row())

    def to_json(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([rec.to_dict() for rec in self.records], f, indent=2, default=str)

    def summary(self) -> str:
        done  = sum(1 for r in self.records if r.lfr is not None)
        fails = sum(1 for r in self.records if r.lfr is not None and r.lfr >= 0.05)
        return f"{done}/{len(self.records)} records complete, {fails} with LFR ≥ 5%"

    def unresolved_pairs(self) -> List[ProvenanceRecord]:
        """
        Flag records whose (code, decoder, p, result_sha256) tuple is
        shared by more than one record — a potential seed collision or
        copy-paste error that must be resolved before publication.
        """
        from collections import defaultdict
        groups: Dict = defaultdict(list)
        for rec in self.records:
            if rec.result_sha256:
                key = (rec.code, rec.decoder, rec.p, rec.result_sha256)
                groups[key].append(rec)
        flagged = []
        for recs in groups.values():
            if len(recs) > 1:
                flagged.extend(recs)
        return flagged
