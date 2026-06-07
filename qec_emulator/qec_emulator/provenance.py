"""
provenance.py — Hash-anchored reproducibility logging.

Every simulation result produced by the QEC Emulator is associated with a
ProvenanceRecord that captures the exact inputs, random seed, software
versions, and output digest needed to reconstruct and verify the result.

This is the core feature that makes results reviewer-auditable.

Usage
-----
    logger = ProvenanceLogger()
    with logger.record(code, decoder, noise_model, p, trials, seed) as rec:
        result = run_experiment(...)
        rec.set_result(result)
    logger.to_csv("manifest.csv")
    logger.to_json("manifest.json")
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_of_dict(d: dict) -> str:
    """Deterministic SHA-256 of a JSON-serialisable dict (sorted keys)."""
    blob = json.dumps(d, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def sha256_of_array(arr: np.ndarray) -> str:
    """SHA-256 of a numpy array's raw bytes."""
    return hashlib.sha256(arr.tobytes()).hexdigest()


def sha256_of_file(path: str | Path) -> str:
    """SHA-256 of a file on disk (for circuit / schedule artefacts)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _software_env() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
    }


# ---------------------------------------------------------------------------
# ProvenanceRecord
# ---------------------------------------------------------------------------

@dataclass
class ProvenanceRecord:
    """
    A complete reproducibility record for one simulation run.

    Fields
    ------
    code        : short name e.g. "BB72"
    decoder     : decoder name e.g. "BP+OSD"
    noise_model : noise model name e.g. "phenomenological"
    p           : physical error rate
    trials      : number of Monte Carlo shots
    seed        : integer RNG seed used
    hx_sha256   : SHA-256 of the Hx parity-check matrix
    hz_sha256   : SHA-256 of the Hz parity-check matrix
    timestamp   : ISO-8601 UTC timestamp of the run
    env         : software environment snapshot
    failures    : observed logical failures (set after run)
    lfr         : logical failure rate = failures / trials
    wilson95_upper : one-sided 95% Wilson upper bound
    result_sha256  : SHA-256 of the result dict (integrity seal)
    decoder_is_fallback : True if optional decoder was unavailable
    notes       : free-form annotation string
    """
    code: str
    decoder: str
    noise_model: str
    p: float
    trials: int
    seed: int
    hx_sha256: str
    hz_sha256: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    env: dict = field(default_factory=_software_env)
    failures: Optional[int] = None
    lfr: Optional[float] = None
    wilson95_upper: Optional[float] = None
    result_sha256: Optional[str] = None
    decoder_is_fallback: bool = False
    notes: str = ""

    def set_result(self, failures: int) -> None:
        """Populate result fields and compute the integrity seal."""
        self.failures = failures
        self.lfr = failures / self.trials if self.trials > 0 else 0.0
        self.wilson95_upper = _wilson95_upper(failures, self.trials)
        result_dict = {
            "code": self.code,
            "decoder": self.decoder,
            "noise_model": self.noise_model,
            "p": self.p,
            "trials": self.trials,
            "seed": self.seed,
            "failures": self.failures,
            "lfr": self.lfr,
            "hx_sha256": self.hx_sha256,
            "hz_sha256": self.hz_sha256,
        }
        self.result_sha256 = sha256_of_dict(result_dict)

    def qc_status(self) -> str:
        """Quick-look QC verdict."""
        if self.lfr is None:
            return "PENDING"
        if self.lfr == 0.0:
            return f"ZERO_FAILURE (upper95={self.wilson95_upper:.5f})"
        if self.lfr < 0.05:
            return "PASS"
        if self.lfr < 0.30:
            return "WARN"
        return "FAIL"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["qc_status"] = self.qc_status()
        return d

    CSV_FIELDS = [
        "code", "decoder", "noise_model", "p", "trials", "seed",
        "failures", "lfr", "wilson95_upper", "hx_sha256", "hz_sha256",
        "result_sha256", "decoder_is_fallback", "qc_status", "timestamp", "notes",
    ]

    def to_csv_row(self) -> dict:
        d = self.to_dict()
        return {k: d.get(k, "") for k in self.CSV_FIELDS}


def _wilson95_upper(k: int, n: int) -> float:
    """One-sided 95% Wilson upper bound for a binomial proportion."""
    if n == 0:
        return 1.0
    z = 1.645  # one-sided 95%
    p = k / n
    denom = 1 + z * z / n
    return min(1.0, (p + z * z / (2 * n) + z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom)


# ---------------------------------------------------------------------------
# ProvenanceLogger
# ---------------------------------------------------------------------------

class ProvenanceLogger:
    """
    Accumulates ProvenanceRecord objects and exports them as CSV or JSON.

    Usage
    -----
        logger = ProvenanceLogger()
        rec = logger.new_record(code, decoder, ...)
        # ... run experiment ...
        rec.set_result(failures)
        logger.to_csv("manifest.csv")
    """

    def __init__(self) -> None:
        self.records: List[ProvenanceRecord] = []

    def new_record(
        self,
        code_name: str,
        decoder_name: str,
        noise_model: str,
        p: float,
        trials: int,
        seed: int,
        hx_sha256: str,
        hz_sha256: str,
        decoder_is_fallback: bool = False,
        notes: str = "",
    ) -> ProvenanceRecord:
        """Create and register a new ProvenanceRecord."""
        rec = ProvenanceRecord(
            code=code_name,
            decoder=decoder_name,
            noise_model=noise_model,
            p=p,
            trials=trials,
            seed=seed,
            hx_sha256=hx_sha256,
            hz_sha256=hz_sha256,
            decoder_is_fallback=decoder_is_fallback,
            notes=notes,
        )
        self.records.append(rec)
        return rec

    def to_csv(self, path: str | Path) -> None:
        """Write all records to a CSV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ProvenanceRecord.CSV_FIELDS)
            writer.writeheader()
            for rec in self.records:
                writer.writerow(rec.to_csv_row())

    def to_json(self, path: str | Path) -> None:
        """Write all records to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                [rec.to_dict() for rec in self.records],
                f, indent=2, default=str
            )

    def summary(self) -> str:
        """One-line summary string."""
        n = len(self.records)
        done = sum(1 for r in self.records if r.lfr is not None)
        fails = sum(1 for r in self.records if r.lfr is not None and r.lfr >= 0.05)
        return f"{done}/{n} records complete, {fails} with LFR ≥ 5%"

    def unresolved_pairs(self) -> List[ProvenanceRecord]:
        """
        Return records flagged as potential identical-LFR pairs.

        Two records are 'unresolved' if they share the same (code, decoder, p)
        and have identical result_sha256. This catches the reviewer concern
        about copy-paste or seed collisions.
        """
        from collections import defaultdict
        groups: dict = defaultdict(list)
        for rec in self.records:
            if rec.result_sha256:
                key = (rec.code, rec.decoder, rec.p, rec.result_sha256)
                groups[key].append(rec)
        flagged = []
        for recs in groups.values():
            if len(recs) > 1:
                flagged.extend(recs)
        return flagged
