"""
export.py — Multi-format result export and import.

Supports CSV (default), JSON, and HDF5 (optional h5py).
Results are represented as lists of dicts and can be round-tripped.

Author: Dr. Saleh H. AlDaajeh  ORCID: 0000-0001-7810-9290
"""
from __future__ import annotations
import csv, json, math, hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def _wilson95(k: int, n: int):
    if n == 0: return 0., 1.
    z = 1.96; p = k/n
    c = (p+z*z/(2*n))/(1+z*z/n)
    m = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return max(0.,c-m), min(1.,c+m)


class ResultSet:
    """
    Container for a set of simulation results with multi-format export.

    Attributes
    ----------
    rows :  list of result dicts (one per (code, p, decoder, trial-batch))
    meta :  free-form metadata dict (run info, versions, timestamps)
    """

    def __init__(self, rows: List[Dict[str, Any]] = None,
                 meta: Dict[str, Any] = None):
        self.rows = rows or []
        self.meta = meta or {
            "generated_utc":  datetime.now(timezone.utc).isoformat(),
            "emulator_version": "1.2.3",
            "author_orcid":    "0000-0001-7810-9290",
        }

    # ── Append helpers ──────────────────────────────────────────────────────

    def add(self, code_name: str, decoder_name: str, p: float,
            failures: int, trials: int, seed: int,
            hx_sha: str = "", hz_sha: str = "",
            extra: Dict = None) -> None:
        lo, hi = _wilson95(failures, trials)
        row = dict(
            code=code_name, decoder=decoder_name, p=p,
            trials=trials, failures=failures,
            lfr=round(failures/trials, 8) if trials else 0,
            ci_lo=round(lo, 8), ci_hi=round(hi, 8),
            seed=seed, hx_sha256=hx_sha, hz_sha256=hz_sha,
        )
        if extra:
            row.update(extra)
        self.rows.append(row)

    # ── Export ──────────────────────────────────────────────────────────────

    def to_csv(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.rows:
            path.write_text("")
            return path
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=self.rows[0].keys())
            w.writeheader()
            w.writerows(self.rows)
        return path

    def to_json(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"meta": self.meta, "rows": self.rows}, indent=2))
        return path

    def to_hdf5(self, path: Union[str, Path]) -> Path:
        """Export to HDF5 if h5py is installed; otherwise raises ImportError."""
        try:
            import h5py, numpy as np
        except ImportError:
            raise ImportError("h5py is required for HDF5 export: pip install h5py")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, 'w') as f:
            for k, v in self.meta.items():
                f.attrs[k] = str(v)
            if self.rows:
                for col in self.rows[0]:
                    vals = [r.get(col, '') for r in self.rows]
                    try:
                        f.create_dataset(col, data=np.array(vals, dtype=float))
                    except (ValueError, TypeError):
                        dt = h5py.special_dtype(vlen=str)
                        ds = f.create_dataset(col, (len(vals),), dtype=dt)
                        for i, v2 in enumerate(vals): ds[i] = str(v2)
        return path

    def save(self, base_path: Union[str, Path],
             formats: List[str] = None) -> Dict[str, Path]:
        """Save in one or more formats. Returns dict of {format: path}."""
        formats = formats or ["csv"]
        base = Path(base_path)
        out = {}
        for fmt in formats:
            fmt = fmt.lower()
            if fmt == "csv":
                out["csv"] = self.to_csv(base.with_suffix('.csv'))
            elif fmt == "json":
                out["json"] = self.to_json(base.with_suffix('.json'))
            elif fmt in ("hdf5", "h5"):
                out["hdf5"] = self.to_hdf5(base.with_suffix('.h5'))
        return out

    # ── Import ──────────────────────────────────────────────────────────────

    @classmethod
    def from_csv(cls, path: Union[str, Path]) -> "ResultSet":
        path = Path(path)
        rows = []
        with open(path, newline='') as f:
            for r in csv.DictReader(f):
                rows.append({k: _try_num(v) for k, v in r.items()})
        return cls(rows=rows, meta={"source": str(path)})

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ResultSet":
        data = json.loads(Path(path).read_text())
        return cls(rows=data.get("rows", []), meta=data.get("meta", {}))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ResultSet":
        path = Path(path)
        if path.suffix == '.csv':  return cls.from_csv(path)
        if path.suffix == '.json': return cls.from_json(path)
        raise ValueError(f"Unknown extension: {path.suffix}")

    # ── Convenience ─────────────────────────────────────────────────────────

    def filter(self, **kwargs) -> "ResultSet":
        """Return a new ResultSet with rows matching all keyword filters."""
        rows = [r for r in self.rows
                if all(str(r.get(k)) == str(v) for k, v in kwargs.items())]
        return ResultSet(rows=rows, meta=dict(self.meta))

    def summary(self) -> str:
        """Short human-readable summary."""
        if not self.rows:
            return "ResultSet: 0 rows"
        codes   = sorted(set(r.get('code','?') for r in self.rows))
        decoders= sorted(set(r.get('decoder','?') for r in self.rows))
        return (f"ResultSet: {len(self.rows)} rows | "
                f"codes: {codes} | decoders: {decoders}")

    def __len__(self) -> int:
        return len(self.rows)

    def __repr__(self) -> str:
        return self.summary()

    def integrity_hash(self) -> str:
        """SHA-256 of the sorted serialised rows for provenance tracking."""
        canonical = json.dumps(sorted(
            [{k: v for k, v in r.items() if k not in ('ci_lo','ci_hi')}
             for r in self.rows],
            key=lambda r: (r.get('code',''), r.get('decoder',''), r.get('p',0))
        ), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()


def _try_num(v: str) -> Any:
    """Attempt numeric coercion; fall back to string."""
    try:    return int(v)
    except: pass
    try:    return float(v)
    except: pass
    return v
