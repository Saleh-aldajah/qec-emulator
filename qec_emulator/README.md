# QEC Emulator

**Reproducible qLDPC benchmark suite with hash-anchored provenance**

[![CI](https://github.com/YOUR_USERNAME/qec-emulator/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/qec-emulator/actions)
[![PyPI version](https://badge.fury.io/py/qec-emulator.svg)](https://pypi.org/project/qec-emulator/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![JOSS](https://joss.theoj.org/papers/XXXXXXX/status.svg)](https://joss.theoj.org/papers/XXXXXXX)

QEC Emulator benchmarks bivariate bicycle (BB) qLDPC codes and **records a
SHA-256 provenance fingerprint for every result** — making every published
logical failure rate directly verifiable by reviewers.

---

## Why this exists

Existing QEC simulation tools (Stim, PyMatching, panqec) are excellent at
producing results. None automatically answers the question a reviewer asks:
*"How do I know these numbers are real?"*

QEC Emulator answers that by attaching, to every simulation run:

| What | Why |
|---|---|
| SHA-256 of H_X and H_Z | Proves which code was used |
| Integer seed | Makes the run exactly reproducible |
| Wilson 95% upper bound | Correct statistics for rare failures |
| Result integrity hash | Detects copy-paste or seed collisions |
| Software environment snapshot | Pins the exact code version |

---

## Installation

```bash
# Core package (self-contained, no optional dependencies required)
pip install qec-emulator

# With BP+OSD and MWPM decoders
pip install "qec-emulator[decoders]"

# Everything including dev tools and docs
pip install "qec-emulator[all]"
```

**Requirements:** Python ≥ 3.9, NumPy ≥ 1.24

---

## 60-second quickstart

```python
from qec_emulator import BB72, LookupDecoder, PhenomenologicalModel
from qec_emulator.benchmark import run_sweep

code    = BB72()                          # BB[[72,12,6]] code
decoder = LookupDecoder(code, radius=2)   # Certified radius-2 lookup
noise   = PhenomenologicalModel()         # Independent Bernoulli errors

logger = run_sweep(
    code, decoder, noise,
    p_values=[0.001, 0.002, 0.004, 0.006, 0.008, 0.010, 0.012],
    trials=5000,
    seed=20260606,   # deterministic — same seed → same result
    batches=2,       # two batches detect seed collisions
)

logger.to_csv("bb72_sweep.csv")   # reviewer-ready CSV
logger.to_json("bb72_sweep.json") # full JSON with all metadata

# Check for potential duplicate records (reviewer concern)
flagged = logger.unresolved_pairs()
print(f"Flagged pairs: {len(flagged)}")  # Should be 0

print(logger.summary())
```

---

## Core API

### Codes

```python
from qec_emulator import BB72, BB144, Steane7

code = BB72()
print(code.n, code.k)          # 72, 12
print(code.commutation_check()) # True
print(code.hx_sha256())        # 64-char hex digest of H_X
```

### Decoders

```python
from qec_emulator import LookupDecoder, BPOSDDecoder, MWPMDecoder

dec = LookupDecoder(code, radius=2)    # self-contained
dec = BPOSDDecoder(code)               # requires: pip install ldpc
dec = MWPMDecoder(code)                # requires: pip install pymatching
```

All decoders fall back gracefully to `LookupDecoder` if optional
dependencies are not installed. The `is_fallback` attribute is recorded
in every provenance record so fallback results are never silently
presented as BP+OSD or MWPM results.

### Noise models

```python
from qec_emulator import PhenomenologicalModel, CircuitLevelModel, HardwareModel

noise = PhenomenologicalModel()          # code-capacity / repeated-cycle
noise = CircuitLevelModel()              # Pauli propagation + noisy readout
noise = HardwareModel("CZ")             # compiled superconducting CZ circuit
noise = HardwareModel("MS")             # compiled trapped-ion MS circuit
```

### Threshold scan

```python
from qec_emulator import BB72, BB144, LookupDecoder
from qec_emulator.benchmark import run_threshold_scan

logger = run_threshold_scan(
    codes=[BB72(), BB144()],
    decoder_factory=lambda code: LookupDecoder(code),
    noise_model=PhenomenologicalModel(),
    p_values=[0.001, 0.002, 0.004, 0.006, 0.008],
    trials=5000,
)
logger.to_csv("threshold_scan.csv")
```

### Distance certificate

```python
from qec_emulator.benchmark import run_distance_certificate

cert = run_distance_certificate(BB72(), max_weight=5)
print(cert["Z_component_hx"]["distance_lower_bound_pass"])  # True
```

---

## Reproducing a published result

Every provenance record contains the exact command needed to re-run it:

```bash
# From a published manifest row:
# code=BB72, decoder=Lookup, p=0.004, trials=5000, seed=20260606
python -c "
from qec_emulator import BB72, LookupDecoder, PhenomenologicalModel
from qec_emulator.benchmark import run_sweep
logger = run_sweep(BB72(), LookupDecoder(BB72()), PhenomenologicalModel(),
                   p_values=[0.004], trials=5000, seed=20260606)
print(logger.records[0].result_sha256)
"
```

If the printed SHA-256 matches the published manifest, the result is verified.

---

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Tests run on Python 3.9, 3.11, and 3.12 in CI.

---

## Citing this software

If you use QEC Emulator in your research, please cite:

```bibtex
@software{qec_emulator_2026,
  author    = {Author, Your},
  title     = {{QEC Emulator}: Reproducible qLDPC benchmark suite
               with hash-anchored provenance},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

A JOSS software paper is under review at
[joss.theoj.org](https://joss.theoj.org/papers/XXXXXXX).

---

## Related tools

| Tool | Role | Relationship |
|---|---|---|
| [Stim](https://github.com/quantumlib/Stim) | Stabilizer circuit simulation | QEC Emulator uses compatible noise models |
| [PyMatching](https://github.com/oscarhiggott/PyMatching) | MWPM decoder | Wrapped by MWPMDecoder |
| [ldpc](https://github.com/quantumlib/ldpc) | BP+OSD decoder | Wrapped by BPOSDDecoder |
| [panqec](https://github.com/panqec/panqec) | QEC visualisation | Complementary |
| [mqt.qecc](https://pypi.org/project/mqt.qecc/) | QEC toolbox | Complementary |

---

## License

MIT — see [LICENSE](LICENSE).
