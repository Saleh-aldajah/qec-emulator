---
title: 'QEC Emulator: A reproducible qLDPC benchmark suite with hash-anchored provenance'
tags:
  - Python
  - quantum error correction
  - qLDPC
  - bivariate bicycle code
  - reproducibility
  - benchmark
authors:
  - name: Your Name
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Your Institution
    index: 1
date: 06 June 2026
bibliography: paper.bib
---

# Summary

Quantum error correction (QEC) benchmarks guide hardware development by
predicting how well a logical qubit is protected against physical noise.
Bivariate bicycle (BB) codes have attracted attention as high-rate, low-overhead
alternatives to the surface code [@bravyi2024], but existing simulation
frameworks focus on performance rather than on making results reproducible under
adversarial reviewer scrutiny.

**QEC Emulator** fills this gap. It provides a Python package that runs Monte Carlo
benchmarks for BB codes and records, for every simulation result, the
SHA-256 digest of the parity-check matrices used, the exact integer seed, the
software environment, and an integrity seal over the output. These
*provenance records* allow a reviewer to download the repository, re-run a
flagged experiment, and confirm that the published logical failure rate (LFR)
is consistent with the raw logs — without trusting the author to report it
correctly.

# Statement of need

The reproducibility problem in computational QEC benchmarking is acute.
Existing tools — Stim [@gidney2021stim], PyMatching [@higgott2023sparse], and
the `ldpc` package [@roffe2020decoding] — are excellent sampling and decoding
engines, but they do not address the provenance layer: none automatically
associates a simulation output with the exact matrix bytes, seed policy, and
software version that produced it.

Sinter [@gidney2021stim] provides sampling orchestration but stores results in a
task-specific JSON format that is not designed for cross-study comparison or
reviewer re-execution. PanQEC [@panteleev2022asymptotically] offers
visualisation but does not log deterministic seeds or matrix fingerprints.

QEC Emulator is the first tool that treats *provenance as a first-class
output*. A ProvenanceLogger accompanies every benchmark run and can be exported
as a CSV or JSON manifest that maps each result row to its SHA-256 digest.
This directly addresses the reviewer concern identified in peer review of qLDPC
benchmark studies: that identical LFR values in two supposedly independent
batches may indicate seed collisions or copy-paste errors rather than genuine
zero-failure runs.

# Functionality

The package provides:

**Code construction.** The `BBCode` class constructs bivariate bicycle codes
over $\mathbb{F}_2[x, y]/(x^\ell - 1, y^m - 1)$ from polynomial parameters.
Named constructors `BB72()` and `BB144()` return the $[[72,12,6]]$ and
$[[144,12,12]]$ instances used in the QAE benchmark study [@author2026qae].
The Steane $[[7,1,3]]$ code is included as a sanity-check reference.
Construction is deterministic; `BBCode.hx_sha256()` returns the SHA-256
fingerprint of the $H_X$ matrix.

**Decoders.** Three decoder classes share a common interface
(`decode_z`, `decode_x`, `decode_success`):

- `LookupDecoder` — exact radius-$t$ syndrome lookup, self-contained (no
  external dependencies). Certified correct for all weight-$\le t$ errors.
- `BPOSDDecoder` — wraps the `ldpc` package [@roffe2020decoding] with
  configurable BP method, OSD order, and channel probability. Falls back
  gracefully to `LookupDecoder` with a `is_fallback=True` flag if `ldpc`
  is not installed, so provenance records are never silently mislabelled.
- `MWPMDecoder` — wraps PyMatching [@higgott2023sparse]. Same fallback
  mechanism.

**Noise models.** Three models are provided:

- `PhenomenologicalModel` — independent Bernoulli errors on data qubits
  with ideal syndrome readout.
- `CircuitLevelModel` — single-cycle Pauli propagation through the BB
  syndrome-extraction CNOT schedule with noisy ancilla preparation, gate
  errors, idle decoherence, and noisy measurement.
- `HardwareModel` — architecture-parameterised model for superconducting CZ
  and trapped-ion MS backends, incorporating $T_1$/$T_2$ decoherence and
  compiled circuit SWAP overhead.

**Benchmark runners.** `run_sweep` sweeps a grid of physical error rates and
returns a `ProvenanceLogger`. `run_threshold_scan` sweeps multiple code sizes
to locate a finite-size crossing. `run_fixed_weight` tests adversarial
fixed-weight error patterns. `run_distance_certificate` runs the
meet-in-the-middle distance lower-bound enumeration from [@bravyi2024] and
returns a structured certificate dict.

**Provenance.** Every `ProvenanceRecord` stores:
code name, decoder name, noise model, physical error rate $p$, trial count,
integer seed, SHA-256 of $H_X$ and $H_Z$, UTC timestamp, Python/NumPy
version, observed failures, LFR, one-sided Wilson 95% upper bound, and an
integrity SHA-256 of the result. The `unresolved_pairs()` method flags any
two records with identical (code, decoder, $p$, result hash) — directly
implementing the QC check recommended in the reviewer-gated evidence framework
of [@author2026qae].

# Usage example

```python
from qec_emulator import BB72, LookupDecoder, PhenomenologicalModel
from qec_emulator.benchmark import run_sweep

code    = BB72()
decoder = LookupDecoder(code, radius=2)
noise   = PhenomenologicalModel()

logger = run_sweep(
    code, decoder, noise,
    p_values=[0.001, 0.002, 0.004, 0.006, 0.008, 0.010, 0.012],
    trials=5000,
    seed=20260606,          # deterministic, reproducible
    batches=2,              # two independent batches detect seed collisions
    verbose=True,
)
logger.to_csv("bb72_sweep_manifest.csv")
logger.to_json("bb72_sweep_manifest.json")

flagged = logger.unresolved_pairs()
if flagged:
    print(f"WARNING: {len(flagged)} potential duplicate records found.")
```

The exported CSV is directly importable into the QAE workbook
[@author2026qae] and satisfies the reviewer-gated evidence principle that every
numerical claim must trace to a raw log with a deterministic seed and a
file hash.

# Testing and continuous integration

The package ships with a pytest suite covering code construction (parameters,
commutation, row/column weights, SHA-256 determinism), decoders (zero-syndrome,
weight-1 correctability, fallback labelling), noise models (zero-$p$ limits,
return types), provenance (set_result, QC status, CSV/JSON round-trip, duplicate
detection), and benchmark runners (determinism, weight-0 correctability, distance
certificate). Tests run on Python 3.9, 3.11, and 3.12 via GitHub Actions.

# Acknowledgements

This work was supported by [funding source]. The BB code construction follows
[@bravyi2024]. The distance certificate algorithm follows the meet-in-the-middle
approach described in [@author2026qae].

# References
