---
title: 'QEC Emulator v2.0.0: A reproducible open-source qLDPC benchmark suite with REST API, CLI, and full interoperability'
tags:
  - Python
  - quantum error correction
  - qLDPC codes
  - bivariate bicycle codes
  - fault-tolerant quantum computing
  - reproducibility
  - benchmark
  - open science
authors:
  - name: Saleh H. AlDaajeh
    orcid: 0000-0001-7810-9290
    corresponding: true
    email: S.aldaajeh@gmail.com
    affiliation: 1
affiliations:
  - name: Independent Researcher, Information Systems and Security
    index: 1
date: 08 June 2026
bibliography: paper.bib
---

# Summary

Quantum low-density parity-check (qLDPC) codes are a leading approach to
low-overhead fault-tolerant quantum computation [@bravyi2024;
@panteleev2022asymptotically].
Benchmarking these codes under realistic noise requires sweeping physical
error rates, comparing decoder algorithms, and testing hardware-compiled
noise models — all while maintaining a verifiable, tamper-evident record
of exactly which matrix bytes, random seed, and software version produced
each result.

**QEC Emulator** is an open-source Python package (MIT licence) that
provides this capability.
It constructs three bivariate bicycle (BB) code instances,
runs single-round and repeated-cycle logical failure rate (LFR)
benchmarks under four noise models and three decoder classes, and records
a SHA-256 integrity seal over every (code, decoder, noise model, seed,
result) tuple.
Every published number is independently verifiable: a reader installs the
package, reruns the experiment with the recorded seed, and compares the
digest.

The package is designed as an extensible research platform.
Version 2.0.0 adds a REST/HTTP API server (FastAPI), a command-line interface
(Typer + Rich), parallel job scheduling, multi-format export (CSV/JSON/HDF5),
Stim circuit and DEM export, NetworkX Tanner/factor graph export, alist
check-matrix interop, publication-quality visualization, YAML/JSON
configuration, and Docker support — bringing the emulator to the same
operational maturity as established simulators such as Cooja.
The source code and companion benchmark data are archived at
<https://doi.org/10.5281/zenodo.20574329> [@aldaajeh2026qaedata].
Its five-layer architecture decouples code construction, decoding,
noise simulation, benchmark running, and provenance logging, enabling
researchers to add new code instances, custom decoder backends, and
user-defined noise models without modifying any other layer.

# Statement of need

## The reproducibility gap in computational QEC

Published QEC benchmarks typically report logical failure rates as bare
floating-point numbers alongside a table or figure.
A reviewer wishing to verify such a result faces three structural
obstacles: (i) parity-check matrices are usually described algebraically
rather than given as exact bytes, so small construction differences go
undetected; (ii) random seeds are rarely recorded, making exact
replication impossible; and (iii) no automated mechanism exists to
distinguish a genuine zero-failure run from a copy-paste error or seed
collision in a multi-row results table.

These are not hypothetical concerns.
During the development of this package a concrete implementation issue
was encountered: PyMatching v2 [@higgott2023sparse] strictly rejects
parity-check matrix columns with weight greater than two (raises
`ValueError`), yet BB codes have column weight three.
An undisclosed naive matrix-loading approach would crash silently and
produce no output at all.
Detecting and documenting such failures requires that the entire
(matrix, decoder, noise, seed) chain be logged and independently
re-executable — precisely what this package provides.

## Gap relative to existing tools

Stim [@gidney2021stim] provides fast stabiliser circuit sampling but
does not manage BB code construction, provenance, or experiment
lifecycle.
PyMatching [@higgott2023sparse] is a decoding engine, not a benchmark
runner.
The `ldpc` package [@roffe2020decoding] supplies BP+OSD decoding but
has no noise simulation or experiment management layer.
PanQEC [@tuckett2021tailored] offers visualisation tools but does not
record deterministic seeds or matrix fingerprints.
None of these tools provides the complete
(code construction → noise simulation → decoding → provenance logging →
QC check) pipeline in a single auditable package.
QEC Emulator fills this gap.

# Software design

## Five-layer architecture

The package is organised as five independent layers, each with a single
scientific responsibility.
Strict decoupling means that adding a new decoder, noise model, or
benchmark runner requires subclassing or implementing one interface
without modifying any other layer.

**Layer 1 — Code construction (`codes.py`).**
The `BBCode` dataclass constructs bivariate bicycle codes over
$\mathbb{F}_2[x,y]/(x^\ell - 1,\, y^m - 1)$ from generator polynomial
exponents.
Three named constructors are pinned to the companion study parameters:

| Constructor | $({\ell,m})$ | $A$, $B$ | $n$ | $k$ | $H_X$ SHA-256 prefix |
|---|---|---|---|---|---|
| `BB72()`  | $(6,6)$   | $x^3{+}y{+}y^2$,\;$y^3{+}x{+}x^2$ | 72 | 12 | `7ab0973b...` |
| `BB144()` | $(12,6)$  | same | 144 | 12 | `195d5586...` |
| `BB288()` | $(12,12)$ | $x^3{+}y^2{+}y^7$,\;$y^3{+}x{+}x^2$ | 288 | 12 | `c06e721e...` |

Every `BBCode` instance exposes `hx_sha256()` and `hz_sha256()` methods
returning the full 64-character SHA-256 digest of the respective matrix's
byte representation, providing a tamper-evident provenance fingerprint.

**Layer 2 — Decoders (`decoders.py`).**
Three decoder classes share a common abstract base interface
(`decode_z`, `decode_x`, `decode_success`):

- `LookupDecoder`: exact radius-$t$ syndrome lookup; self-contained,
  no external dependencies.
- `BPOSDDecoder`: wraps the `ldpc` package v2 `BpOsdDecoder` class API
  [@roffe2020decoding]. The legacy `bposd_decoder` function segfaults
  in `ldpc` $\ge$ 2.0 and is explicitly avoided.
- `MWPMDecoder`: wraps PyMatching v2 [@higgott2023sparse] with DEM chain
  decomposition. BB codes have column weight three; PyMatching v2 rejects
  direct matrix loading.
  Each weight-3 column $\{a,b,c\}$ is decomposed into two edges
  $\{a,b\}$ and $\{b,c\}$ via `add_edge()`, following the standard
  Stim/DEM approach [@gidney2021stim].
  Global high-weight boundary edges resolve odd-parity syndrome
  components that arise from logical operators.
  `MWPMDecoder` is a lossy reference baseline: the $\{a,c\}$ correlation
  is discarded, and benchmarks in the companion study show it fails
  substantially more often than `BPOSDDecoder` on BB codes [@aldaajeh2026qae].

All decoders fall back gracefully to `LookupDecoder`
(`is_fallback=True`) when optional dependencies are absent, ensuring
provenance records are never silently mislabelled.

**Layer 3 — Noise models (`noise.py`).**
Four models cover the range studied in the companion paper:

- `PhenomenologicalModel`: independent Bernoulli Pauli errors, ideal
  syndrome readout (code-capacity baseline).
- `RepeatedCycleModel`: $d$ syndrome rounds per trial with data and
  measurement noise; returns differential detector syndrome arrays for
  space-time decoders.
- `CircuitLevelModel`: Pauli propagation through the full CNOT schedule
  with gate faults, ancilla noise, and noisy syndrome readout.
- `HardwareModel`: architecture-parameterised single-cycle simulation
  for superconducting CZ and trapped-ion MS backends.

**Layer 4 — Benchmark runners (`benchmark.py`).**
Six runners implement the complete experimental protocols of the companion
study [@aldaajeh2026qae]:

- `run_sweep`: single-round sweep across a $p$-grid.
- `run_repeated_cycle`: proper $d$-round space-time benchmark with
  BP+OSD or MWPM decoding.
- `run_threshold_scan`: multi-code finite-size comparison.
- `run_fixed_weight`: fixed-weight decoder-policy test.
- `run_distance_certificate`: meet-in-the-middle distance lower-bound
  enumeration.
- `run_hardware_ablation` *(v2.0.0)*: eight-condition simulation-based
  ablation study that enables and disables SWAP overhead, idle noise,
  and measurement noise independently to attribute the CZ/MS performance
  gap under specified simulation models.
- `run_uf_decoder` *(v2.0.0)*: Union-Find BFS reference decoder that
  avoids the DEM approximation; benchmarks confirm it fails substantially
  more often than `BPOSDDecoder`, corroborating that the graph-decoder
  inferiority reflects the fundamental weight-3 hyperedge structure of
  BB Tanner graphs, not only the DEM approximation.

**Layer 5 — Provenance (`provenance.py`).**
Every `ProvenanceRecord` stores 15 fields including the SHA-256 digests
of $H_X$ and $H_Z$, the integer seed, raw failure count, Wilson 95\%
upper confidence bound, UTC timestamp, Python/NumPy version, and an
integrity seal:
$$
\texttt{result\_sha256} = \mathrm{SHA256}\!\bigl(
  \mathrm{JSON\_sort}(
    \mathrm{code},\, \mathrm{decoder},\, p,\, n_\text{trials},\,
    \mathrm{seed},\, \mathrm{failures},\, \texttt{hx\_sha256},\,
    \texttt{hz\_sha256})
\bigr).
$$
Any change to the code matrix, seed, trial count, or failure count
produces a completely different digest.
`ProvenanceLogger.unresolved_pairs()` flags any two records sharing an
identical `(code, decoder, p, result_sha256)` tuple, distinguishing
genuine zero-failure runs from seed collisions or copy-paste errors.

## Extension points

Four extension points require no modification to the core package:

1. **New BB code**: `BBCode(ell, m, a_exp, b_exp)`.
2. **New CSS code**: override `hx` and `hz` in a `BBCode` subclass.
3. **New decoder**: subclass `BaseDecoder`; implement `decode_z` and `decode_x`.
4. **New noise model**: implement `(code, p, rng) → (z_mask, x_mask, syn_z, syn_x)`.

Potential research extensions include full finite-size scaling with
$\ge 3$ code sizes and circuit-level noise, hyperedge-aware MWPM
implementations, new hardware backend presets, and networked BB code
simulation following @shaw2026networked.

# Usage examples

## Single-round sweep with provenance

```python
from qec_emulator import BB72, BPOSDDecoder, PhenomenologicalModel
from qec_emulator.benchmark import run_sweep

logger = run_sweep(
    BB72(), BPOSDDecoder(BB72()), PhenomenologicalModel(),
    p_values=[0.004, 0.006, 0.008, 0.010, 0.012],
    trials=10000, seed=20260607,
)
logger.to_csv("bb72_results.csv")
assert len(logger.unresolved_pairs()) == 0   # QC: no seed collisions
```

## Repeated-cycle benchmark across three code sizes

```python
from qec_emulator import BB72, BB144, BB288
from qec_emulator.benchmark import run_repeated_cycle

for code, rounds in [(BB72(), 6), (BB144(), 12), (BB288(), 18)]:
    logger = run_repeated_cycle(
        code, decoder_type='bposd',
        p_values=[0.004, 0.006, 0.008, 0.010, 0.012],
        rounds=rounds, trials=2000, seed=20260607,
    )
    logger.to_json(f"rc_bposd_{code.n}.json")
```

## Hardware ablation study

```python
from qec_emulator import BB72
from qec_emulator.benchmark import run_hardware_ablation

logger = run_hardware_ablation(BB72(), p_base=5e-4, trials=3000)
for rec in logger.records:
    print(f"{rec.noise_model:<40} LFR={rec.lfr:.5f}")
```

## Union-Find BFS reference decoder

```python
from qec_emulator import BB72
from qec_emulator.benchmark import run_uf_decoder

logger = run_uf_decoder(
    BB72(), p_values=[0.004, 0.008, 0.012], trials=10000
)
```

# Testing and continuous integration

The package ships with 56 pytest tests (54 active, 2 correctly skipped
when optional dependencies are absent) covering all five layers:

- **Code layer**: construction parameters, commutation ($H_XH_Z^T=0$),
  encoding dimension $k$, SHA-256 determinism and pinned digests for all
  three BB codes, row and column weight verification.
- **Decoder layer**: zero-syndrome correctness, weight-1 exhaustive
  correctability, ldpc v2 API regression (no segfault), PyMatching v2
  column-weight handling (no `ValueError`), odd-parity syndrome
  resolution, fallback labelling.
- **Noise layer**: zero-$p$ limits, differential array shapes from
  `RepeatedCycleModel`, hardware backend validation.
- **Benchmark layer**: deterministic seeds, `run_repeated_cycle`
  BP+OSD and MWPM, invalid decoder type error, `run_fixed_weight`
  certification of $w\le t$, `run_distance_certificate` BB72 pass,
  `run_hardware_ablation` eight-record output and CZ>pheno ordering,
  `run_uf_decoder` positive LFR at moderate $p$.
- **Provenance layer**: integrity seal sensitivity, CSV/JSON round-trip,
  `unresolved_pairs` detection.

Tests run on Python 3.9, 3.11, and 3.12 via GitHub Actions
(`.github/workflows/ci.yml`).

# Acknowledgements

The author thanks the developers of Stim [@gidney2021stim],
PyMatching [@higgott2023sparse], and the `ldpc` package
[@roffe2020decoding], which are integrated as optional dependencies.
The BB code parameters follow Bravyi et al. [@bravyi2024]; the companion data archive is at [@aldaajeh2026qaedata];
the networked-code architecture context is informed by
Shaw and Rengaswamy [@shaw2026networked].

# References
