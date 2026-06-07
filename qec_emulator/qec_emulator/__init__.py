"""
qec_emulator — Reproducible qLDPC benchmark suite with hash-anchored provenance.

Quick start
-----------
    pip install qec-emulator

    from qec_emulator import BB72, LookupDecoder, PhenomenologicalModel
    from qec_emulator.benchmark import run_sweep

    code    = BB72()
    decoder = LookupDecoder(code)
    noise   = PhenomenologicalModel()

    logger = run_sweep(code, decoder, noise, p_values=[0.001, 0.004, 0.008], trials=5000)
    logger.to_csv("results.csv")
    print(logger.summary())

Cite as
-------
    [Author]. (2026). QEC Emulator (Version 1.0.0). Zenodo.
    https://doi.org/10.5281/zenodo.XXXXXXX
"""

__version__ = "1.0.0"
__author__ = "QAE Study Authors"
__license__ = "MIT"

# Codes
from .codes import BB72, BB144, Steane7, BBCode

# Decoders
from .decoders import LookupDecoder, BPOSDDecoder, MWPMDecoder

# Noise models
from .noise import PhenomenologicalModel, CircuitLevelModel, HardwareModel

# Provenance
from .provenance import ProvenanceLogger, ProvenanceRecord, sha256_of_dict, sha256_of_file

# Benchmark runners (also importable from qec_emulator.benchmark)
from .benchmark import run_sweep, run_threshold_scan, run_fixed_weight, run_distance_certificate

__all__ = [
    # codes
    "BB72", "BB144", "Steane7", "BBCode",
    # decoders
    "LookupDecoder", "BPOSDDecoder", "MWPMDecoder",
    # noise
    "PhenomenologicalModel", "CircuitLevelModel", "HardwareModel",
    # provenance
    "ProvenanceLogger", "ProvenanceRecord", "sha256_of_dict", "sha256_of_file",
    # benchmark
    "run_sweep", "run_threshold_scan", "run_fixed_weight", "run_distance_certificate",
]
