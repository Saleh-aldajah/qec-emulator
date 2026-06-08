"""
qec_emulator — Reproducible qLDPC benchmark suite with hash-anchored provenance.

Author : Dr. Saleh H. AlDaajeh
ORCID  : 0000-0001-7810-9290
Contact: S.aldaajeh@gmail.com
Archive: doi:10.5281/zenodo.20574329

What changed in v1.2.3 (QAE study v10)
-----------------------------------------------
* "first publicly documented correct MWPM implementation" claim removed
  from MWPMDecoder docstring; replaced with "provides a documented
  DEM chain decomposition approach."
* "extreme BP+OSD advantage" removed; replaced with "substantial advantage."
* "inadequate for BB codes" removed; replaced with tested-baseline language.
* "graph-decoder inferiority" removed; replaced with "underperformance on
  tested BB instances."
* Version strings in all docstrings updated to v1.2.3 / QAE study v10.
* Emulator kept in continuous sync with manuscript claim language per
  the standing update policy established at v1.2.1.

What changed in v1.2.2 (QAE study v9)
-----------------------------------------------
* generate_minimax_bounds() added — computes i.i.d. stochastic fidelity
  lower bounds, reproduces data/minimax_lower_bounds.csv exactly.
* run_sweep and run_threshold_scan docstrings: CI labels corrected from
  "CP" to "Wilson" (the implementation uses Wilson intervals, not
  Clopper-Pearson; this was a documentation error).
* run_hardware_ablation docstring: precision percentage estimates
  ("≈45%", "≈22%") removed; language now says "simulation-model
  attribution" with explicit Monte Carlo uncertainty caveat.
* run_uf_decoder docstring: "fail catastrophically" and "fundamental"
  removed; replaced with "perform substantially more poorly than BP+OSD."
* run_hardware_ablation: stable_seed() confirmed for all conditions
  (was already using _stable_seed() since v1.2.1).

What changed in v1.2.1 (QAE study v8 — seed stability patch)
-----------------------------------------------
* All benchmark runners that used Python built-in hash() for seed
  derivation now use _stable_seed(*args), a SHA-256-based deterministic
  function that produces identical seeds across Python processes regardless
  of PYTHONHASHSEED.  Affected functions: run_repeated_cycle,
  run_hardware_ablation, run_uf_decoder.
* run_hardware_ablation docstring clarifies simulation-model scope:
  contributions attributed under specified noise parameters only; on-device
  verification with real hardware remains future work.
* run_uf_decoder docstring notes both graph-based references (UF-BFS and
  MWPMDecoder) fail substantially more often than BPOSDDecoder.

What changed in v1.2.0 (QAE study v6)
---------------------------------------
* run_hardware_ablation() added — eight-condition simulation-based ablation
  attributing the CZ/MS performance gap to individual noise sources under
  the specified simulation models (not causal real-hardware experiments).
* run_uf_decoder() added — Union-Find BFS reference decoder that avoids the
  DEM chain approximation; found to perform worse than MWPMDecoder on BB codes,
  confirming the graph-decoder inferiority reflects the fundamental weight-3
  hyperedge structure.
* MWPMDecoder docstring updated: explicitly labelled as a lossy reference
  baseline; UF-BFS finding added to documentation.
* BB288 SHA-256 docstring corrected (c06e721e... was incorrectly cited as
  7b3380... in v1.1.0).
* verify_certificate.py added to the companion archive for exhaustive BB72
  certificate verification (O1–O4), exits non-zero on any failure.
* minimax_lower_bounds.csv regenerated with correct delta=0.1 multiplier.
* verify_hashes.py updated to exit non-zero on missing files or hash mismatches.

What changed in v1.1.0 (QAE study v5)
---------------------------------------
* BB288() added — third code size (ell=12, m=12, A=x^3+y^2+y^7, B=y^3+x+x^2).
* BPOSDDecoder: uses ldpc v2 BpOsdDecoder class API (bposd_decoder segfaults
  in ldpc >= 2.0).
* MWPMDecoder: DEM chain decomposition via add_edge() + global boundary edges.
* RepeatedCycleModel: d rounds + measurement noise, returns differential
  detector syndromes.
* run_repeated_cycle(): space-time BP+OSD or MWPM benchmark runner.
"""
__version__  = "2.0.0"
__author__   = "Dr. Saleh H. AlDaajeh"
__email__    = "S.aldaajeh@gmail.com"
__orcid__    = "0000-0001-7810-9290"
__license__  = "MIT"

from .codes      import BB72, BB144, BB288, Steane7, BBCode
from .decoders   import LookupDecoder, BPOSDDecoder, MWPMDecoder
from .noise      import (PhenomenologicalModel, RepeatedCycleModel,
                          CircuitLevelModel, HardwareModel)
from .provenance import (ProvenanceLogger, ProvenanceRecord,
                          sha256_of_dict, sha256_of_file)
from .benchmark  import (run_sweep, run_repeated_cycle, run_threshold_scan,
                          run_fixed_weight, run_distance_certificate,
                          run_hardware_ablation, run_uf_decoder,
                          generate_minimax_bounds)
from .config        import (SimulationConfig, CodeConfig, NoiseConfig,
                             DecoderConfig, RunnerConfig, ExportConfig,
                             ServerConfig, write_default_config)
from .export        import ResultSet
from .interop       import (to_stim_detector_error_model, to_stim_circuit,
                             to_networkx_tanner, to_networkx_factor,
                             to_alist, from_alist, to_sparse_dict,
                             from_sparse_dict, interop_summary)
from .scheduler     import SimJob, JobScheduler
from .visualization import (plot_lfr_vs_p, plot_threshold,
                             plot_risk_ratios, plot_tanner_graph, sparkline)


__all__ = [
    "BB72", "BB144", "BB288", "Steane7", "BBCode",
    "LookupDecoder", "BPOSDDecoder", "MWPMDecoder",
    "PhenomenologicalModel", "RepeatedCycleModel",
    "CircuitLevelModel", "HardwareModel",
    "ProvenanceLogger", "ProvenanceRecord",
    "sha256_of_dict", "sha256_of_file",
    "run_sweep", "run_repeated_cycle", "run_threshold_scan",
    "run_fixed_weight", "run_distance_certificate",
    "run_hardware_ablation", "run_uf_decoder",
    "generate_minimax_bounds",
    # config
    "SimulationConfig", "write_default_config",
    # export
    "ResultSet",
    # interop
    "to_stim_detector_error_model", "to_stim_circuit",
    "to_networkx_tanner", "to_networkx_factor",
    "to_alist", "from_alist", "interop_summary",
    # scheduler
    "SimJob", "JobScheduler",
    # visualization
    "plot_lfr_vs_p", "plot_threshold", "plot_risk_ratios",
    "plot_tanner_graph", "sparkline",
]
