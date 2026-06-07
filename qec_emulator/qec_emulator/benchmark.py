"""
benchmark.py — High-level Monte Carlo benchmark runners.

All functions return a ProvenanceLogger containing fully hash-anchored
records that can be written to CSV/JSON for manuscript submission.

Quick start
-----------
    from qec_emulator import BB72, LookupDecoder, PhenomenologicalModel
    from qec_emulator.benchmark import run_sweep

    code    = BB72()
    decoder = LookupDecoder(code)
    noise   = PhenomenologicalModel()

    logger = run_sweep(
        code, decoder, noise,
        p_values=[0.001, 0.002, 0.004, 0.006, 0.008],
        trials=5000,
        seed=20260606,
    )
    logger.to_csv("results/bb72_sweep.csv")
    print(logger.summary())
"""
from __future__ import annotations

import time
from typing import Callable, List, Optional, Sequence, Union

import numpy as np

from .codes import BBCode
from .decoders import BaseDecoder
from .noise import PhenomenologicalModel
from .provenance import ProvenanceLogger, ProvenanceRecord


# ---------------------------------------------------------------------------
# Core sweep runner
# ---------------------------------------------------------------------------

def run_sweep(
    code: BBCode,
    decoder: BaseDecoder,
    noise_model,
    p_values: Sequence[float],
    trials: int = 5000,
    seed: int = 20260606,
    batches: int = 1,
    verbose: bool = True,
) -> ProvenanceLogger:
    """
    Run Monte Carlo benchmark across a grid of physical error rates.

    Parameters
    ----------
    code : BBCode
        The quantum code to benchmark.
    decoder : BaseDecoder
        Decoder instance (LookupDecoder, BPOSDDecoder, MWPMDecoder).
    noise_model : callable
        Noise model instance (PhenomenologicalModel, CircuitLevelModel, etc.).
    p_values : sequence of float
        Physical error rates to sweep.
    trials : int
        Number of Monte Carlo shots per (p, batch) point.
    seed : int
        Base seed. Each (p, batch) point uses seed ^ hash(p_index * 100 + batch).
    batches : int
        Number of independent batches per p value (use ≥ 2 to detect seed collisions).
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    ProvenanceLogger
        Contains one ProvenanceRecord per (p, batch) pair.
    """
    logger = ProvenanceLogger()
    code_name = f"BB{code.n}" if hasattr(code, "n") else "Unknown"
    hx_sha = code.hx_sha256()
    hz_sha = code.hz_sha256()

    total = len(p_values) * batches
    done = 0

    for p_idx, p in enumerate(p_values):
        for batch in range(batches):
            batch_seed = (seed ^ (p_idx * 100 + batch) * 2654435761) & 0xFFFFFFFF
            rng = np.random.default_rng(batch_seed)

            rec = logger.new_record(
                code_name=code_name,
                decoder_name=decoder.name,
                noise_model=noise_model.name,
                p=p,
                trials=trials,
                seed=batch_seed,
                hx_sha256=hx_sha,
                hz_sha256=hz_sha,
                decoder_is_fallback=decoder.is_fallback,
                notes=f"batch={batch}",
            )

            failures = 0
            for _ in range(trials):
                z_mask, x_mask, syn_z, syn_x = noise_model(code, p, rng)
                ok = decoder.decode_success(z_mask, x_mask, syn_z, syn_x)
                if not ok:
                    failures += 1

            rec.set_result(failures)
            done += 1

            if verbose:
                print(
                    f"  [{done:>{len(str(total))}}/{total}] "
                    f"p={p:.4f} batch={batch} "
                    f"LFR={rec.lfr:.5f} "
                    f"({failures}/{trials}) "
                    f"QC={rec.qc_status()}"
                )

    return logger


# ---------------------------------------------------------------------------
# Threshold scan
# ---------------------------------------------------------------------------

def run_threshold_scan(
    codes: Sequence[BBCode],
    decoder_factory: Callable[[BBCode], BaseDecoder],
    noise_model,
    p_values: Sequence[float],
    trials: int = 5000,
    seed: int = 20260606,
    verbose: bool = True,
) -> ProvenanceLogger:
    """
    Sweep multiple code sizes to identify a finite-size threshold crossing.

    Parameters
    ----------
    codes : sequence of BBCode
        Typically [BB72(), BB144()] for a two-code crossing analysis.
    decoder_factory : callable
        Function that takes a BBCode and returns a decoder. Required because
        each code needs its own decoder instance.
    noise_model, p_values, trials, seed, verbose
        Same as run_sweep.

    Returns
    -------
    ProvenanceLogger
        Records for all (code, p) combinations.
    """
    master_logger = ProvenanceLogger()
    for code_idx, code in enumerate(codes):
        decoder = decoder_factory(code)
        if verbose:
            code_name = f"BB{code.n}"
            print(f"\n=== {code_name} / {decoder.name} ===")
        sub_logger = run_sweep(
            code, decoder, noise_model, p_values,
            trials=trials,
            seed=seed ^ (code_idx * 999983),
            verbose=verbose,
        )
        master_logger.records.extend(sub_logger.records)
    return master_logger


# ---------------------------------------------------------------------------
# Fixed-weight adversarial experiment
# ---------------------------------------------------------------------------

def run_fixed_weight(
    code: BBCode,
    decoder: BaseDecoder,
    weights: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7, 8),
    trials: int = 2000,
    seed: int = 20260520,
    verbose: bool = True,
) -> ProvenanceLogger:
    """
    Test decoding success rate for fixed-weight random error patterns.

    Rows with weight ≤ ⌊(d-1)/2⌋ should achieve 100% success for a code
    with certified distance d.
    """
    logger = ProvenanceLogger()
    rng = np.random.default_rng(seed)
    code_name = f"BB{code.n}"
    hx_sha = code.hx_sha256()
    hz_sha = code.hz_sha256()

    for w in weights:
        rec = logger.new_record(
            code_name=code_name,
            decoder_name=decoder.name,
            noise_model=f"fixed_weight_w{w}",
            p=0.0,
            trials=trials,
            seed=seed,
            hx_sha256=hx_sha,
            hz_sha256=hz_sha,
            decoder_is_fallback=decoder.is_fallback,
            notes=f"weight={w}",
        )
        failures = 0
        for _ in range(trials):
            z_mask = _random_weight_mask(code.n, w, rng)
            x_mask = _random_weight_mask(code.n, w, rng)
            if not decoder.decode_success(z_mask, x_mask):
                failures += 1
        rec.set_result(failures)
        if verbose:
            print(f"  w={w}: LFR={rec.lfr:.4f} ({failures}/{trials}) QC={rec.qc_status()}")

    return logger


# ---------------------------------------------------------------------------
# Certificate verification runner
# ---------------------------------------------------------------------------

def run_distance_certificate(
    code: BBCode,
    max_weight: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Enumerate all zero-syndrome vectors through max_weight using meet-in-the-middle.

    Returns a certificate dict compatible with bb72_certificate_report.json.
    """
    from .codes import in_rowspace, syndrome_from_mask, weight
    import itertools

    results = {}
    left_max = max_weight // 2
    right_max = max_weight - left_max

    for label, cols, stab_rref, stab_pivots in [
        ("Z_component_hx", code.hx_cols, code.hz_rref, code.hz_pivots),
        ("X_component_hz", code.hz_cols, code.hx_rref, code.hx_pivots),
    ]:
        left_by_syn: dict = {}
        left_count = right_count = 0
        nontrivial = []
        zero_masks: set = set()

        for w in range(left_max + 1):
            for comb in itertools.combinations(range(code.n), w):
                syn = mask = 0
                for j in comb:
                    syn ^= cols[j]
                    mask |= 1 << j
                left_by_syn.setdefault(syn, []).append(mask)
                left_count += 1

        for w in range(right_max + 1):
            for comb in itertools.combinations(range(code.n), w):
                syn = rmask = 0
                for j in comb:
                    syn ^= cols[j]
                    rmask |= 1 << j
                right_count += 1
                for lmask in left_by_syn.get(syn, []):
                    if lmask & rmask:
                        continue
                    full = lmask | rmask
                    wt = bin(full).count("1")
                    if wt == 0 or wt > max_weight or full in zero_masks:
                        continue
                    zero_masks.add(full)
                    vec = np.array([(full >> i) & 1 for i in range(code.n)], dtype=np.uint8)
                    if not in_rowspace(vec, stab_rref, stab_pivots):
                        nontrivial.append({"weight": wt, "support": [i for i in range(code.n) if (full >> i) & 1]})
                        break
            if nontrivial:
                break

        results[label] = {
            "max_weight_checked": max_weight,
            "method": "exact meet-in-the-middle split ≤2 plus ≤3",
            "left_partial_patterns": left_count,
            "right_partial_patterns": right_count,
            "unique_zero_syndrome_vectors_weight_leq_max": len(zero_masks),
            "nontrivial_logical_witnesses": nontrivial,
            "distance_lower_bound_pass": len(nontrivial) == 0,
        }
        if verbose:
            status = "PASS" if results[label]["distance_lower_bound_pass"] else "FAIL"
            print(f"  {label}: {status}")

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_weight_mask(n: int, w: int, rng: np.random.Generator) -> int:
    if w == 0:
        return 0
    choices = rng.choice(n, size=min(w, n), replace=False)
    return int(sum(1 << int(j) for j in choices))
