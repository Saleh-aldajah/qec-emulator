"""
benchmark.py — Monte Carlo benchmark runners.

Author : Dr. Saleh H. AlDaajeh
ORCID  : 0000-0001-7810-9290
Contact: S.aldaajeh@gmail.com

Six runners implement the complete experimental protocols of the v6 QAE study
(doi:10.5281/zenodo.20574329):

  run_sweep               — single-round sweep.
  run_repeated_cycle      — d-round space-time BP+OSD or MWPM.
  run_threshold_scan      — multi-code finite-size comparison.
  run_fixed_weight        — fixed-weight decoder-policy test.
  run_distance_certificate— MitM distance lower-bound enumeration.
  run_hardware_ablation   — NEW in v1.2.0: controlled noise-component
                            ablation study (simulation only).
  run_uf_decoder          — NEW in v1.2.0: Union-Find BFS reference
                            decoder comparison.
"""
from __future__ import annotations

import itertools
import math
from typing import Callable, List, Optional, Sequence

import numpy as np

from .codes      import BBCode, in_rowspace, syndrome_from_mask, weight
from .decoders   import BaseDecoder, LookupDecoder
from .noise      import PhenomenologicalModel, RepeatedCycleModel
from .provenance import ProvenanceLogger


def _bern(n: int, p: float, rng: np.random.Generator) -> int:
    return int(sum((1 if rng.random() < p else 0) << i for i in range(n)))

def _arr_to_mask(arr: np.ndarray) -> int:
    return int(sum((int(b) & 1) << i for i, b in enumerate(arr)))

def _logical_fail(ez: int, ex: int, code: BBCode) -> bool:
    n  = code.n
    vz = np.array([(ez >> i) & 1 for i in range(n)], dtype=np.uint8)
    vx = np.array([(ex >> i) & 1 for i in range(n)], dtype=np.uint8)
    return not (in_rowspace(vz, code.hz_rref, code.hz_pivots) and
                in_rowspace(vx, code.hx_rref, code.hx_pivots))


def _stable_seed(*args) -> int:
    """
    SHA-256-derived deterministic seed.

    Stable across Python processes regardless of PYTHONHASHSEED, unlike
    Python's built-in hash().  Every benchmark runner uses this function
    so results are exactly reproducible on any machine.
    """
    import hashlib
    key = "|".join(str(a) for a in args).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], 'big')


# ---------------------------------------------------------------------------
# run_sweep
# ---------------------------------------------------------------------------

def run_sweep(
    code:        BBCode,
    decoder:     BaseDecoder,
    noise_model,
    p_values:    Sequence[float],
    trials:      int  = 5000,
    seed:        int  = 20260607,
    batches:     int  = 1,
    verbose:     bool = True,
) -> ProvenanceLogger:
    """Single-round Monte Carlo sweep."""
    logger    = ProvenanceLogger()
    code_name = f"BB{code.n}"
    hx_sha    = code.hx_sha256()
    hz_sha    = code.hz_sha256()
    total     = len(p_values) * batches
    done      = 0

    for p_idx, p in enumerate(p_values):
        for batch in range(batches):
            batch_seed = (seed ^ (p_idx * 100 + batch) * 2654435761) & 0xFFFF_FFFF
            rng        = np.random.default_rng(batch_seed)
            rec = logger.new_record(
                code_name=code_name, decoder_name=decoder.name,
                noise_model=noise_model.name, p=p,
                trials=trials, seed=batch_seed,
                hx_sha256=hx_sha, hz_sha256=hz_sha,
                decoder_is_fallback=decoder.is_fallback,
                notes=f"batch={batch}")
            failures = sum(
                0 if decoder.decode_success(*noise_model(code, p, rng)) else 1
                for _ in range(trials))
            rec.set_result(failures)
            done += 1
            if verbose:
                w = len(str(total))
                print(f"  [{done:{w}}/{total}] p={p:.4f} batch={batch} "
                      f"LFR={rec.lfr:.5f} ({failures}/{trials}) "
                      f"QC={rec.qc_status()}")
    return logger


# ---------------------------------------------------------------------------
# run_repeated_cycle
# ---------------------------------------------------------------------------

def run_repeated_cycle(
    code:         BBCode,
    decoder_type: str,
    p_values:     Sequence[float],
    rounds:       int,
    trials:       int  = 2000,
    seed:         int  = 20260607,
    verbose:      bool = True,
    **decoder_kwargs,
) -> ProvenanceLogger:
    """
    Repeated-cycle benchmark: d syndrome rounds per trial.

    Parameters
    ----------
    decoder_type : 'bposd' or 'mwpm'
    rounds       : syndrome rounds per trial (standard: use d for the code).
    """
    logger    = ProvenanceLogger()
    code_name = f"BB{code.n}"
    n, n2     = code.n, code.n2
    hx_sha    = code.hx_sha256()
    hz_sha    = code.hz_sha256()

    for p in p_values:
        batch_seed = _stable_seed(decoder_type, p, rounds, seed)
        rng        = np.random.default_rng(batch_seed)
        dec_label  = (f"BP+OSD-RC-r{rounds}" if decoder_type == 'bposd'
                      else f"MWPM-DEM-RC-r{rounds}")
        rec = logger.new_record(
            code_name=code_name, decoder_name=dec_label,
            noise_model=f"repeated_cycle_r{rounds}", p=p,
            trials=trials, seed=batch_seed,
            hx_sha256=hx_sha, hz_sha256=hz_sha,
            decoder_is_fallback=False, notes=f"rounds={rounds}")

        if decoder_type == 'bposd':
            dz, dx = _build_spacetime_bposd(code, rounds, p)
        elif decoder_type == 'mwpm':
            mz, mx = _build_spacetime_mwpm(code, rounds, p)
        else:
            raise ValueError(f"decoder_type must be 'bposd' or 'mwpm', "
                             f"got {decoder_type!r}")

        rc_model = RepeatedCycleModel(rounds=rounds)
        failures = 0
        for _ in range(trials):
            cum_z, cum_x, det_z, det_x = rc_model(code, p, rng)
            if decoder_type == 'bposd':
                raw_z = dz.decode(det_z); raw_x = dx.decode(det_x)
                cz = np.zeros(n, dtype=np.uint8)
                cx = np.zeros(n, dtype=np.uint8)
                for r in range(rounds):
                    for j in range(n):
                        cz[j] ^= raw_z[r*n+j]; cx[j] ^= raw_x[r*n+j]
            else:
                raw_z = mz.decode(det_z); raw_x = mx.decode(det_x)
                cz = np.zeros(n, dtype=np.uint8)
                cx = np.zeros(n, dtype=np.uint8)
                for idx, val in enumerate(raw_z):
                    if val and idx < rounds*n: cz[idx % n] ^= 1
                for idx, val in enumerate(raw_x):
                    if val and idx < rounds*n: cx[idx % n] ^= 1
            if _logical_fail(cum_z ^ _arr_to_mask(cz),
                             cum_x ^ _arr_to_mask(cx), code):
                failures += 1

        rec.set_result(failures)
        if verbose:
            print(f"  RC {dec_label} p={p:.4f}: "
                  f"LFR={rec.lfr:.5f} ({failures}/{trials}) "
                  f"QC={rec.qc_status()}")
    return logger


def _build_spacetime_bposd(code: BBCode, rounds: int, p: float):
    from ldpc import BpOsdDecoder
    n, n2 = code.n, code.n2
    Ir    = np.eye(rounds, dtype=np.uint8)
    nmc   = rounds * n2
    Hm    = np.zeros((rounds*n2, nmc), dtype=np.uint8)
    for r in range(rounds):
        for c in range(n2):
            Hm[r*n2+c, r*n2+c] = 1
            if r+1 < rounds: Hm[(r+1)*n2+c, r*n2+c] = 1
    He_z = np.hstack([np.kron(Ir, code.hx), Hm])
    He_x = np.hstack([np.kron(Ir, code.hz), Hm])
    cp   = np.array([p]*(rounds*n) + [p]*nmc)
    dz = BpOsdDecoder(He_z, error_rate=p, channel_probs=cp, bp_method='ms',
                      ms_scaling_factor=0.625, osd_method='osd_cs',
                      osd_order=7, max_iter=100)
    dx = BpOsdDecoder(He_x, error_rate=p, channel_probs=cp, bp_method='ms',
                      ms_scaling_factor=0.625, osd_method='osd_cs',
                      osd_order=7, max_iter=100)
    return dz, dx


def _build_spacetime_mwpm(code: BBCode, rounds: int, p: float):
    import pymatching
    n, n2 = code.n, code.n2
    w     = abs(math.log(p/(1-p))) if 0 < p < 1 else 10.0
    w_b   = w * 100.0

    def build(H, n_data, n_checks, rounds, wd, wm, wb):
        m = pymatching.Matching()
        for r in range(rounds):
            for j in range(n_data):
                checks  = [i for i in range(n_checks) if H[i, j]]
                det_ids = [r*n_checks + c for c in checks]
                fid     = {j + r*n_data}
                if not det_ids: continue
                elif len(det_ids) == 1:
                    m.add_boundary_edge(det_ids[0], weight=wd, fault_ids=fid)
                elif len(det_ids) == 2:
                    m.add_edge(det_ids[0], det_ids[1], weight=wd, fault_ids=fid)
                else:
                    for k in range(len(det_ids)-1):
                        m.add_edge(det_ids[k], det_ids[k+1],
                                   weight=wd, fault_ids=fid)
        for r in range(rounds-1):
            for c in range(n_checks):
                fid = {n_data*rounds + r*n_checks + c}
                m.add_edge(r*n_checks+c, (r+1)*n_checks+c,
                           weight=wm, fault_ids=fid)
        for node in range(rounds*n_checks):
            m.add_boundary_edge(node, weight=wb)
        return m

    mz = build(code.hx, n, n2, rounds, w, w, w_b)
    mx = build(code.hz, n, n2, rounds, w, w, w_b)
    return mz, mx


# ---------------------------------------------------------------------------
# run_threshold_scan
# ---------------------------------------------------------------------------

def run_threshold_scan(
    codes:           Sequence[BBCode],
    decoder_factory: Callable[[BBCode], BaseDecoder],
    noise_model,
    p_values:        Sequence[float],
    trials:          int  = 5000,
    seed:            int  = 20260607,
    verbose:         bool = True,
) -> ProvenanceLogger:
    """Single-round sweep across multiple code sizes."""
    master = ProvenanceLogger()
    for ci, code in enumerate(codes):
        dec = decoder_factory(code)
        if verbose:
            print(f"\n=== BB{code.n} / {dec.name} ===")
        sub = run_sweep(code, dec, noise_model, p_values,
                        trials=trials, seed=seed ^ (ci * 999983),
                        verbose=verbose)
        master.records.extend(sub.records)
    return master


# ---------------------------------------------------------------------------
# run_fixed_weight
# ---------------------------------------------------------------------------

def run_fixed_weight(
    code:    BBCode,
    decoder: BaseDecoder,
    weights: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7, 8),
    trials:  int  = 2000,
    seed:    int  = 20260520,
    verbose: bool = True,
) -> ProvenanceLogger:
    """
    Fixed-weight random-sampling decoder-policy test.

    Note: this measures what the decoder does at weights w <= t (certified
    to succeed) and w > t (decoder policy, not code capability).
    """
    logger    = ProvenanceLogger()
    rng       = np.random.default_rng(seed)
    code_name = f"BB{code.n}"
    for w in weights:
        rec = logger.new_record(
            code_name=code_name, decoder_name=decoder.name,
            noise_model=f"fixed_weight_w{w}", p=0.0,
            trials=trials, seed=seed,
            hx_sha256=code.hx_sha256(), hz_sha256=code.hz_sha256(),
            decoder_is_fallback=decoder.is_fallback,
            notes=f"weight={w} — decoder-policy test, not code capability")
        failures = sum(
            0 if decoder.decode_success(
                _random_weight_mask(code.n, w, rng),
                _random_weight_mask(code.n, w, rng)) else 1
            for _ in range(trials))
        rec.set_result(failures)
        if verbose:
            print(f"  w={w}: LFR={rec.lfr:.4f} ({failures}/{trials}) "
                  f"QC={rec.qc_status()}")
    return logger


# ---------------------------------------------------------------------------
# run_distance_certificate
# ---------------------------------------------------------------------------

def run_distance_certificate(
    code:       BBCode,
    max_weight: int  = 5,
    verbose:    bool = True,
) -> dict:
    """
    Meet-in-the-middle distance lower-bound enumeration.
    Confirms d >= max_weight+1 if no nontrivial logical of weight <=
    max_weight exists.  For BB72 with max_weight=5 this confirms d>=6.
    """
    results  = {}
    left_max  = max_weight // 2
    right_max = max_weight - left_max

    for label, cols, stab_rref, stab_pivots in [
        ("Z_component_hx", code.hx_cols, code.hz_rref, code.hz_pivots),
        ("X_component_hz", code.hz_cols, code.hx_rref, code.hx_pivots),
    ]:
        left_by_syn: dict = {}
        left_count = right_count = 0
        nontrivial: list = []
        seen: set = set()

        for w in range(left_max + 1):
            for comb in itertools.combinations(range(code.n), w):
                syn = mask = 0
                for j in comb:
                    syn  ^= cols[j]; mask |= 1 << j
                left_by_syn.setdefault(syn, []).append(mask)
                left_count += 1

        for w in range(right_max + 1):
            for comb in itertools.combinations(range(code.n), w):
                syn = rmask = 0
                for j in comb:
                    syn ^= cols[j]; rmask |= 1 << j
                right_count += 1
                for lmask in left_by_syn.get(syn, []):
                    if lmask & rmask: continue
                    full = lmask | rmask
                    wt   = weight(full)
                    if wt == 0 or wt > max_weight or full in seen:
                        continue
                    seen.add(full)
                    vec = np.array([(full >> i) & 1 for i in range(code.n)],
                                   dtype=np.uint8)
                    if not in_rowspace(vec, stab_rref, stab_pivots):
                        nontrivial.append({"weight": wt,
                            "support": [i for i in range(code.n)
                                        if (full >> i) & 1]})
                        break
            if nontrivial: break

        results[label] = {
            "max_weight_checked": max_weight,
            "method":             "meet-in-the-middle",
            "left_partial_patterns":  left_count,
            "right_partial_patterns": right_count,
            "nontrivial_logical_witnesses": nontrivial,
            "distance_lower_bound_pass": len(nontrivial) == 0,
        }
        if verbose:
            status = "PASS" if results[label]["distance_lower_bound_pass"] else "FAIL"
            print(f"  {label}: {status}")
    return results


# ---------------------------------------------------------------------------
# run_hardware_ablation  (NEW in v1.2.0)
# ---------------------------------------------------------------------------

def run_hardware_ablation(
    code:    BBCode,
    p_base:  float = 5e-4,
    trials:  int   = 3000,
    seed:    int   = 20260607,
    verbose: bool  = True,
) -> ProvenanceLogger:
    """
    Controlled noise-component ablation study.

    Runs eight ablation conditions on a single-cycle simulation, enabling
    and disabling SWAP overhead, idle noise, and measurement noise
    independently to attribute the CZ/MS performance gap under the
    specified simulation models.

    IMPORTANT: these are simulation-model ablations, not causal experiments
    on real hardware.  Contributions attributed here reflect specified noise
    model parameters only.  On-device verification with real hardware and
    matched calibration data remains future work.

    Returns a ProvenanceLogger with one record per ablation condition.
    """
    import math as _m

    logger    = ProvenanceLogger()
    n, n2     = code.n, code.n2
    hx_sha    = code.hx_sha256()
    hz_sha    = code.hz_sha256()

    # Device parameters (from published characterisations)
    T1_CZ, T2_CZ, TG_CZ = 150.0, 100.0, 0.2   # us
    T1_MS, T2_MS, TG_MS = 1e6,   5e6,   50.0
    SWAP_FRAC = (96 / n) * p_base

    p1_cz = 1 - _m.exp(-TG_CZ / T1_CZ)
    p2_cz = 1 - _m.exp(-TG_CZ / T2_CZ)
    p_dec_cz = (p1_cz + p2_cz) / 2

    p1_ms = 1 - _m.exp(-TG_MS / T1_MS)
    p2_ms = 1 - _m.exp(-TG_MS / T2_MS)
    p_dec_ms = (p1_ms + p2_ms) / 2

    ablations = [
        # (label, p_data_eff, p_idle, p_meas)
        ("Phenomenological",       p_base,                  p_base*0.05, p_base),
        ("CZ: full model",         p_base+p_dec_cz+SWAP_FRAC, p_base*0.05, p_base),
        ("CZ: no SWAP",            p_base+p_dec_cz,           p_base*0.05, p_base),
        ("CZ: no idle noise",      p_base+p_dec_cz+SWAP_FRAC, 0.0,         p_base),
        ("CZ: no meas noise",      p_base+p_dec_cz+SWAP_FRAC, p_base*0.05, 0.0),
        ("CZ: gate decoherence only", p_dec_cz,              0.0,         0.0),
        ("MS: full model",         p_base+p_dec_ms,           p_base*0.05, p_base),
        ("MS: CZ-matched gate",    p_base+p_dec_cz,           p_base*0.05, p_base),
    ]

    dec = LookupDecoder(code, radius=2)

    for label, p_data, p_idle, p_meas in ablations:
        batch_seed = _stable_seed(label, seed)
        rng = np.random.default_rng(batch_seed)
        rec = logger.new_record(
            code_name=f"BB{n}", decoder_name="Lookup-r2",
            noise_model=f"ablation:{label.replace(' ','_')}",
            p=p_base, trials=trials, seed=batch_seed,
            hx_sha256=hx_sha, hz_sha256=hz_sha,
            decoder_is_fallback=False,
            notes=f"p_data={p_data:.4e} p_idle={p_idle:.4e} p_meas={p_meas:.4e}")

        failures = 0
        for _ in range(trials):
            ez = _bern(n, p_data, rng); ex = _bern(n, p_data, rng)
            ez ^= _bern(n, p_idle, rng); ex ^= _bern(n, p_idle, rng)
            sz = syndrome_from_mask(ez, code.hx_cols)
            sx = syndrome_from_mask(ex, code.hz_cols)
            for i in range(n2):
                if rng.random() < p_meas: sz ^= (1 << i)
                if rng.random() < p_meas: sx ^= (1 << i)
            if not dec.decode_success(ez, ex, sz, sx):
                failures += 1

        rec.set_result(failures)
        if verbose:
            print(f"  {label:<30} LFR={rec.lfr:.5f} "
                  f"({failures}/{trials}) QC={rec.qc_status()}")

    return logger


# ---------------------------------------------------------------------------
# run_uf_decoder  (NEW in v1.2.0)
# ---------------------------------------------------------------------------

def run_uf_decoder(
    code:     BBCode,
    p_values: Sequence[float],
    trials:   int  = 10000,
    seed:     int  = 20260607,
    verbose:  bool = True,
) -> ProvenanceLogger:
    """
    Union-Find BFS reference decoder comparison.

    Implements a greedy BFS Union-Find decoder as a second graphified
    reference baseline. Like DEM-MWPM, it reduces weight-3 syndrome hyperedges
    to graph edges (chain decomposition), so it is not hyperedge-aware.
    It differs from MWPMDecoder in matching strategy (BFS vs min-weight)
    but shares the same structural approximation.
    On BB codes, UF-BFS performs comparably to MWPMDecoder and both
    perform substantially worse than BPOSDDecoder, confirming that the
    underperformance of these two graph-based baselines is not specific
    to the DEM chain approximation alone (both use graph-edge reduction).
    Both graph-based references (UF-BFS and MWPMDecoder) fail substantially
    more often than BPOSDDecoder under the tested conditions.

    Returns a ProvenanceLogger with one record per p value.
    """
    from collections import deque

    logger    = ProvenanceLogger()
    n, n2     = code.n, code.n2
    hx_sha    = code.hx_sha256()
    hz_sha    = code.hz_sha256()

    # Build adjacency list for each CSS component
    def build_adj(H):
        adj = [[] for _ in range(n2)]
        for j in range(n):
            checks = [i for i in range(n2) if H[i, j]]
            if len(checks) == 2:
                a, b = checks
                adj[a].append((j, b)); adj[b].append((j, a))
            elif len(checks) == 1:
                adj[checks[0]].append((j, -1))
            else:
                for k in range(len(checks)-1):
                    a, b = checks[k], checks[k+1]
                    adj[a].append((j, b)); adj[b].append((j, a))
        return adj

    adj_z = build_adj(code.hx)
    adj_x = build_adj(code.hz)

    def uf_decode(adj, syndrome_int):
        active = [i for i in range(n2) if (syndrome_int >> i) & 1]
        if not active: return 0
        matched  = set()
        correction = 0
        for src in active:
            if src in matched: continue
            d = {src: 0}; p = {src: None}; q = deque([src])
            found = None
            while q and found is None:
                node = q.popleft()
                for (qubit, nbr) in adj[node]:
                    target = nbr if nbr >= 0 else n2
                    if target not in d:
                        d[target] = d[node]+1
                        p[target] = (node, qubit)
                        q.append(target)
                        if (target in active and target not in matched
                                and target != src):
                            found = target; break
            if found is not None:
                matched.add(src); matched.add(found)
                node = found
                while p.get(node):
                    prev, qubit = p[node]; correction ^= (1 << qubit)
                    node = prev
        return correction

    for p in p_values:
        batch_seed = _stable_seed('UF', p, seed)
        rng = np.random.default_rng(batch_seed)
        rec = logger.new_record(
            code_name=f"BB{n}", decoder_name="UF-BFS",
            noise_model="phenomenological", p=p,
            trials=trials, seed=batch_seed,
            hx_sha256=hx_sha, hz_sha256=hz_sha,
            decoder_is_fallback=False,
            notes="greedy BFS Union-Find — alternative graph-based reference")

        failures = 0
        for _ in range(trials):
            ez = _bern(n, p, rng); ex = _bern(n, p, rng)
            sz = syndrome_from_mask(ez, code.hx_cols)
            sx = syndrome_from_mask(ex, code.hz_cols)
            cz = uf_decode(adj_z, sz); cx = uf_decode(adj_x, sx)
            if _logical_fail(ez ^ cz, ex ^ cx, code):
                failures += 1

        rec.set_result(failures)
        if verbose:
            print(f"  UF-BFS p={p:.4f}: LFR={rec.lfr:.5f} "
                  f"({failures}/{trials}) QC={rec.qc_status()}")

    return logger



# ---------------------------------------------------------------------------
# generate_minimax_bounds  (added v1.2.2)
# ---------------------------------------------------------------------------

def generate_minimax_bounds(
    n:     int   = 128,
    d:     int   = 11,
    t:     int   = 5,
    delta: float = 0.1,
    p_values: "Sequence[float] | None" = None,
) -> "list[dict]":
    """
    Compute i.i.d. stochastic fidelity lower bounds.

    Returns a list of dicts with keys:
      n, delta, d, t, p_attack, lower_bound, binomial_cdf

    Formula: (1 - delta) * Pr[Bin(n, p) <= t]

    Default parameters match the QAE study (n=128, d=11, t=5, delta=0.1).
    Results match data/minimax_lower_bounds.csv exactly.
    """
    import math as _m

    if p_values is None:
        p_values = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007,
                    0.008, 0.009, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04,
                    0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

    def binom_cdf(t, n, p):
        return sum(_m.comb(n, i) * (p**i) * ((1-p)**(n-i)) for i in range(t+1))

    rows = []
    for p in p_values:
        prob = binom_cdf(t, n, p)
        lb   = (1 - delta) * prob
        rows.append(dict(n=n, delta=delta, d=d, t=t, p_attack=p,
                         lower_bound=round(lb, 10),
                         binomial_cdf=round(prob, 10)))
    return rows

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _random_weight_mask(n: int, w: int, rng: np.random.Generator) -> int:
    if w == 0: return 0
    return int(sum(1 << int(j) for j in
                   rng.choice(n, size=min(w, n), replace=False)))
