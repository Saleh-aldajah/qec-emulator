"""
noise.py — Noise models for Monte Carlo QEC experiments.

Author : Dr. Saleh H. AlDaajeh
ORCID  : 0000-0001-7810-9290
Contact: S.aldaajeh@gmail.com

Four noise models are provided, covering the range studied in the v5
QAE paper (doi:10.5281/zenodo.20574329):

PhenomenologicalModel   — single-round independent Bernoulli errors,
                          ideal syndrome readout.

RepeatedCycleModel      — NEW in v5: d syndrome rounds per trial with
                          independent data errors AND measurement errors
                          each round, plus a noiseless final round.
                          Returns the differential detector syndrome
                          needed by space-time decoders.

CircuitLevelModel       — single-cycle Pauli propagation through the
                          BB syndrome-extraction CNOT schedule with
                          noisy ancilla prep, gate faults, idle errors,
                          and noisy syndrome measurement.

HardwareModel           — architecture-parameterised (CZ or MS backend)
                          single-cycle compiled simulation.

All models are callable: model(code, p, rng) -> (z_mask, x_mask, syn_z, syn_x).
Syndrome overrides are None for ideal-readout models.
For RepeatedCycleModel the return is instead the differential detector
arrays needed by the space-time decoder.
"""
from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np

from .codes import BBCode, syndrome_from_mask

NoiseResult = Tuple[int, int, Optional[int], Optional[int]]


# ---------------------------------------------------------------------------
# Bernoulli mask helper
# ---------------------------------------------------------------------------

def _bernoulli_mask(n: int, p: float, rng: np.random.Generator) -> int:
    return int(sum((1 if rng.random() < p else 0) << i for i in range(n)))

def _arr_to_mask(arr: np.ndarray) -> int:
    return int(sum((int(b) & 1) << i for i, b in enumerate(arr)))


# ---------------------------------------------------------------------------
# Phenomenological model
# ---------------------------------------------------------------------------

class PhenomenologicalModel:
    """
    Independent Bernoulli Pauli errors, ideal syndrome readout.

    Standard code-capacity / single-round model.  I use this as the
    baseline for the single-round decoder comparison in the v5 study.
    """
    name = "phenomenological"

    def __call__(self, code: BBCode, p: float, rng: np.random.Generator) -> NoiseResult:
        z_mask = _bernoulli_mask(code.n, p, rng)
        x_mask = _bernoulli_mask(code.n, p, rng)
        return z_mask, x_mask, None, None


# ---------------------------------------------------------------------------
# Repeated-cycle model  (NEW in v5)
# ---------------------------------------------------------------------------

class RepeatedCycleModel:
    """
    d syndrome rounds per trial with noisy data and measurement errors.

    Protocol matches the v5 QAE study repeated-cycle benchmark:
      - rounds 1 .. d-1: Bernoulli data errors at rate p_data,
        Bernoulli measurement errors at rate p_meas.
      - round d: noiseless (standard fault-tolerant protocol).

    Returns
    -------
    Instead of (z_mask, x_mask, syn_z, syn_x), this model returns:
    (cumulative_z_mask, cumulative_x_mask, det_z_array, det_x_array)

    where det_z_array and det_x_array are numpy arrays of length
    (rounds * n_checks) containing the differential detector syndromes
    needed by space-time decoders (BP+OSD or MWPM space-time graph).

    Use RepeatedCycleRunner (in benchmark.py) instead of run_sweep
    when using this model.

    Parameters
    ----------
    rounds : int
        Number of syndrome rounds per trial (typically = d).
    p_meas : float or None
        Measurement error rate.  If None, defaults to p_data.
    """
    name = "repeated_cycle"

    def __init__(self, rounds: int, p_meas: Optional[float] = None) -> None:
        self.rounds = rounds
        self._p_meas_override = p_meas

    def __call__(
        self,
        code: BBCode,
        p: float,
        rng: np.random.Generator,
    ) -> Tuple[int, int, np.ndarray, np.ndarray]:
        n, n2     = code.n, code.n2
        rounds    = self.rounds
        p_meas    = self._p_meas_override if self._p_meas_override is not None else p

        cum_z = cum_x = 0
        syns_z: List[int] = []
        syns_x: List[int] = []

        for r in range(rounds):
            if r < rounds - 1:
                ez = _bernoulli_mask(n, p, rng)
                ex = _bernoulli_mask(n, p, rng)
            else:
                ez = ex = 0   # noiseless final round
            cum_z ^= ez
            cum_x ^= ex
            sz = syndrome_from_mask(cum_z, code.hx_cols)
            sx = syndrome_from_mask(cum_x, code.hz_cols)
            if r < rounds - 1:
                for i in range(n2):
                    if rng.random() < p_meas: sz ^= (1 << i)
                    if rng.random() < p_meas: sx ^= (1 << i)
            syns_z.append(sz)
            syns_x.append(sx)

        # Differential detector syndromes
        det_z = np.zeros(rounds * n2, dtype=np.uint8)
        det_x = np.zeros(rounds * n2, dtype=np.uint8)
        pz = px = 0
        for r, (sz, sx) in enumerate(zip(syns_z, syns_x)):
            d2z = sz ^ pz; d2x = sx ^ px
            for i in range(n2):
                det_z[r * n2 + i] = (d2z >> i) & 1
                det_x[r * n2 + i] = (d2x >> i) & 1
            pz = sz; px = sx

        return cum_z, cum_x, det_z, det_x


# ---------------------------------------------------------------------------
# Circuit-level model (single-cycle)
# ---------------------------------------------------------------------------

class CircuitLevelModel:
    """
    Pauli propagation through the BB syndrome-extraction circuit.

    Single-cycle model.  Returns noisy syndrome overrides so the
    decoder must cope with measurement faults as well as data errors.
    """
    name = "circuit"

    def __init__(
        self,
        p_prep_ratio: float = 1.0,
        p_meas_ratio: float = 1.0,
        p_idle_ratio: float = 0.05,
    ) -> None:
        self.p_prep_ratio = p_prep_ratio
        self.p_meas_ratio = p_meas_ratio
        self.p_idle_ratio = p_idle_ratio

    def __call__(self, code: BBCode, p: float, rng: np.random.Generator) -> NoiseResult:
        p_prep = p * self.p_prep_ratio
        p_gate = p
        p_meas = p * self.p_meas_ratio
        p_idle = p * self.p_idle_ratio

        total  = code.n2 + code.n + code.n2
        x_arr  = np.zeros(total, dtype=np.uint8)
        z_arr  = np.zeros(total, dtype=np.uint8)
        d_base = code.n2
        z_base = code.n2 + code.n

        for q in list(range(code.n2)) + list(range(z_base, total)):
            if rng.random() < p_prep:
                _flip_pauli(x_arr, z_arr, q, rng)

        schedule_x = [1, 4, 3, 5, 0, 2]
        schedule_z = [3, 5, 0, 1, 2, 4]

        for t in range(6):
            sx, sz = schedule_x[t], schedule_z[t]
            for c in range(code.n2):
                for check_type, sched, c_base, t_base in [
                    ("X", sx, 0,      d_base),
                    ("Z", sz, d_base, z_base),
                ]:
                    try:
                        dq = _neighbor_simple(code, check_type, c, sched)
                        ctrl = (c_base + c) if check_type == "X" else (d_base + dq)
                        tgt  = (d_base + dq) if check_type == "X" else (z_base + c)
                        _apply_cnot(x_arr, z_arr, ctrl, tgt)
                        if rng.random() < p_gate:
                            _flip_2q(x_arr, z_arr, ctrl, tgt, rng)
                    except Exception:
                        pass
            for q in range(d_base, d_base + code.n):
                if rng.random() < p_idle:
                    _flip_pauli(x_arr, z_arr, q, rng)

        data_z = z_arr[d_base: d_base + code.n]
        data_x = x_arr[d_base: d_base + code.n]
        z_mask = _arr_to_mask(data_z)
        x_mask = _arr_to_mask(data_x)

        syn_z = syndrome_from_mask(z_mask, code.hx_cols)
        syn_x = syndrome_from_mask(x_mask, code.hz_cols)
        for i in range(code.n2):
            if rng.random() < p_meas: syn_z ^= (1 << i)
            if rng.random() < p_meas: syn_x ^= (1 << i)

        return z_mask, x_mask, syn_z, syn_x


# ---------------------------------------------------------------------------
# Hardware model (single-cycle)
# ---------------------------------------------------------------------------

class HardwareModel:
    """
    Architecture-parameterised single-cycle simulation (CZ or MS backend).

    Device parameters match those used in the v5 QAE hardware simulations.
    The CZ/MS performance gap (1.52–2.11× in the v5 study) is a combined
    effect of SWAP routing overhead, native-gate error, idle decoherence,
    and schedule depth; these confounders are not isolated here.
    """
    name = "hardware"

    PRESETS: dict = {
        "CZ": dict(T1_us=150.0, T2_us=100.0, t_gate_us=0.2,
                   swap_count=96, schedule_ticks=128),
        "MS": dict(T1_us=1_000_000.0, T2_us=5_000_000.0, t_gate_us=50.0,
                   swap_count=18, schedule_ticks=74),
    }

    def __init__(self, backend: str = "CZ") -> None:
        if backend not in self.PRESETS:
            raise ValueError(f"backend must be 'CZ' or 'MS', got {backend!r}")
        self.backend = backend
        pr = self.PRESETS[backend]
        self.T1_us          = pr["T1_us"]
        self.T2_us          = pr["T2_us"]
        self.t_gate_us      = pr["t_gate_us"]
        self.swap_count     = pr["swap_count"]
        self.schedule_ticks = pr["schedule_ticks"]
        self.name           = f"hardware-{backend.lower()}"

    def _p_decoherence(self, t_us: float) -> float:
        p1 = 1 - np.exp(-t_us / self.T1_us)
        p2 = 1 - np.exp(-t_us / self.T2_us)
        return float(np.clip((p1 + p2) / 2, 0, 1))

    def __call__(self, code: BBCode, p_base: float, rng: np.random.Generator) -> NoiseResult:
        t_cycle = self.schedule_ticks * self.t_gate_us
        p_dec   = self._p_decoherence(t_cycle)
        p_eff   = min(1.0, p_base + p_dec + p_base * self.swap_count / code.n)
        z_mask  = _bernoulli_mask(code.n, p_eff, rng)
        x_mask  = _bernoulli_mask(code.n, p_eff, rng)
        return z_mask, x_mask, None, None


# ---------------------------------------------------------------------------
# Circuit helpers
# ---------------------------------------------------------------------------

def _flip_pauli(x: np.ndarray, z: np.ndarray, q: int,
                rng: np.random.Generator) -> None:
    r = int(rng.integers(1, 4))
    if r & 1: x[q] ^= 1
    if r & 2: z[q] ^= 1

def _flip_2q(x: np.ndarray, z: np.ndarray, q1: int, q2: int,
             rng: np.random.Generator) -> None:
    r = int(rng.integers(1, 16))
    p1, p2 = r & 3, (r >> 2) & 3
    if p1 & 1: x[q1] ^= 1
    if p1 & 2: z[q1] ^= 1
    if p2 & 1: x[q2] ^= 1
    if p2 & 2: z[q2] ^= 1

def _apply_cnot(x: np.ndarray, z: np.ndarray, ctrl: int, tgt: int) -> None:
    x[tgt]  ^= x[ctrl]
    z[ctrl] ^= z[tgt]

def _neighbor_simple(code: BBCode, check_type: str,
                     check_index: int, direction: int) -> int:
    mat = code._A if check_type == "X" else code._B
    offset = code.n2 if direction >= 3 else 0
    row = mat[check_index % mat.shape[0], :]
    nz  = np.nonzero(row)[0]
    return int(nz[direction % len(nz)]) + offset
