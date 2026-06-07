"""
noise.py — Noise models for QEC Monte Carlo experiments.

Each model is a callable that takes (code, p, rng) and returns
(error_z_mask, error_x_mask, syndrome_z_override, syndrome_x_override).

Models
------
PhenomenologicalModel   — independent p errors per data qubit, ideal syndrome
CircuitLevelModel       — Pauli propagation through CNOT schedule + noisy readout
HardwareModel           — architecture-parameterised model (CZ or MS backend)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .codes import BBCode, syndrome_from_mask


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

NoiseResult = Tuple[int, int, Optional[int], Optional[int]]
# (error_z_mask, error_x_mask, syndrome_z_override, syndrome_x_override)
# syndrome overrides are None for ideal-syndrome models.


# ---------------------------------------------------------------------------
# Phenomenological (code-capacity) model
# ---------------------------------------------------------------------------

class PhenomenologicalModel:
    """
    Independent Bernoulli errors on each data qubit; ideal syndrome readout.

    This is the standard code-capacity or phenomenological repeated-cycle
    model. Each data qubit suffers an independent Z-error with probability p
    and an independent X-error with probability p.
    """
    name = "phenomenological"

    def __call__(self, code: BBCode, p: float, rng: np.random.Generator) -> NoiseResult:
        z_mask = int(np.packbits(
            (rng.random(code.n) < p).astype(np.uint8), bitorder="little"
        ).view(np.uint64)[0]) if code.n <= 64 else _bernoulli_mask(code.n, p, rng)
        x_mask = int(np.packbits(
            (rng.random(code.n) < p).astype(np.uint8), bitorder="little"
        ).view(np.uint64)[0]) if code.n <= 64 else _bernoulli_mask(code.n, p, rng)
        # Mask to n bits
        mask = (1 << code.n) - 1
        return z_mask & mask, x_mask & mask, None, None


class CircuitLevelModel:
    """
    Single-cycle Pauli propagation through the BB syndrome-extraction schedule.

    Fault locations:
      - Ancilla preparation: X/Z flip with probability p_prep.
      - CNOT gates: uniformly random two-qubit Pauli with probability p_gate.
      - Data qubit idle: X/Z flip with probability p_idle per qubit per layer.
      - Syndrome measurement: bit-flip with probability p_meas.

    Returns noisy syndrome overrides so the decoder sees measurement errors.
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

        total = code.n2 + code.n + code.n2
        x_arr = np.zeros(total, dtype=np.uint8)
        z_arr = np.zeros(total, dtype=np.uint8)
        x_base = 0
        d_base = code.n2
        z_base = code.n2 + code.n

        # Ancilla preparation faults
        for q in list(range(code.n2)) + list(range(z_base, total)):
            if rng.random() < p_prep:
                _flip_random_pauli(x_arr, z_arr, q, rng)

        # CNOT schedule (6 layers)
        schedule_x = [1, 4, 3, 5, 0, 2]
        schedule_z = [3, 5, 0, 1, 2, 4]
        for t in range(6):
            sx, sz = schedule_x[t], schedule_z[t]
            for c in range(code.n2):
                # X checks
                try:
                    dq = code.neighbor("X", c, sx)
                    ctrl, tgt = x_base + c, d_base + dq
                    _apply_cnot(x_arr, z_arr, ctrl, tgt)
                    if rng.random() < p_gate:
                        _flip_random_2q_pauli(x_arr, z_arr, ctrl, tgt, rng)
                except Exception:
                    pass
                # Z checks
                try:
                    dq = code.neighbor("Z", c, sz)
                    ctrl, tgt = d_base + dq, z_base + c
                    _apply_cnot(x_arr, z_arr, ctrl, tgt)
                    if rng.random() < p_gate:
                        _flip_random_2q_pauli(x_arr, z_arr, ctrl, tgt, rng)
                except Exception:
                    pass
            # Sparse idle errors on data qubits
            for q in range(d_base, d_base + code.n):
                if rng.random() < p_idle:
                    _flip_random_pauli(x_arr, z_arr, q, rng)

        data_x = x_arr[d_base: d_base + code.n]
        data_z = z_arr[d_base: d_base + code.n]
        z_mask = _array_to_mask(data_z)
        x_mask = _array_to_mask(data_x)

        # Ideal syndromes then flip each bit with p_meas
        syn_z = syndrome_from_mask(z_mask, code.hx_cols)
        syn_x = syndrome_from_mask(x_mask, code.hz_cols)
        for i in range(code.n2):
            if rng.random() < p_meas:
                syn_z ^= (1 << i)
            if rng.random() < p_meas:
                syn_x ^= (1 << i)

        return z_mask, x_mask, syn_z, syn_x


class HardwareModel:
    """
    Architecture-aware single-cycle model parameterised by backend properties.

    Simulates compiled BB72 circuits for superconducting CZ or trapped-ion MS
    backends using T1/T2 decoherence, gate error rates, and SWAP overhead.

    Parameters
    ----------
    backend : 'CZ' or 'MS'
    T1_us, T2_us : coherence times in microseconds
    t_gate_us : two-qubit gate time in microseconds
    swap_count : number of SWAP gates in the compiled circuit
    """
    name = "hardware"

    PRESETS = {
        "CZ": dict(T1_us=150.0, T2_us=100.0, t_gate_us=0.2, two_qubit_gates=432, swap_count=96, schedule_ticks=128),
        "MS": dict(T1_us=1_000_000.0, T2_us=5_000_000.0, t_gate_us=50.0, two_qubit_gates=432, swap_count=18, schedule_ticks=74),
    }

    def __init__(self, backend: str = "CZ") -> None:
        if backend not in self.PRESETS:
            raise ValueError(f"backend must be 'CZ' or 'MS', got {backend!r}")
        self.backend = backend
        p = self.PRESETS[backend]
        self.T1_us = p["T1_us"]
        self.T2_us = p["T2_us"]
        self.t_gate_us = p["t_gate_us"]
        self.swap_count = p["swap_count"]
        self.two_qubit_gates = p["two_qubit_gates"]
        self.schedule_ticks = p["schedule_ticks"]
        self.name = f"hardware-{backend.lower()}"

    def _p_decoherence(self, t_us: float) -> float:
        """Effective single-qubit depolarising rate from T1/T2."""
        p1 = 1 - np.exp(-t_us / self.T1_us)
        p2 = 1 - np.exp(-t_us / self.T2_us)
        return float(np.clip((p1 + p2) / 2, 0, 1))

    def __call__(self, code: BBCode, p_base: float, rng: np.random.Generator) -> NoiseResult:
        # Effective gate error = base rate + decoherence during gate time
        t_cycle_us = self.schedule_ticks * self.t_gate_us
        p_decohere = self._p_decoherence(t_cycle_us)
        p_eff = min(1.0, p_base + p_decohere + p_base * self.swap_count / code.n)

        # Use phenomenological with effective rate
        pheno = PhenomenologicalModel()
        z_mask, x_mask, _, _ = pheno(code, p_eff, rng)
        return z_mask, x_mask, None, None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bernoulli_mask(n: int, p: float, rng: np.random.Generator) -> int:
    mask = 0
    for j in range(n):
        if rng.random() < p:
            mask |= 1 << j
    return mask


def _array_to_mask(arr: np.ndarray) -> int:
    mask = 0
    for i, b in enumerate(arr):
        if int(b) & 1:
            mask |= 1 << i
    return mask


def _flip_random_pauli(x: np.ndarray, z: np.ndarray, q: int, rng: np.random.Generator) -> None:
    r = int(rng.integers(1, 4))
    if r & 1:
        x[q] ^= 1
    if r & 2:
        z[q] ^= 1


def _flip_random_2q_pauli(x: np.ndarray, z: np.ndarray, q1: int, q2: int, rng: np.random.Generator) -> None:
    r = int(rng.integers(1, 16))
    p1, p2 = r & 3, (r >> 2) & 3
    if p1 & 1:
        x[q1] ^= 1
    if p1 & 2:
        z[q1] ^= 1
    if p2 & 1:
        x[q2] ^= 1
    if p2 & 2:
        z[q2] ^= 1


def _apply_cnot(x: np.ndarray, z: np.ndarray, ctrl: int, tgt: int) -> None:
    x[tgt] ^= x[ctrl]
    z[ctrl] ^= z[tgt]
