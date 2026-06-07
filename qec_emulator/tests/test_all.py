"""
tests/test_all.py — Comprehensive test suite for qec_emulator.

Run with:  pytest tests/ -v
"""
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from qec_emulator import (
    BB72, BB144, Steane7, BBCode,
    LookupDecoder, BPOSDDecoder, MWPMDecoder,
    PhenomenologicalModel, CircuitLevelModel, HardwareModel,
    ProvenanceLogger, ProvenanceRecord,
    run_sweep, run_fixed_weight, run_distance_certificate,
    sha256_of_dict,
)


# ===========================================================================
# Code construction tests
# ===========================================================================

class TestBB72:
    def setup_method(self):
        self.code = BB72()

    def test_parameters(self):
        assert self.code.n == 72
        assert self.code.k == 12

    def test_commutation(self):
        assert self.code.commutation_check()

    def test_row_weights(self):
        row_wts_x = set(self.code.hx.sum(axis=1).tolist())
        row_wts_z = set(self.code.hz.sum(axis=1).tolist())
        assert row_wts_x == {6}, f"Expected HX row weight 6, got {row_wts_x}"
        assert row_wts_z == {6}, f"Expected HZ row weight 6, got {row_wts_z}"

    def test_column_weights(self):
        col_wts_x = set(self.code.hx.sum(axis=0).tolist())
        col_wts_z = set(self.code.hz.sum(axis=0).tolist())
        assert col_wts_x == {3}
        assert col_wts_z == {3}

    def test_sha256_deterministic(self):
        code2 = BB72()
        assert self.code.hx_sha256() == code2.hx_sha256()
        assert self.code.hz_sha256() == code2.hz_sha256()

    def test_to_dict(self):
        d = self.code.to_dict()
        assert d["n"] == 72
        assert d["k"] == 12
        assert d["commutation_pass"] is True
        assert len(d["hx_sha256"]) == 64

    def test_rank(self):
        assert self.code.rx == 30
        assert self.code.rz == 30


class TestBB144:
    def test_parameters(self):
        code = BB144()
        assert code.n == 144
        assert code.commutation_check()

    def test_k_positive(self):
        code = BB144()
        assert code.k > 0


class TestSteane7:
    def test_parameters(self):
        code = Steane7()
        assert code.n == 7
        assert code.k == 1

    def test_k_positive(self):
        code = Steane7()
        assert code.k > 0


# ===========================================================================
# Decoder tests
# ===========================================================================

class TestLookupDecoder:
    def setup_method(self):
        self.code = BB72()
        self.dec = LookupDecoder(self.code, radius=2)

    def test_zero_syndrome(self):
        assert self.dec.decode_z(0) == 0
        assert self.dec.decode_x(0) == 0

    def test_weight1_correctable(self):
        """Every single-qubit error must be in the lookup table."""
        for q in range(self.code.n):
            mask = 1 << q
            from qec_emulator.codes import syndrome_from_mask
            syn = syndrome_from_mask(mask, self.code.hx_cols)
            corr = self.dec.decode_z(syn)
            assert corr != 0, f"Qubit {q} Z-error not corrected"

    def test_decode_success_trivial(self):
        """No error → always success."""
        assert self.dec.decode_success(0, 0)

    def test_not_fallback(self):
        assert self.dec.is_fallback is False


class TestBPOSDDecoder:
    def test_instantiation_no_crash(self):
        """BPOSDDecoder should instantiate even if ldpc is not installed."""
        code = BB72()
        dec = BPOSDDecoder(code)
        # Either real or fallback — both must handle zero syndrome
        assert dec.decode_z(0) == 0
        assert dec.decode_x(0) == 0

    def test_fallback_label(self):
        """If ldpc is not installed, is_fallback must be True."""
        try:
            import ldpc
            pytest.skip("ldpc installed — fallback path not active")
        except ImportError:
            code = BB72()
            dec = BPOSDDecoder(code)
            assert dec.is_fallback is True


class TestMWPMDecoder:
    def test_instantiation_no_crash(self):
        code = BB72()
        dec = MWPMDecoder(code)
        assert dec.decode_z(0) == 0

    def test_fallback_label(self):
        try:
            import pymatching
            pytest.skip("pymatching installed — fallback path not active")
        except ImportError:
            code = BB72()
            dec = MWPMDecoder(code)
            assert dec.is_fallback is True


# ===========================================================================
# Noise model tests
# ===========================================================================

class TestPhenomenologicalModel:
    def test_zero_p(self):
        code = BB72()
        model = PhenomenologicalModel()
        rng = np.random.default_rng(42)
        z, x, sz, sx = model(code, 0.0, rng)
        assert z == 0
        assert x == 0
        assert sz is None
        assert sx is None

    def test_returns_in_range(self):
        code = BB72()
        model = PhenomenologicalModel()
        rng = np.random.default_rng(42)
        z, x, _, _ = model(code, 0.5, rng)
        maxmask = (1 << code.n) - 1
        assert z & maxmask == z
        assert x & maxmask == x


class TestCircuitLevelModel:
    def test_returns_syndrome_overrides(self):
        code = BB72()
        model = CircuitLevelModel()
        rng = np.random.default_rng(42)
        z, x, sz, sx = model(code, 0.01, rng)
        assert sz is not None
        assert sx is not None


class TestHardwareModel:
    def test_cz_instantiation(self):
        m = HardwareModel("CZ")
        assert m.name == "hardware-cz"

    def test_ms_instantiation(self):
        m = HardwareModel("MS")
        assert m.name == "hardware-ms"

    def test_invalid_backend(self):
        with pytest.raises(ValueError):
            HardwareModel("QPU9000")


# ===========================================================================
# Provenance tests
# ===========================================================================

class TestProvenanceRecord:
    def test_set_result(self):
        rec = ProvenanceRecord(
            code="BB72", decoder="Lookup", noise_model="pheno",
            p=0.004, trials=1000, seed=42,
            hx_sha256="a" * 64, hz_sha256="b" * 64,
        )
        rec.set_result(5)
        assert rec.failures == 5
        assert abs(rec.lfr - 0.005) < 1e-10
        assert rec.result_sha256 is not None
        assert len(rec.result_sha256) == 64

    def test_qc_status_pass(self):
        rec = ProvenanceRecord(
            code="BB72", decoder="Lookup", noise_model="pheno",
            p=0.001, trials=5000, seed=1, hx_sha256="a"*64, hz_sha256="b"*64,
        )
        rec.set_result(10)
        assert rec.qc_status() == "PASS"

    def test_qc_status_zero(self):
        rec = ProvenanceRecord(
            code="BB72", decoder="Lookup", noise_model="pheno",
            p=0.0001, trials=5000, seed=1, hx_sha256="a"*64, hz_sha256="b"*64,
        )
        rec.set_result(0)
        assert "ZERO_FAILURE" in rec.qc_status()

    def test_sha256_deterministic(self):
        def make():
            r = ProvenanceRecord(
                code="BB72", decoder="Lookup", noise_model="pheno",
                p=0.004, trials=1000, seed=42,
                hx_sha256="a"*64, hz_sha256="b"*64,
            )
            r.set_result(5)
            return r.result_sha256
        assert make() == make()


class TestProvenanceLogger:
    def test_csv_roundtrip(self):
        code = BB72()
        dec = LookupDecoder(code)
        noise = PhenomenologicalModel()
        logger = run_sweep(code, dec, noise, p_values=[0.004], trials=200, seed=1, verbose=False)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        logger.to_csv(path)
        import csv
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["code"] == "BB72"
        assert float(rows[0]["p"]) == 0.004

    def test_json_roundtrip(self):
        code = BB72()
        dec = LookupDecoder(code)
        noise = PhenomenologicalModel()
        logger = run_sweep(code, dec, noise, p_values=[0.002], trials=100, seed=2, verbose=False)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        logger.to_json(path)
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert data[0]["decoder"] == "Lookup"

    def test_unresolved_pairs_detection(self):
        """Identical result_sha256 in same (code, decoder, p) group is flagged."""
        logger = ProvenanceLogger()
        for _ in range(2):
            rec = logger.new_record("BB72", "Lookup", "pheno", 0.004, 1000, 42, "a"*64, "b"*64)
            rec.set_result(0)  # Identical zero results
        flagged = logger.unresolved_pairs()
        assert len(flagged) == 2

    def test_different_seeds_not_flagged(self):
        """Different seeds producing different results must NOT be flagged."""
        code = BB72()
        dec = LookupDecoder(code)
        noise = PhenomenologicalModel()
        logger = run_sweep(code, dec, noise, p_values=[0.006], trials=500, seed=100, batches=2, verbose=False)
        flagged = logger.unresolved_pairs()
        assert len(flagged) == 0, "Two independent batches flagged as colliding"


# ===========================================================================
# Benchmark runner tests
# ===========================================================================

class TestRunSweep:
    def test_returns_logger(self):
        code = BB72()
        dec = LookupDecoder(code)
        noise = PhenomenologicalModel()
        logger = run_sweep(code, dec, noise, p_values=[0.004], trials=100, seed=42, verbose=False)
        assert len(logger.records) == 1

    def test_all_records_complete(self):
        code = BB72()
        dec = LookupDecoder(code)
        noise = PhenomenologicalModel()
        ps = [0.002, 0.004, 0.006]
        logger = run_sweep(code, dec, noise, p_values=ps, trials=100, verbose=False)
        assert all(r.lfr is not None for r in logger.records)
        assert all(r.result_sha256 is not None for r in logger.records)

    def test_deterministic(self):
        """Same seed → identical LFR."""
        code = BB72()
        dec = LookupDecoder(code)
        noise = PhenomenologicalModel()
        kw = dict(p_values=[0.004], trials=200, seed=7, verbose=False)
        r1 = run_sweep(code, dec, noise, **kw).records[0]
        r2 = run_sweep(code, dec, noise, **kw).records[0]
        assert r1.failures == r2.failures
        assert r1.result_sha256 == r2.result_sha256

    def test_different_seeds_not_always_identical(self):
        """Different seeds at moderate p should (almost certainly) give different failure counts."""
        code = BB72()
        dec = LookupDecoder(code)
        noise = PhenomenologicalModel()
        r1 = run_sweep(code, dec, noise, p_values=[0.3], trials=300, seed=1, verbose=False).records[0]
        r2 = run_sweep(code, dec, noise, p_values=[0.3], trials=300, seed=9999, verbose=False).records[0]
        # At p=0.3 both will have many failures, but different seeds → different counts
        # Extremely unlikely (probability ~1/sqrt(300) ≈ 5%) to be equal
        assert r1.result_sha256 != r2.result_sha256


class TestRunFixedWeight:
    def test_weight0_always_passes(self):
        code = BB72()
        dec = LookupDecoder(code)
        logger = run_fixed_weight(code, dec, weights=[0], trials=50, verbose=False)
        assert logger.records[0].failures == 0

    def test_weight1_always_passes_lookup(self):
        """Radius-2 lookup corrects all weight-1 errors."""
        code = BB72()
        dec = LookupDecoder(code)
        logger = run_fixed_weight(code, dec, weights=[1], trials=100, verbose=False)
        assert logger.records[0].failures == 0


class TestRunDistanceCertificate:
    def test_bb72_passes(self):
        code = BB72()
        result = run_distance_certificate(code, max_weight=5, verbose=False)
        assert result["Z_component_hx"]["distance_lower_bound_pass"] is True
        assert result["X_component_hz"]["distance_lower_bound_pass"] is True

    def test_steane7_passes(self):
        """Steane7 distance certificate must run without error."""
        code = Steane7()
        result = run_distance_certificate(code, max_weight=3, verbose=False)
        assert "Z_component_hx" in result
        assert "X_component_hz" in result


# ===========================================================================
# sha256 utility tests
# ===========================================================================

def test_sha256_of_dict_deterministic():
    d = {"code": "BB72", "p": 0.004, "trials": 5000}
    assert sha256_of_dict(d) == sha256_of_dict(d)


def test_sha256_of_dict_sensitive():
    d1 = {"code": "BB72", "p": 0.004}
    d2 = {"code": "BB72", "p": 0.005}
    assert sha256_of_dict(d1) != sha256_of_dict(d2)
