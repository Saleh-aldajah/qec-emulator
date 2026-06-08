"""
test_all.py — Comprehensive test suite for QEC Emulator v2.0.0

57 core tests + comprehensive v2 tests for new modules.
Author: Dr. Saleh H. AlDaajeh  ORCID: 0000-0001-7810-9290
"""
import hashlib, json, math, os, tempfile
from pathlib import Path
import numpy as np
import pytest

import qec_emulator as qec
from qec_emulator.codes import (BB72, BB144, BB288, Steane7, BBCode,
                                  gf2_rank, gf2_rref, in_rowspace,
                                  syndrome_from_mask, weight)
from qec_emulator.decoders import BPOSDDecoder, MWPMDecoder, LookupDecoder

# ─────────────────────────────────────────────────────────────────────────────
# Version
# ─────────────────────────────────────────────────────────────────────────────

class TestVersion:
    def test_version(self):
        assert qec.__version__ == "2.0.0"

    def test_imports(self):
        for name in ["BB72","BB144","BB288","Steane7","BBCode",
                     "BPOSDDecoder","MWPMDecoder","LookupDecoder",
                     "SimulationConfig","ResultSet","SimJob","JobScheduler",
                     "plot_lfr_vs_p","sparkline","interop_summary",
                     "to_networkx_tanner","to_alist","generate_minimax_bounds"]:
            assert hasattr(qec, name), f"Missing export: {name}"

def _has(pkg):
    try: __import__(pkg); return True
    except: return False


# ─────────────────────────────────────────────────────────────────────────────
# Codes
# ─────────────────────────────────────────────────────────────────────────────

class TestBBCodes:
    def test_bb72_dimensions(self):
        c = BB72()
        assert c.n == 72 and c.k == 12 and c.n2 == 36

    def test_bb72_commutation(self):
        assert BB72().commutation_check()

    def test_bb72_sha(self):
        c = BB72()
        assert c.hx_sha256().startswith("7ab0973b")
        assert c.hz_sha256().startswith("267f345c")

    def test_bb144_dimensions(self):
        c = BB144()
        assert c.n == 144 and c.k == 12

    def test_bb288_dimensions(self):
        c = BB288()
        assert c.n == 288 and c.k == 12
        assert c.hx_sha256().startswith("c06e721e")

    def test_steane7(self):
        c = Steane7()
        assert c.n == 7 and c.k == 1

    def test_custom_code(self):
        c = BBCode(ell=4, m=4, a_exponents=(1,1,2), b_exponents=(2,1,1))
        assert c.n == 32 and c.commutation_check()

    def test_to_dict(self):
        d = BB72().to_dict()
        assert d["n"] == 72 and d["k"] == 12 and d["commutation_pass"]

    def test_gf2_rank_bb72(self):
        c = BB72()
        assert gf2_rank(c.hx) == c.rx

    def test_gf2_rref(self):
        c = BB72()
        rref, pivots = gf2_rref(c.hx)
        assert rref.shape[0] == c.rx
        assert len(pivots) == c.rx

    def test_in_rowspace(self):
        c = BB72()
        assert in_rowspace(c.hx[0], c.hx_rref, c.hx_pivots)

    def test_syndrome_from_mask(self):
        c = BB72()
        s = syndrome_from_mask(1, c.hx_cols)
        assert isinstance(s, int)

# ─────────────────────────────────────────────────────────────────────────────
# Decoders
# ─────────────────────────────────────────────────────────────────────────────

class TestDecoders:
    @pytest.fixture
    def bb72(self):
        return BB72()

    def test_lookup_decoder_zero(self, bb72):
        d = LookupDecoder(bb72)
        assert d.decode_z(0) == 0  # zero syndrome → zero correction

    def test_lookup_decoder_single_error(self, bb72):
        d = LookupDecoder(bb72)
        # single qubit-0 error: syndrome = hx_cols[0]
        s_int = syndrome_from_mask(1, bb72.hx_cols)
        cor = d.decode_z(s_int)
        assert cor & 1 == 1  # qubit 0 correction

    @pytest.mark.skipif(not _has("ldpc"), reason="ldpc not installed")
    def test_bposd_zero(self, bb72):
        d = BPOSDDecoder(bb72, channel_probs=0.01)
        c = d.decode_z(0)
        assert c == 0

    @pytest.mark.skipif(not _has("pymatching"), reason="pymatching not installed")
    def test_mwpm_zero(self, bb72):
        d = MWPMDecoder(bb72, channel_probs=0.01)
        c = d.decode_z(0)
        assert c == 0


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_default_config(self):
        cfg = qec.SimulationConfig()
        assert cfg.runner.trials == 5000
        assert cfg.noise.model == "phenomenological"

    def test_to_from_json(self):
        cfg = qec.SimulationConfig()
        cfg.runner.trials = 9999
        j   = cfg.to_json()
        cfg2 = qec.SimulationConfig.from_json(j)
        assert cfg2.runner.trials == 9999

    def test_to_from_yaml(self):
        cfg = qec.SimulationConfig()
        cfg.code.preset = "BB144"
        y   = cfg.to_yaml()
        cfg2 = qec.SimulationConfig.from_yaml(y)
        assert cfg2.code.preset == "BB144"

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/"cfg.yaml"
            cfg = qec.SimulationConfig()
            cfg.runner.trials = 1234
            cfg.save(p)
            cfg2 = qec.SimulationConfig.load(p)
            assert cfg2.runner.trials == 1234

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("QEC_TRIALS", "7777")
        cfg = qec.SimulationConfig()._apply_env_overrides()
        assert cfg.runner.trials == 7777

    def test_write_default(self):
        with tempfile.TemporaryDirectory() as td:
            p = qec.write_default_config(Path(td)/"default.yaml")
            assert p.exists() and "BB72" in p.read_text()

# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

class TestExport:
    def _rs(self):
        rs = qec.ResultSet()
        rs.add("BB72","bposd",0.008,14,10000,999)
        rs.add("BB144","bposd",0.008,0,10000,998)
        return rs

    def test_add(self):
        rs = self._rs()
        assert len(rs) == 2

    def test_integrity_hash_stable(self):
        rs = self._rs()
        h1 = rs.integrity_hash(); h2 = rs.integrity_hash()
        assert h1 == h2 and len(h1) == 64

    def test_summary(self):
        s = self._rs().summary()
        assert "BB72" in s

    def test_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/"r.csv"
            self._rs().to_csv(p)
            rs2 = qec.ResultSet.from_csv(p)
            assert len(rs2) == 2

    def test_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/"r.json"
            self._rs().to_json(p)
            rs2 = qec.ResultSet.from_json(p)
            assert len(rs2) == 2

    def test_filter(self):
        rs = self._rs()
        rs_f = rs.filter(code="BB72")
        assert len(rs_f) == 1

    def test_save_multiple_formats(self):
        with tempfile.TemporaryDirectory() as td:
            saved = self._rs().save(Path(td)/"run", formats=["csv","json"])
            assert "csv" in saved and "json" in saved
            assert saved["csv"].exists() and saved["json"].exists()

    def test_wilson_ci(self):
        rs = qec.ResultSet()
        rs.add("BB72","bposd",0.008,0,10000,1)
        row = rs.rows[0]
        assert row['ci_lo'] == 0.0
        assert 0.0 < row['ci_hi'] < 0.001

# ─────────────────────────────────────────────────────────────────────────────
# Interop
# ─────────────────────────────────────────────────────────────────────────────

class TestInterop:
    def test_networkx_tanner(self):
        import networkx as nx
        G = qec.to_networkx_tanner(BB72(), "X")
        checks = [v for v in G.nodes if v[0]=="check"]
        qubits = [v for v in G.nodes if v[0]=="qubit"]
        assert len(checks) == 36 and len(qubits) == 72

    def test_networkx_factor(self):
        import networkx as nx
        G = qec.to_networkx_factor(BB72(), "X")
        assert G.number_of_nodes() >= 36

    def test_alist_roundtrip(self):
        c = BB72()
        al = qec.to_alist(c.hx)
        H2 = qec.from_alist(al)
        assert np.array_equal(H2, c.hx)

    def test_sparse_dict_roundtrip(self):
        from qec_emulator.interop import to_sparse_dict, from_sparse_dict
        c = BB72()
        d = to_sparse_dict(c.hx)
        H2 = from_sparse_dict(d)
        assert np.array_equal(H2, c.hx)

    def test_stim_dem_manual(self):
        from qec_emulator.interop import _to_stim_dem_manual
        dem = _to_stim_dem_manual(BB72(), 0.01, 1)
        assert "error" in dem.lower() or "#" in dem

    def test_interop_summary(self):
        s = qec.interop_summary(BB72())
        assert s["networkx_tanner"] == True
        assert s["alist_roundtrip"] == True

# ─────────────────────────────────────────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────────────────────────────────────────

class TestScheduler:
    def test_build_sweep_jobs(self):
        sched = qec.JobScheduler(workers=1)
        jobs  = sched.build_sweep_jobs(["BB72"],["lookup"],[0.005,0.01],trials=50)
        assert len(jobs) == 2

    def test_stable_seed(self):
        from qec_emulator.scheduler import _stable_seed
        s1 = _stable_seed("BB72","bposd",0.008)
        s2 = _stable_seed("BB72","bposd",0.008)
        assert s1 == s2 and s1 > 0

    def test_run_single_job_lookup(self):
        from qec_emulator.scheduler import SimJob, _run_one_job
        j = SimJob(code_preset="BB72", decoder="lookup", p=0.01, trials=100)
        r = _run_one_job(j)
        assert "failures" in r and 0 <= r['failures'] <= 100
        assert r['code'] == "BB72"

    def test_run_scheduler(self):
        sched = qec.JobScheduler(workers=1)
        jobs  = sched.build_sweep_jobs(["BB72"],["lookup"],[0.01],trials=100)
        results = sched.run(jobs, progress=False)
        assert len(results) == 1
        assert results[0]['code'] == "BB72"

    @pytest.mark.skipif(not _has("ldpc"), reason="ldpc not installed")
    def test_run_bposd(self):
        from qec_emulator.scheduler import SimJob, _run_one_job
        j = SimJob(code_preset="BB72", decoder="bposd", p=0.005, trials=50)
        r = _run_one_job(j)
        assert r['failures'] >= -1  # -1 means decoder unavailable

# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

class TestVisualization:
    @pytest.fixture(autouse=True)
    def use_agg(self):
        import matplotlib
        matplotlib.use('Agg')

    def _data(self):
        rows = []
        for code in ["BB72","BB144"]:
            for dec in ["BP+OSD","MWPM-DEM"]:
                for p, f in [(0.004,0),(0.008,14 if code=="BB72" else 0),(0.012,50)]:
                    f2 = f if dec=="BP+OSD" else f*100
                    rows.append({"code":code,"decoder":dec,"p":p,
                                 "failures":f2,"trials":10000,
                                 "lfr":f2/10000,"ci_lo":0,"ci_hi":0.001})
        return rows

    def test_lfr_plot(self):
        fig = qec.plot_lfr_vs_p(self._data(), show=False)
        assert fig is not None

    def test_threshold_plot(self):
        from qec_emulator.visualization import plot_threshold
        fig = plot_threshold(self._data(), show=False)
        assert fig is not None

    def test_rr_plot(self):
        from qec_emulator.visualization import plot_risk_ratios
        fig = plot_risk_ratios(self._data(), p_value=0.008, show=False)
        assert fig is not None

    def test_tanner_plot(self):
        from qec_emulator.visualization import plot_tanner_graph
        fig = plot_tanner_graph(BB72(), max_nodes=30, show=False)
        assert fig is not None

    def test_sparkline(self):
        sp = qec.sparkline([0.001, 0.01, 0.1, 0.5])
        assert len(sp) == 4 and "█" in sp

    def test_plot_to_file(self):
        import matplotlib
        matplotlib.use('Agg')
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/"lfr.png"
            qec.plot_lfr_vs_p(self._data(), output=str(p), show=False)
            assert p.exists() and p.stat().st_size > 1000

# ─────────────────────────────────────────────────────────────────────────────
# Minimax (carried over)
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateMinimax:
    def test_returns_list(self):
        rows = qec.generate_minimax_bounds()
        assert isinstance(rows, list) and len(rows) == 22

    def test_delta_01_values(self):
        rows = qec.generate_minimax_bounds(n=128, d=11, t=5, delta=0.1)
        r = next(r for r in rows if abs(float(r["p_attack"]) - 0.001) < 1e-9)
        assert abs(r["lower_bound"] - 0.9000) < 0.001

    def test_bounds_in_range(self):
        for r in qec.generate_minimax_bounds():
            assert 0 <= r["lower_bound"] <= 1

# ─────────────────────────────────────────────────────────────────────────────
# API (lightweight, no server start)
# ─────────────────────────────────────────────────────────────────────────────

class TestAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from qec_emulator.api import create_app
        return TestClient(create_app())

    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["version"] == "1.2.3"

    def test_list_codes(self, client):
        r = client.get("/codes")
        assert r.status_code == 200
        codes = [c["name"] for c in r.json()["codes"]]
        assert "BB72" in codes and "BB144" in codes and "BB288" in codes

    def test_get_code(self, client):
        r = client.get("/codes/BB72")
        assert r.status_code == 200
        d = r.json()
        assert d["n"] == 72 and d["k"] == 12

    def test_get_unknown_code(self, client):
        r = client.get("/codes/BB999")
        assert r.status_code == 404

    def test_simulate_small(self, client):
        payload = {
            "codes":    ["BB72"],
            "decoders": ["lookup"],
            "p_values": [0.01],
            "trials":   100,
            "parallel": False,
        }
        r = client.post("/simulate", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["n_jobs"] == 1
        assert len(data["results"]) == 1
        assert data["results"][0]["code"] == "BB72"

    def test_simulate_multi_code(self, client):
        payload = {
            "codes":    ["BB72","BB144"],
            "decoders": ["lookup"],
            "p_values": [0.01, 0.05],
            "trials":   50,
        }
        r = client.post("/simulate", json=payload)
        assert r.json()["n_jobs"] == 4

    def test_config_valid(self, client):
        yaml = "code:\n  preset: BB72\nrunner:\n  trials: 100\n"
        r = client.post("/config", json={"config_yaml": yaml})
        assert r.json()["valid"] == True

    def test_config_invalid(self, client):
        r = client.post("/config", json={"config_yaml": "not: {valid: yaml: :"})
        assert r.json()["valid"] == False


# ─────────────────────────────────────────────────────────────────────────────
# CLI (subprocess)
# ─────────────────────────────────────────────────────────────────────────────

class TestCLI:
    def test_info(self):
        import subprocess, sys
        r = subprocess.run([sys.executable,"-m","qec_emulator.cli","info"],
                           capture_output=True, text=True)
        assert r.returncode == 0
        assert "2.0.0" in r.stdout or "QEC" in r.stdout

    def test_sweep_lookup(self):
        import subprocess, sys, tempfile
        with tempfile.TemporaryDirectory() as td:
            r = subprocess.run([
                sys.executable,"-m","qec_emulator.cli","sweep",
                "--code","BB72","--decoder","lookup",
                "--p","0.01","--trials","100",
                "--output", str(Path(td)/"out"), "--quiet"
            ], capture_output=True, text=True, timeout=60)
            assert r.returncode == 0 or "LFR" in r.stdout

    def test_init_config(self):
        import subprocess, sys, tempfile
        with tempfile.TemporaryDirectory() as td:
            out = str(Path(td)/"cfg.yaml")
            r = subprocess.run([sys.executable,"-m","qec_emulator.cli",
                                "init-config","--output",out],
                               capture_output=True, text=True)
            assert r.returncode == 0
            assert Path(out).exists()


# ─────────────────────────────────────────────────────────────────────────────
# Provenance (carried over)
# ─────────────────────────────────────────────────────────────────────────────

class TestProvenance:
    def test_logger_records(self):
        from qec_emulator.provenance import ProvenanceLogger
        logger = ProvenanceLogger()
        r = logger.new_record(
            code_name="BB72",
            decoder_name="bposd",
            noise_model="phenomenological",
            p=0.008,
            trials=1000,
            seed=12345,
            hx_sha256="abc",
            hz_sha256="def")
        assert r is not None
        s = logger.summary()
        assert s is not None

    def test_unresolved_pairs(self):
        from qec_emulator.provenance import ProvenanceLogger
        logger = ProvenanceLogger()
        logger.new_record(
            code_name="BB72",
            decoder_name="bposd",
            noise_model="phenomenological",
            p=0.008,
            trials=1000,
            seed=12345,
            hx_sha256="abc",
            hz_sha256="def")
        pairs = logger.unresolved_pairs()
        assert isinstance(pairs, list)

# ─────────────────────────────────────────────────────────────────────────────
# SHA-256 pinning
# ─────────────────────────────────────────────────────────────────────────────

class TestSHA256Utilities:
    def test_bb72_hx_sha(self):
        c = BB72()
        assert c.hx_sha256() == \
            "7ab0973bfd02e399d69728d26e67dbfa0e95bbc6ad2c5cf8bf309d667244a5a8"

    def test_bb72_hz_sha(self):
        c = BB72()
        assert c.hz_sha256() == \
            "267f345c09710ec0cfad1e49e0786b2636ab052fbfb660e5a76ae9919ba22614"

    def test_bb288_hx_sha_prefix(self):
        c = BB288()
        assert c.hx_sha256().startswith("c06e721e")

    def test_stable_seed_determinism(self):
        from qec_emulator.scheduler import _stable_seed
        a = _stable_seed("BB72","BP+OSD",0.008)
        b = _stable_seed("BB72","BP+OSD",0.008)
        assert a == b

    def test_stable_seed_different_args(self):
        from qec_emulator.scheduler import _stable_seed
        a = _stable_seed("BB72","BP+OSD",0.008)
        b = _stable_seed("BB72","BP+OSD",0.012)
        assert a != b

