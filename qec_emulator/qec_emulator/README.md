# QEC Emulator v2.0.0

**Reproducible qLDPC decoder benchmarking with full interoperability.**

Author: Dr. Saleh H. AlDaajeh · ORCID: 0000-0001-7810-9290  
License: MIT · Archive DOI: 10.5281/zenodo.20574329

---

## What's new in v2.0.0

| Feature | Status |
|---|---|
| REST/HTTP API server (FastAPI) | ✓ New |
| CLI (`qec-emulator` command) | ✓ New |
| YAML/JSON configuration files | ✓ New |
| Parallel job scheduler | ✓ New |
| Multi-format export (CSV/JSON/HDF5) | ✓ New |
| Stim circuit & DEM export | ✓ New |
| NetworkX Tanner/factor graph export | ✓ New |
| PyMatching direct graph construction | ✓ New |
| alist / sparse-dict check-matrix interop | ✓ New |
| LFR, threshold, risk-ratio, Tanner plots | ✓ New |
| Docker / docker-compose support | ✓ New |
| 70-test suite (all pass) | ✓ |

---

## Quick start

### Install
```bash
pip install -e .
# or with all optional extras:
pip install ldpc==2.4.1 pymatching==2.4.0 stim h5py
```

### CLI

```bash
# Run a sweep
qec-emulator sweep --code BB72 --decoder bposd --trials 5000

# Multiple codes and decoders
qec-emulator sweep --code BB72 --code BB144 --code BB288 \
    --decoder bposd --decoder lookup \
    --p 0.001 --p 0.004 --p 0.008 --p 0.012 \
    --trials 10000 --output results/sweep --format csv --format json

# From a config file
qec-emulator init-config         # write default config
qec-emulator sweep --config qec_config.yaml

# Verify BB72 certificate
qec-emulator verify

# Start REST API server
qec-emulator server --port 8765

# Plot results
qec-emulator plot results/sweep.csv --type lfr --output lfr.png
qec-emulator plot results/sweep.csv --type threshold
qec-emulator plot results/sweep.csv --type rr

# Show installed components
qec-emulator info
```

### Docker

```bash
docker build -t qec-emulator .
docker run -p 8765:8765 qec-emulator          # API server
docker run qec-emulator qec-emulator info     # check components
```

### REST API

```bash
# Start server
qec-emulator server --port 8765

# Health check
curl http://localhost:8765/

# List codes
curl http://localhost:8765/codes

# Run a simulation
curl -X POST http://localhost:8765/simulate \
  -H "Content-Type: application/json" \
  -d '{"codes":["BB72"],"decoders":["bposd"],"p_values":[0.008],"trials":1000}'

# Submit async job
curl -X POST http://localhost:8765/jobs \
  -H "Content-Type: application/json" \
  -d '{"codes":["BB72","BB144"],"decoders":["bposd","lookup"],"p_values":[0.004,0.008,0.012],"trials":5000}'

# Poll job
curl http://localhost:8765/jobs/{job_id}

# Download results as CSV
curl http://localhost:8765/results/{job_id}?fmt=csv
```

### Python API

```python
import qec_emulator as qec

# Run a sweep
from qec_emulator.scheduler import JobScheduler, SimJob
sched   = JobScheduler(workers=4)
jobs    = sched.build_sweep_jobs(["BB72","BB144"], ["bposd"], [0.004,0.008,0.012], trials=5000)
results = sched.run(jobs)

# Save results
rs = qec.ResultSet(rows=results)
rs.save("results/sweep", formats=["csv","json"])

# Plot
qec.plot_lfr_vs_p(results, output="lfr.png", show=False)
qec.plot_threshold(results)

# Interop
G  = qec.to_networkx_tanner(qec.BB72(), "X")
al = qec.to_alist(qec.BB72().hx)

# Config
cfg = qec.SimulationConfig.load("qec_config.yaml")
```

---

## Architecture (5 layers)

```
Layer 1: codes.py         — BBCode, BB72/BB144/BB288/Steane7, SHA-256 pins
Layer 2: decoders.py      — BPOSDDecoder, MWPMDecoder, LookupDecoder
Layer 3: noise.py         — Phenomenological, RepeatedCycle, CircuitLevel, Hardware
Layer 4: benchmark.py     — run_sweep, run_repeated_cycle, run_threshold_scan, ...
Layer 5: provenance.py    — ProvenanceRecord, ProvenanceLogger

Extensions (v2.0):
         config.py        — YAML/JSON SimulationConfig
         export.py        — ResultSet, CSV/JSON/HDF5
         interop.py       — Stim, NetworkX, alist, sparse
         scheduler.py     — JobScheduler, SimJob, parallel execution
         visualization.py — plot_lfr_vs_p, threshold, risk ratios, Tanner
         api.py           — FastAPI REST server
         cli.py           — Typer CLI with rich output
```

---

## Connecting from a remote desktop

The REST API allows connection from any machine on the same network:

```bash
# On the server machine:
qec-emulator server --host 0.0.0.0 --port 8765

# From your desktop (Python):
import httpx
r = httpx.post("http://SERVER_IP:8765/simulate",
               json={"codes":["BB72"],"decoders":["bposd"],
                     "p_values":[0.008],"trials":1000})
print(r.json())

# From curl:
curl http://SERVER_IP:8765/
```

See docs at `http://SERVER_IP:8765/docs` for the full interactive Swagger UI.
