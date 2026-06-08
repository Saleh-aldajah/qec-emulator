"""
api.py — FastAPI REST/HTTP server for the QEC Emulator.

Provides a web API analogous to Cooja's socket/RMI interface,
enabling remote simulation control from any client (desktop app,
Jupyter, curl, etc.).

Endpoints
---------
GET  /                      — health + version
GET  /codes                 — list available code presets
POST /simulate              — run a sweep job (synchronous)
POST /jobs                  — submit async job, returns job_id
GET  /jobs/{job_id}         — poll job status / results
GET  /jobs/{job_id}/stream  — SSE streaming results
DELETE /jobs/{job_id}       — cancel job
GET  /results/{job_id}      — download results as CSV or JSON
POST /verify                — run verify_certificate on BB72

Author: Dr. Saleh H. AlDaajeh  ORCID: 0000-0001-7810-9290
"""
from __future__ import annotations
import asyncio, hashlib, json, time, uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse, PlainTextResponse, JSONResponse
    from pydantic import BaseModel, Field
    _FASTAPI = True
except ImportError:
    _FASTAPI = False

from .codes     import BB72, BB144, BB288
from .scheduler import SimJob, _run_one_job, JobScheduler
from .config    import SimulationConfig
from .export    import ResultSet


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

if _FASTAPI:
    class SimulateRequest(BaseModel):
        codes:    List[str]  = Field(["BB72"], description="BB72|BB144|BB288")
        decoders: List[str]  = Field(["bposd"], description="bposd|lookup")
        p_values: List[float]= Field([0.001, 0.004, 0.008, 0.012])
        trials:   int        = Field(1000, ge=1, le=100000)
        rounds:   int        = Field(1,    ge=1, le=50)
        parallel: bool       = Field(False)
        workers:  int        = Field(2,    ge=1, le=8)

    class JobStatus(BaseModel):
        job_id:   str
        status:   str   # queued|running|complete|failed
        n_done:   int
        n_total:  int
        results:  List[Dict] = []
        error:    Optional[str] = None

    class ConfigRequest(BaseModel):
        config_yaml: str


# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------

_jobs: Dict[str, Dict] = {}
_executor = ProcessPoolExecutor(max_workers=4)


def _make_app() -> "FastAPI":
    if not _FASTAPI:
        raise ImportError("fastapi + uvicorn required: pip install fastapi uvicorn")

    app = FastAPI(
        title="QEC Emulator API",
        description=(
            "REST interface for the QEC Emulator — BB code benchmarking, "
            "decoder comparison, and reproducible simulation.\n\n"
            "Author: Dr. Saleh H. AlDaajeh  ORCID: 0000-0001-7810-9290"
        ),
        version="1.2.3",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── GET / ──────────────────────────────────────────────────────────────

    @app.get("/", tags=["meta"])
    def root():
        return {
            "status":  "ok",
            "service": "QEC Emulator API",
            "version": "1.2.3",
            "author":  "Dr. Saleh H. AlDaajeh",
            "orcid":   "0000-0001-7810-9290",
            "docs":    "/docs",
        }

    # ── GET /codes ─────────────────────────────────────────────────────────

    @app.get("/codes", tags=["codes"])
    def list_codes():
        return {
            "codes": [
                {"name": "BB72",  "n": 72,  "k": 12, "d_lower": 6,
                 "description": "BB[[72,12,6]] ell=6 m=6"},
                {"name": "BB144", "n": 144, "k": 12, "d_lower": 12,
                 "description": "BB[[144,12,12]] ell=12 m=6"},
                {"name": "BB288", "n": 288, "k": 12, "d_lower": 18,
                 "description": "BB[[288,12,18]] ell=12 m=12"},
            ]
        }

    # ── GET /codes/{name} ─────────────────────────────────────────────────

    @app.get("/codes/{name}", tags=["codes"])
    def get_code(name: str):
        code_map = {"BB72": BB72, "BB144": BB144, "BB288": BB288}
        if name not in code_map:
            raise HTTPException(404, f"Unknown code: {name}")
        c = code_map[name]()
        return c.to_dict()

    # ── POST /simulate (synchronous, small jobs) ───────────────────────────

    @app.post("/simulate", tags=["simulation"])
    def simulate(req: SimulateRequest):
        sched = JobScheduler(workers=req.workers if req.parallel else 1)
        jobs  = sched.build_sweep_jobs(
            codes=req.codes, decoders=req.decoders,
            p_values=req.p_values, trials=req.trials, rounds=req.rounds)

        if req.parallel and req.workers > 1:
            results = sched.run(jobs, progress=False)
        else:
            results = [_run_one_job(j) for j in jobs]

        rs = ResultSet(rows=results)
        return {
            "n_jobs":    len(results),
            "integrity": rs.integrity_hash(),
            "results":   results,
        }

    # ── POST /jobs (async submission) ─────────────────────────────────────

    @app.post("/jobs", tags=["jobs"])
    def submit_job(req: SimulateRequest, background: BackgroundTasks):
        job_id = str(uuid.uuid4())[:8]
        sched  = JobScheduler(workers=min(req.workers, 4))
        jobs   = sched.build_sweep_jobs(
            codes=req.codes, decoders=req.decoders,
            p_values=req.p_values, trials=req.trials, rounds=req.rounds)

        _jobs[job_id] = {
            "status":  "queued",
            "n_done":  0,
            "n_total": len(jobs),
            "results": [],
            "error":   None,
        }

        def _worker():
            try:
                _jobs[job_id]["status"] = "running"
                for result in sched.run_streaming(jobs):
                    _jobs[job_id]["results"].append(result)
                    _jobs[job_id]["n_done"] += 1
                _jobs[job_id]["status"] = "complete"
            except Exception as e:
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["error"]  = str(e)

        background.add_task(_worker)
        return {"job_id": job_id, "n_jobs": len(jobs), "status": "queued"}

    # ── GET /jobs/{job_id} ─────────────────────────────────────────────────

    @app.get("/jobs/{job_id}", tags=["jobs"])
    def get_job(job_id: str):
        if job_id not in _jobs:
            raise HTTPException(404, f"Job {job_id} not found")
        j = _jobs[job_id]
        return {
            "job_id":  job_id,
            "status":  j["status"],
            "n_done":  j["n_done"],
            "n_total": j["n_total"],
            "results": j["results"],
            "error":   j["error"],
        }

    # ── DELETE /jobs/{job_id} ──────────────────────────────────────────────

    @app.delete("/jobs/{job_id}", tags=["jobs"])
    def cancel_job(job_id: str):
        if job_id not in _jobs:
            raise HTTPException(404, f"Job {job_id} not found")
        _jobs[job_id]["status"] = "cancelled"
        return {"job_id": job_id, "status": "cancelled"}

    # ── GET /results/{job_id} ─────────────────────────────────────────────

    @app.get("/results/{job_id}", tags=["results"])
    def download_results(job_id: str, fmt: str = "json"):
        if job_id not in _jobs:
            raise HTTPException(404, f"Job {job_id} not found")
        rows = _jobs[job_id]["results"]
        rs   = ResultSet(rows=rows)
        if fmt == "csv":
            import io, csv as csv_mod
            buf = io.StringIO()
            if rows:
                w = csv_mod.DictWriter(buf, fieldnames=rows[0].keys())
                w.writeheader(); w.writerows(rows)
            return PlainTextResponse(buf.getvalue(), media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={job_id}.csv"})
        return JSONResponse({"meta": rs.meta, "rows": rows})

    # ── GET /jobs/{job_id}/stream (Server-Sent Events) ────────────────────

    @app.get("/jobs/{job_id}/stream", tags=["jobs"])
    async def stream_results(job_id: str):
        """Stream results as SSE (text/event-stream)."""
        if job_id not in _jobs:
            raise HTTPException(404, f"Job {job_id} not found")

        async def event_gen():
            seen = 0
            while True:
                job  = _jobs[job_id]
                rows = job["results"]
                for row in rows[seen:]:
                    yield f"data: {json.dumps(row)}\n\n"
                    seen += 1
                if job["status"] in ("complete", "failed", "cancelled"):
                    yield f"data: {json.dumps({'event':'done','status':job['status']})}\n\n"
                    break
                await asyncio.sleep(0.5)

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    # ── POST /verify ──────────────────────────────────────────────────────

    @app.post("/verify", tags=["certificate"])
    def run_verification():
        """Run the BB72 finite-instance certificate (O1–O4)."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "qec_emulator.cli", "verify"],
            capture_output=True, text=True, timeout=120)
        return {
            "exit_code": result.returncode,
            "stdout":    result.stdout,
            "stderr":    result.stderr,
            "pass":      result.returncode == 0,
        }

    # ── POST /config ──────────────────────────────────────────────────────

    @app.post("/config", tags=["meta"])
    def parse_config(req: ConfigRequest):
        """Parse and validate a YAML config string."""
        try:
            cfg = SimulationConfig.from_yaml(req.config_yaml)
            return {"valid": True, "parsed": cfg.to_dict()}
        except Exception as e:
            return {"valid": False, "error": str(e)}

    return app


def create_app():
    """Public factory — call this to get the FastAPI app."""
    return _make_app()


def run_server(host: str = "0.0.0.0", port: int = 8765, reload: bool = False):
    """Launch the API server (blocking)."""
    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn is required: pip install uvicorn")
    app = create_app()
    uvicorn.run(app, host=host, port=port, reload=reload, log_level="info")
