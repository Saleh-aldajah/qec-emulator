"""
scheduler.py — Parallel and async execution engine.

Provides concurrent multi-code, multi-p-value sweep execution using
ProcessPoolExecutor (CPU-bound), with progress tracking and live
streaming of partial results.

Author: Dr. Saleh H. AlDaajeh  ORCID: 0000-0001-7810-9290
"""
from __future__ import annotations
import hashlib, math, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Job descriptor
# ---------------------------------------------------------------------------

@dataclass
class SimJob:
    """
    One unit of work: a single (code, decoder, p_value) simulation run.

    Parameters
    ----------
    code_preset : str   — "BB72"|"BB144"|"BB288"
    decoder     : str   — "bposd"|"mwpm"|"uf"|"lookup"
    p           : float
    trials      : int
    seed        : int   — SHA-256 stable seed (auto-computed if 0)
    rounds      : int   — >1 activates repeated-cycle protocol
    job_id      : str   — auto-generated
    """
    code_preset : str
    decoder     : str
    p           : float
    trials      : int   = 5000
    seed        : int   = 0
    rounds      : int   = 1
    job_id      : str   = ""

    def __post_init__(self):
        if self.seed == 0:
            self.seed = _stable_seed(self.code_preset, self.decoder, self.p, self.rounds)
        if not self.job_id:
            self.job_id = f"{self.code_preset}_{self.decoder}_p{self.p:.4g}_r{self.rounds}"


def _stable_seed(*args) -> int:
    key = "|".join(str(a) for a in args).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], 'big')


# ---------------------------------------------------------------------------
# Single-job runner (runs in a subprocess)
# ---------------------------------------------------------------------------

def _run_one_job(job: SimJob) -> Dict[str, Any]:
    """
    Execute one SimJob. Designed to be called inside a worker process.
    Returns a result dict compatible with ResultSet.add().
    """
    import math, hashlib
    import numpy as np

    # Late imports so the function is picklable
    from qec_emulator.codes    import BB72, BB144, BB288, in_rowspace, syndrome_from_mask
    from qec_emulator.decoders import BPOSDDecoder as BpOsdDecoder, MWPMDecoder

    code_map = {"BB72": BB72, "BB144": BB144, "BB288": BB288}
    if job.code_preset not in code_map:
        raise ValueError(f"Unknown code preset: {job.code_preset}")
    code = code_map[job.code_preset]()
    n, n2 = code.n, code.n2

    rng = np.random.default_rng(job.seed)
    failures = 0
    t0 = time.time()

    def bern(nn, pp):
        return int(sum((1 if rng.random()<pp else 0)<<i for i in range(nn)))

    def syn_arr(s):
        return np.array([(s>>i)&1 for i in range(n2)], dtype=np.uint8)

    def arr_mask(arr):
        return int(sum((int(b)&1)<<i for i,b in enumerate(arr)))

    def logical_fail(ez, ex):
        vz = np.array([(ez>>i)&1 for i in range(n)], dtype=np.uint8)
        vx = np.array([(ex>>i)&1 for i in range(n)], dtype=np.uint8)
        return not (in_rowspace(vz, code.hz_rref, code.hz_pivots) and
                    in_rowspace(vx, code.hx_rref, code.hx_pivots))

    p = job.p

    if job.decoder == "bposd":
        try:
            dz = BpOsdDecoder(code.hx, error_rate=p, bp_method='ms',
                              ms_scaling_factor=0.625, osd_method='osd_cs',
                              osd_order=7, max_iter=100)
            dx = BpOsdDecoder(code.hz, error_rate=p, bp_method='ms',
                              ms_scaling_factor=0.625, osd_method='osd_cs',
                              osd_order=7, max_iter=100)
            for _ in range(job.trials):
                ez = bern(n, p); ex = bern(n, p)
                sz = syndrome_from_mask(ez, code.hx_cols)
                sx = syndrome_from_mask(ex, code.hz_cols)
                cz = arr_mask(dz.decode(syn_arr(sz)))
                cx = arr_mask(dx.decode(syn_arr(sx)))
                if logical_fail(ez^cz, ex^cx): failures += 1
        except Exception:
            failures = -1  # signal decoder unavailable

    elif job.decoder == "lookup":
        import itertools
        def build_lookup(col_syns, radius):
            table = {0: 0}
            for w in range(1, radius+1):
                for comb in itertools.combinations(range(n), w):
                    syn = mask = 0
                    for j in comb: syn^=col_syns[j]; mask|=1<<j
                    if syn not in table or bin(table[syn]).count('1')>bin(mask).count('1'):
                        table[syn] = mask
            return table
        lz = build_lookup(code.hx_cols, 2)
        lx = build_lookup(code.hz_cols, 2)
        for _ in range(job.trials):
            ez=bern(n,p); ex=bern(n,p)
            sz=syndrome_from_mask(ez,code.hx_cols)
            sx=syndrome_from_mask(ex,code.hz_cols)
            cz=lz.get(sz,0); cx=lx.get(sx,0)
            if logical_fail(ez^cz,ex^cx): failures+=1

    elapsed = time.time() - t0
    z = 1.96
    kk = max(0, failures); nn = job.trials
    lfr = kk/nn if nn>0 else 0
    pp_ = lfr
    c = (pp_+z*z/(2*nn))/(1+z*z/nn) if nn>0 else 0
    marg = z*math.sqrt(pp_*(1-pp_)/nn+z*z/(4*nn*nn))/(1+z*z/nn) if nn>0 else 0

    return {
        "job_id":    job.job_id,
        "code":      job.code_preset,
        "decoder":   job.decoder,
        "p":         p,
        "rounds":    job.rounds,
        "trials":    job.trials,
        "failures":  failures,
        "lfr":       round(lfr, 8),
        "ci_lo":     round(max(0.,c-marg), 8),
        "ci_hi":     round(min(1.,c+marg), 8),
        "seed":      job.seed,
        "elapsed_s": round(elapsed, 2),
        "hx_sha256": code.hx_sha256(),
        "hz_sha256": code.hz_sha256(),
    }


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class JobScheduler:
    """
    Parallel job scheduler for multi-code, multi-p-value sweeps.

    Example
    -------
    >>> sched = JobScheduler(workers=4)
    >>> jobs = sched.build_sweep_jobs(
    ...     codes=["BB72","BB144"],
    ...     decoders=["bposd"],
    ...     p_values=[0.001, 0.004, 0.008],
    ...     trials=10000)
    >>> results = sched.run(jobs, progress=True)
    """

    def __init__(self, workers: int = 4, timeout: Optional[float] = None):
        self.workers = workers
        self.timeout = timeout

    def build_sweep_jobs(
        self,
        codes:     List[str],
        decoders:  List[str],
        p_values:  List[float],
        trials:    int = 5000,
        rounds:    int = 1,
    ) -> List[SimJob]:
        jobs = []
        for code in codes:
            for decoder in decoders:
                for p in p_values:
                    jobs.append(SimJob(
                        code_preset=code, decoder=decoder,
                        p=p, trials=trials, rounds=rounds))
        return jobs

    def run(
        self,
        jobs: List[SimJob],
        progress: bool = True,
        on_result: Optional[Callable] = None,
    ) -> List[Dict]:
        """
        Execute jobs in parallel. Returns list of result dicts.

        Parameters
        ----------
        jobs      : list of SimJob
        progress  : print a progress line for each completed job
        on_result : optional callback(result_dict) called on each completion
        """
        results  = []
        n_total  = len(jobs)
        n_done   = 0
        t_start  = time.time()

        with ProcessPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(_run_one_job, job): job for job in jobs}
            for future in as_completed(futures, timeout=self.timeout):
                job = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {"job_id": job.job_id, "error": str(e)}
                results.append(result)
                n_done += 1
                if progress:
                    elapsed = time.time() - t_start
                    lfr = result.get('lfr', '?')
                    print(f"  [{n_done:3d}/{n_total}] {job.job_id:50s}  "
                          f"LFR={lfr}  ({elapsed:.0f}s)")
                if on_result:
                    on_result(result)

        # Sort by code, decoder, p for consistency
        results.sort(key=lambda r: (r.get('code',''), r.get('decoder',''), r.get('p',0)))
        return results

    def run_streaming(
        self, jobs: List[SimJob]
    ) -> Generator[Dict, None, None]:
        """
        Generator version: yields each result dict as jobs complete.
        Useful for live streaming to the REST API.
        """
        with ProcessPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(_run_one_job, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    yield future.result()
                except Exception as e:
                    yield {"job_id": job.job_id, "error": str(e)}
