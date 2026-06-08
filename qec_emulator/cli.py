"""
cli.py — Command-line interface for the QEC Emulator.

Usage
-----
    qec-emulator --help
    qec-emulator sweep --code BB72 --decoder bposd --trials 5000
    qec-emulator sweep --config qec_config.yaml
    qec-emulator verify
    qec-emulator server --port 8765
    qec-emulator export results.csv --format json
    qec-emulator plot results.csv --type lfr

Author: Dr. Saleh H. AlDaajeh  ORCID: 0000-0001-7810-9290
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table   import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

app     = typer.Typer(name="qec-emulator",
                      help="QEC Emulator CLI — BB code decoder benchmarking",
                      add_completion=False)
console = Console()


# ---------------------------------------------------------------------------
# qec-emulator sweep
# ---------------------------------------------------------------------------

@app.command()
def sweep(
    code:    List[str]  = typer.Option(["BB72"], "--code",    "-c",
                            help="BB72|BB144|BB288 (repeat for multiple)"),
    decoder: List[str]  = typer.Option(["bposd"], "--decoder", "-d",
                            help="bposd|mwpm|uf|lookup"),
    p_values:List[float]= typer.Option([0.001,0.004,0.008,0.012], "--p", "-p"),
    trials:  int        = typer.Option(5000,   "--trials",   "-t"),
    rounds:  int        = typer.Option(1,      "--rounds",   "-r"),
    output:  Optional[str] = typer.Option(None, "--output", "-o",
                            help="Save results to file (csv/json auto-detected)"),
    fmt:     List[str]  = typer.Option(["csv"], "--format", "-f",
                            help="Output format: csv|json|hdf5"),
    parallel:bool       = typer.Option(False, "--parallel",
                            help="Use parallel worker pool"),
    workers: int        = typer.Option(4, "--workers", "-w"),
    config:  Optional[str] = typer.Option(None, "--config",
                            help="Load settings from YAML/JSON config file"),
    verbose: bool       = typer.Option(True, "--verbose/--quiet"),
):
    """Run a Monte Carlo sweep over one or more codes and decoders."""

    # Load config file if provided (CLI flags override)
    if config:
        from .config import SimulationConfig
        cfg = SimulationConfig.load(config)
        if not typer.get_default("code"):    code     = [cfg.code.preset or "BB72"]
        if not typer.get_default("decoder"): decoder  = [cfg.decoder.name]
        if not typer.get_default("p"):       p_values = cfg.noise.p_values
        if not typer.get_default("trials"):  trials   = cfg.runner.trials
        if not typer.get_default("rounds"):  rounds   = cfg.noise.rounds
        if not typer.get_default("output"):  output   = Path(cfg.export.output_dir)/"run"
        if not typer.get_default("format"):  fmt      = cfg.export.formats
        parallel = cfg.runner.parallel
        workers  = cfg.runner.workers

    from .scheduler import SimJob, _run_one_job, JobScheduler
    from .export    import ResultSet

    sched = JobScheduler(workers=workers)
    jobs  = sched.build_sweep_jobs(
        codes=list(code), decoders=list(decoder),
        p_values=list(p_values), trials=trials, rounds=rounds)

    console.print(f"[bold green]QEC Emulator v1.2.3[/bold green]")
    console.print(f"  Jobs: {len(jobs)}, trials: {trials}, parallel: {parallel}")
    console.print()

    results = []
    t0 = time.time()

    if parallel:
        with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                      BarColumn(), TimeElapsedColumn(), console=console) as prog:
            task = prog.add_task("Simulating...", total=len(jobs))
            def on_result(r):
                prog.advance(task)
                results.append(r)
            for r in sched.run_streaming(jobs):
                prog.advance(task)
                results.append(r)
    else:
        with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                      BarColumn(), TimeElapsedColumn(), console=console) as prog:
            task = prog.add_task("Simulating...", total=len(jobs))
            for job in jobs:
                r = _run_one_job(job)
                results.append(r)
                prog.advance(task)
                if verbose:
                    prog.print(
                        f"  {r['code']:6s} {r['decoder']:10s} p={r['p']:.4f}: "
                        f"{r['failures']:5d}/{r['trials']}  LFR={r['lfr']:.5f}  "
                        f"[{r['ci_lo']:.5f},{r['ci_hi']:.5f}]")

    elapsed = time.time() - t0
    _print_results_table(results)
    console.print(f"\n[dim]Total elapsed: {elapsed:.1f}s[/dim]")

    if output:
        rs = ResultSet(rows=results)
        saved = rs.save(output, formats=list(fmt))
        for f, p in saved.items():
            console.print(f"[green]Saved {f.upper()}:[/green] {p}")


# ---------------------------------------------------------------------------
# qec-emulator verify
# ---------------------------------------------------------------------------

@app.command()
def verify(
    code: str = typer.Option("BB72", "--code", "-c",
                              help="Code to certify (currently only BB72)"),
    output: Optional[str] = typer.Option(None, "--output", "-o"),
):
    """Run the finite-instance BB72 certificate (O1–O4)."""
    console.print("[bold]BB72 Certificate Verification[/bold]")
    from .codes import BB72, gf2_rref, in_rowspace
    import itertools, hashlib

    code_obj = BB72()
    n, n2 = code_obj.n, code_obj.n2
    HX, HZ = code_obj.hx, code_obj.hz

    results = {}

    # O1
    comm = bool(((HX @ HZ.T) & 1 == 0).all())
    console.print(f"  O1 Commutation:  {'[green]PASS[/green]' if comm else '[red]FAIL[/red]'}")
    results["O1"] = comm

    # O2
    k = code_obj.k
    console.print(f"  O2 Dimension k={k}: {'[green]PASS[/green]' if k==12 else '[red]FAIL[/red]'}")
    results["O2"] = k == 12

    # O3 SHA
    exp_hx = "7ab0973bfd02e399d69728d26e67dbfa0e95bbc6ad2c5cf8bf309d667244a5a8"
    exp_hz = "267f345c09710ec0cfad1e49e0786b2636ab052fbfb660e5a76ae9919ba22614"
    sha_ok = (code_obj.hx_sha256()==exp_hx and code_obj.hz_sha256()==exp_hz)
    console.print(f"  O3a SHA-256:     {'[green]PASS[/green]' if sha_ok else '[red]FAIL[/red]'}")
    results["O3a"] = sha_ok

    # O3b MitM
    from .codes import syndrome_from_mask
    HX_rref, HX_piv = gf2_rref(HX); HZ_rref, HZ_piv = gf2_rref(HZ)

    def hx_col(j):
        return int(sum((int(HX[i,j])&1)<<i for i in range(n2)))
    def hz_col(j):
        return int(sum((int(HZ[i,j])&1)<<i for i in range(n2)))

    hx_cols = [hx_col(j) for j in range(n)]
    hz_cols = [hz_col(j) for j in range(n)]

    def mitm_lb(col_syns, H_rref, H_piv):
        import numpy as np
        left = {}
        for w in range(3):
            for comb in itertools.combinations(range(n), w):
                syn = mask = 0
                for j in comb: syn ^= col_syns[j]; mask |= 1<<j
                left.setdefault(syn,[]).append(mask)
        witnesses = []
        for w in range(4):
            for comb in itertools.combinations(range(n), w):
                syn = rmask = 0
                for j in comb: syn ^= col_syns[j]; rmask |= 1<<j
                for lmask in left.get(syn,[]):
                    if lmask & rmask: continue
                    full = lmask | rmask
                    if full == 0: continue
                    vec = np.array([(full>>i)&1 for i in range(n)],dtype=np.uint8)
                    if not in_rowspace(vec, H_rref, H_piv):
                        witnesses.append(full)
        return witnesses

    with console.status("Running MitM d≥6 check (Z)..."):
        wz = mitm_lb(hx_cols, HZ_rref, HZ_piv)
    with console.status("Running MitM d≥6 check (X)..."):
        wx = mitm_lb(hz_cols, HX_rref, HX_piv)

    d6 = len(wz)==0 and len(wx)==0
    console.print(f"  O3b MitM d≥6:   {'[green]PASS[/green]' if d6 else '[red]FAIL[/red]'} "
                  f"(Z witnesses={len(wz)}, X witnesses={len(wx)})")
    results["O3b"] = d6

    # O4
    def build_lut(col_syns, radius):
        table = {0:0}
        for w in range(1,radius+1):
            for comb in itertools.combinations(range(n),w):
                syn=mask=0
                for j in comb: syn^=col_syns[j]; mask|=1<<j
                if syn not in table or bin(table[syn]).count('1')>bin(mask).count('1'):
                    table[syn]=mask
        return table
    lz = build_lut(hx_cols, 2); lx = build_lut(hz_cols, 2)
    import numpy as np
    fail = pats = 0
    for w in range(3):
        for bits in itertools.combinations(range(n), w):
            mask=sum(1<<j for j in bits)
            sz=syndrome_from_mask(mask,hx_cols); sx=syndrome_from_mask(mask,hz_cols)
            cz=lz.get(sz,0); cx=lx.get(sx,0)
            rz=mask^cz; rx=mask^cx
            vz=np.array([(rz>>i)&1 for i in range(n)],dtype=np.uint8)
            vx=np.array([(rx>>i)&1 for i in range(n)],dtype=np.uint8)
            if not(in_rowspace(vz,HZ_rref,HZ_piv) and
                   in_rowspace(vx,HX_rref,HX_piv)): fail+=1
            pats+=1
    console.print(f"  O4 t=2 radius:  {'[green]PASS[/green]' if fail==0 else '[red]FAIL[/red]'} "
                  f"({pats} patterns, {fail} failures)")
    results["O4"] = fail==0

    all_pass = all(results.values())
    console.print()
    console.print(f"[bold {'green' if all_pass else 'red'}]"
                  f"{'ALL FOUR OBLIGATIONS PASS' if all_pass else 'FAILURES DETECTED'}[/bold]")

    if output:
        Path(output).write_text(json.dumps(results, indent=2))
    sys.exit(0 if all_pass else 1)


# ---------------------------------------------------------------------------
# qec-emulator server
# ---------------------------------------------------------------------------

@app.command()
def server(
    host:   str = typer.Option("0.0.0.0", "--host"),
    port:   int = typer.Option(8765,      "--port", "-p"),
    reload: bool= typer.Option(False,     "--reload"),
):
    """Start the REST API server."""
    console.print(f"[bold green]QEC Emulator API server[/bold green] "
                  f"listening on http://{host}:{port}")
    console.print(f"  API docs:  http://{host}:{port}/docs")
    console.print(f"  Health:    http://{host}:{port}/")
    from .api import run_server
    run_server(host=host, port=port, reload=reload)


# ---------------------------------------------------------------------------
# qec-emulator export
# ---------------------------------------------------------------------------

@app.command()
def export(
    input_file:  str = typer.Argument(..., help="Input CSV or JSON results file"),
    output_file: str = typer.Argument(..., help="Output file path"),
    fmt:         str = typer.Option("json", "--format", "-f", help="csv|json|hdf5"),
):
    """Convert results between formats."""
    from .export import ResultSet
    rs = ResultSet.load(input_file)
    rs.save(Path(output_file).with_suffix(''), formats=[fmt])
    console.print(f"[green]Exported {len(rs)} rows to {output_file}[/green]")


# ---------------------------------------------------------------------------
# qec-emulator plot
# ---------------------------------------------------------------------------

@app.command()
def plot(
    input_file: str = typer.Argument(..., help="CSV or JSON results file"),
    plot_type:  str = typer.Option("lfr", "--type", "-t",
                                   help="lfr|threshold|rr|tanner"),
    code_filter: Optional[str] = typer.Option(None, "--code"),
    decoder_filter: Optional[str] = typer.Option(None, "--decoder"),
    output: Optional[str] = typer.Option(None, "--output", "-o"),
    no_show: bool = typer.Option(False, "--no-show"),
):
    """Generate a plot from results data."""
    from .export        import ResultSet
    from .visualization import plot_lfr_vs_p, plot_threshold, plot_risk_ratios

    rs = ResultSet.load(input_file)
    codes    = [code_filter]    if code_filter    else None
    decoders = [decoder_filter] if decoder_filter else None

    if plot_type == "lfr":
        plot_lfr_vs_p(rs.rows, codes=codes, decoders=decoders,
                      output=output, show=not no_show)
    elif plot_type == "threshold":
        d = decoders[0] if decoders else "BP+OSD"
        plot_threshold(rs.rows, decoder=d, output=output, show=not no_show)
    elif plot_type == "rr":
        plot_risk_ratios(rs.rows, output=output, show=not no_show)
    else:
        console.print(f"[red]Unknown plot type: {plot_type}[/red]")


# ---------------------------------------------------------------------------
# qec-emulator init-config
# ---------------------------------------------------------------------------

@app.command(name="init-config")
def init_config(
    output: str = typer.Option("qec_config.yaml", "--output", "-o"),
):
    """Write a default configuration file."""
    from .config import write_default_config
    p = write_default_config(output)
    console.print(f"[green]Default config written to {p}[/green]")
    console.print("Edit it and run:  qec-emulator sweep --config qec_config.yaml")


# ---------------------------------------------------------------------------
# qec-emulator info
# ---------------------------------------------------------------------------

@app.command()
def info():
    """Show version, installed decoders, and available codes."""
    console.print("[bold]QEC Emulator v1.2.3[/bold]")
    console.print("  Author:  Dr. Saleh H. AlDaajeh")
    console.print("  ORCID:   0000-0001-7810-9290")
    console.print()

    t = Table("Component", "Status", show_header=True)
    for lib, extra in [
        ("ldpc",       "BpOsdDecoder (BP+OSD)"),
        ("pymatching", "MWPMDecoder"),
        ("stim",       "Stim circuit interop"),
        ("networkx",   "Tanner/factor graph export"),
        ("matplotlib", "Visualization"),
        ("fastapi",    "REST API server"),
        ("h5py",       "HDF5 export"),
    ]:
        try:
            __import__(lib)
            t.add_row(f"{lib}", f"[green]available[/green]  ({extra})")
        except ImportError:
            t.add_row(f"{lib}", f"[yellow]not installed[/yellow]  ({extra})")
    console.print(t)


# ---------------------------------------------------------------------------
# _print_results_table
# ---------------------------------------------------------------------------

def _print_results_table(results: list):
    if not results: return
    t = Table("Code","Decoder","p","Failures","LFR","95% CI",
              show_header=True, header_style="bold cyan")
    for r in results:
        lfr = r.get('lfr', 0)
        lo  = r.get('ci_lo', lfr)
        hi  = r.get('ci_hi', lfr)
        fail= r.get('failures', -1)
        lfr_color = "red" if lfr > 0.1 else "yellow" if lfr > 0.01 else "green"
        t.add_row(
            r.get('code','?'), r.get('decoder','?'),
            f"{r.get('p',0):.4f}",
            str(fail),
            f"[{lfr_color}]{lfr:.5f}[/{lfr_color}]",
            f"[{lo:.5f}, {hi:.5f}]",
        )
    console.print(t)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app()

if __name__ == "__main__":
    main()
