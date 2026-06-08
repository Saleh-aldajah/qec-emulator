"""
visualization.py — Result plotting and Tanner graph rendering.

Provides publication-quality plots:
  - LFR vs p_physical (log-log)
  - Threshold crossing (multi-code finite-size scaling)
  - Risk ratio bar chart
  - Tanner graph layout

Author: Dr. Saleh H. AlDaajeh  ORCID: 0000-0001-7810-9290
"""
from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Colour / style palette
# ---------------------------------------------------------------------------

PALETTE = {
    "BB72":    "#1f77b4",
    "BB144":   "#ff7f0e",
    "BB288":   "#2ca02c",
    "BP+OSD":  "#1f77b4",
    "MWPM-DEM":"#d62728",
    "UF-BFS":  "#9467bd",
    "lookup":  "#8c564b",
}
MARKERS = {"BB72": "o", "BB144": "s", "BB288": "^"}


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError("matplotlib is required: pip install matplotlib")


# ---------------------------------------------------------------------------
# LFR vs p plot
# ---------------------------------------------------------------------------

def plot_lfr_vs_p(
    data: List[Dict],
    codes: Optional[List[str]] = None,
    decoders: Optional[List[str]] = None,
    title: str = "Logical Failure Rate vs Physical Error Rate",
    output: Optional[Union[str, Path]] = None,
    show: bool = True,
    loglog: bool = True,
) -> "matplotlib.figure.Figure":
    """
    Plot LFR vs p for one or more (code, decoder) pairs.

    Parameters
    ----------
    data     : list of result dicts (from ResultSet.rows or scheduler output)
    codes    : filter to these code names (None = all)
    decoders : filter to these decoder names (None = all)
    title    : plot title
    output   : if provided, save figure to this path
    show     : call plt.show()
    loglog   : use log-log scale

    Returns
    -------
    matplotlib Figure
    """
    plt = _require_matplotlib()

    codes_set    = set(codes)    if codes    else None
    decoders_set = set(decoders) if decoders else None

    # Group by (code, decoder)
    groups: Dict[Tuple, List] = {}
    for r in data:
        c = r.get('code','?'); d = r.get('decoder','?')
        if codes_set    and c not in codes_set:    continue
        if decoders_set and d not in decoders_set: continue
        key = (c, d)
        groups.setdefault(key, []).append(r)

    fig, ax = plt.subplots(figsize=(7, 5))

    for (code, decoder), rows in sorted(groups.items()):
        rows_sorted = sorted(rows, key=lambda r: r.get('p', 0))
        ps   = [r['p']        for r in rows_sorted]
        lfrs = [r.get('lfr',0)         for r in rows_sorted]
        lfrs_plot = [max(1e-6, v) for v in lfrs]
        lo   = [max(0.0, min(lp, lp - r.get('ci_lo', 0.0))) for r, lp in zip(rows_sorted, lfrs_plot)]
        hi   = [max(0.0, r.get('ci_hi', lp) - r.get('lfr', lp)) for r, lp in zip(rows_sorted, lfrs_plot)]
        color  = PALETTE.get(code, PALETTE.get(decoder, None))
        color  = PALETTE.get(code, PALETTE.get(decoder, None))
        marker = MARKERS.get(code, 'o')
        ls     = '--' if 'MWPM' in decoder or 'UF' in decoder else '-'
        label  = f"{code} / {decoder}"
        # Clamp error bars so they don't go below the plot floor (1e-6)
        lo_clamped = [max(0.0, min(lv, lp)) for lv, lp in zip(lo, lfrs_plot)]
        ax.errorbar(ps, lfrs_plot, yerr=[lo_clamped, hi],
                    fmt=f"{marker}{ls}", color=color,
                    label=label, capsize=3, markersize=5, linewidth=1.5)

    if loglog:
        ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel("Physical error rate $p$", fontsize=11)
    ax.set_ylabel("Logical failure rate (LFR)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches='tight')
    if show:
        try:
            plt.show()
        except Exception:
            pass
    return fig


# ---------------------------------------------------------------------------
# Threshold / finite-size scaling
# ---------------------------------------------------------------------------

def plot_threshold(
    data: List[Dict],
    decoder: str = "BP+OSD",
    title: str = "Finite-size scaling — threshold crossing",
    output: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> "matplotlib.figure.Figure":
    """
    Plot LFR vs p for multiple codes on a single axes to show threshold crossing.
    Each code is a separate curve; the crossing point is the pseudo-threshold.
    """
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 5))

    groups: Dict[str, List] = {}
    for r in data:
        if r.get('decoder') != decoder: continue
        groups.setdefault(r.get('code','?'), []).append(r)

    for code, rows in sorted(groups.items()):
        rows_sorted = sorted(rows, key=lambda r: r.get('p', 0))
        ps   = [r['p'] for r in rows_sorted]
        lfrs = [max(1e-7, r.get('lfr', 0)) for r in rows_sorted]
        ax.semilogy(ps, lfrs, f"{MARKERS.get(code,'o')}-",
                    color=PALETTE.get(code), label=code,
                    markersize=5, linewidth=1.5)

    ax.set_xlabel("Physical error rate $p$", fontsize=11)
    ax.set_ylabel("Logical failure rate (LFR)", fontsize=11)
    ax.set_title(f"{title}\nDecoder: {decoder}", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches='tight')
    if show:
        try: plt.show()
        except: pass
    return fig


# ---------------------------------------------------------------------------
# Risk ratio bar chart
# ---------------------------------------------------------------------------

def plot_risk_ratios(
    data: List[Dict],
    baseline_decoder: str = "BP+OSD",
    compare_decoder:  str = "MWPM-DEM",
    p_value: float = 0.008,
    output: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> "matplotlib.figure.Figure":
    """
    Bar chart of Katz risk ratios (compare / baseline) at a fixed p.
    Infinite ratios (baseline = 0) are shown with a hatched bar + 'inf' label.
    """
    plt = _require_matplotlib()

    codes = []
    ratios = []; lower = []; upper = []

    code_base = {r.get('code'): r for r in data
                 if r.get('decoder')==baseline_decoder and abs(r.get('p',0)-p_value)<1e-9}
    code_comp = {r.get('code'): r for r in data
                 if r.get('decoder')==compare_decoder and abs(r.get('p',0)-p_value)<1e-9}

    for code in sorted(set(code_base) & set(code_comp)):
        rb = code_base[code]; rc = code_comp[code]
        kb = rb.get('failures',0); kc = rc.get('failures',0)
        n  = rb.get('trials', 1)
        if kb == 0:
            ratios.append(float('inf')); lower.append(0); upper.append(0)
        else:
            rr = (kc/n) / (kb/n)
            z  = 1.96
            se = math.sqrt(1/kc - 1/n + 1/kb - 1/n) if kc > 0 else 0
            lo = rr * math.exp(-z*se) if se > 0 else rr
            hi = rr * math.exp( z*se) if se > 0 else rr
            ratios.append(rr); lower.append(rr-lo); upper.append(hi-rr)
        codes.append(code)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(codes))
    finite = [(i, r) for i, r in enumerate(ratios) if not math.isinf(r)]
    inf_idx= [i for i, r in enumerate(ratios) if math.isinf(r)]

    if finite:
        fi, fr = zip(*finite)
        lo_f = [lower[i] for i in fi]
        hi_f = [upper[i] for i in fi]
        ax.bar(fi, fr, color='steelblue', alpha=0.8, label='finite RR')
        ax.errorbar(list(fi), fr, yerr=[lo_f, hi_f],
                    fmt='none', color='black', capsize=4)

    max_rr = max((r for r in ratios if not math.isinf(r)), default=10)
    for i in inf_idx:
        ax.bar([i], [max_rr * 1.2], color='crimson', alpha=0.5, hatch='//',
               label='infinite RR (baseline k=0)')
        ax.text(i, max_rr * 1.25, '∞', ha='center', fontsize=12, color='crimson')

    ax.set_xticks(list(x)); ax.set_xticklabels(codes, fontsize=10)
    ax.set_ylabel(f"Risk ratio ({compare_decoder} / {baseline_decoder})")
    ax.set_title(f"Risk ratios at p={p_value}")
    handles = [h for h, _ in (ax.get_legend_handles_labels()[0], ax.get_legend_handles_labels()[1])]
    ax.legend(fontsize=9)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches='tight')
    if show:
        try: plt.show()
        except: pass
    return fig


# ---------------------------------------------------------------------------
# Tanner graph layout
# ---------------------------------------------------------------------------

def plot_tanner_graph(
    code: "BBCode",
    which: str = "X",
    max_nodes: int = 60,
    output: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> "matplotlib.figure.Figure":
    """
    Draw the Tanner graph of the code using NetworkX spring layout.
    Check nodes are blue squares; qubit nodes are orange circles.
    """
    plt = _require_matplotlib()
    try:
        import networkx as nx
        from .interop import to_networkx_tanner
    except ImportError:
        raise ImportError("networkx is required")

    G = to_networkx_tanner(code, which)
    H = code.hx if which == "X" else code.hz
    n2, n = H.shape

    # Sample if too large
    if n + n2 > max_nodes:
        sample_n  = max_nodes // 2
        sample_n2 = max_nodes - sample_n
        keep_qubits = [("qubit", j) for j in range(min(sample_n, n))]
        keep_checks = [("check", i) for i in range(min(sample_n2, n2))]
        G = G.subgraph(keep_qubits + keep_checks).copy()

    check_nodes = [v for v in G.nodes if v[0]=="check"]
    qubit_nodes = [v for v in G.nodes if v[0]=="qubit"]
    pos = nx.spring_layout(G, seed=42, k=2)

    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, nodelist=check_nodes,
                           node_shape='s', node_color='steelblue',
                           node_size=200, ax=ax, label="Check")
    nx.draw_networkx_nodes(G, pos, nodelist=qubit_nodes,
                           node_shape='o', node_color='coral',
                           node_size=150, ax=ax, label="Qubit")
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5, width=0.8)
    ax.set_title(f"Tanner graph — BB[[{code.n},{code.k},?]] {which}-type "
                 f"(showing ≤{max_nodes} nodes)", fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.axis('off')
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches='tight')
    if show:
        try: plt.show()
        except: pass
    return fig


# ---------------------------------------------------------------------------
# ASCII sparkline (terminal-friendly)
# ---------------------------------------------------------------------------

def sparkline(values: Sequence[float], width: int = 40) -> str:
    """Return a unicode block sparkline for quick terminal inspection."""
    BLOCKS = "▁▂▃▄▅▆▇█"
    if not values: return ""
    mn, mx = min(values), max(values)
    rng = mx - mn or 1
    chars = [BLOCKS[int((v - mn) / rng * (len(BLOCKS)-1))] for v in values]
    return "".join(chars)
