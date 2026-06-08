"""
interop.py — Interoperability with external QEC tools.

Provides adapters for:
  - Stim  (circuit export + DEM generation)
  - NetworkX (Tanner graph / factor graph export)
  - PyMatching (direct graph construction)
  - Standard check-matrix formats (alist, sparse)

Author: Dr. Saleh H. AlDaajeh  ORCID: 0000-0001-7810-9290
"""
from __future__ import annotations
from typing import Any, Dict
import numpy as np

from .codes import BBCode


# ---------------------------------------------------------------------------
# Stim interop
# ---------------------------------------------------------------------------

def to_stim_detector_error_model(
    code: BBCode,
    p:    float,
    rounds: int = 1,
    before_round_data_depolarization: float = 0.0,
    before_measure_flip_probability:  float = 0.0,
) -> str:
    """
    Generate a Stim Detector Error Model (DEM) string for the given BB code.

    The DEM uses the chain-decomposed hyperedge representation required by
    PyMatching v2 (weight-3 columns split into two weight-2 edges).

    Parameters
    ----------
    code   : BBCode instance
    p      : physical error rate
    rounds : number of syndrome rounds (1 = memory experiment)
    before_round_data_depolarization : idle depolarization rate
    before_measure_flip_probability  : measurement flip rate

    Returns
    -------
    str : Stim DEM text (can be saved as .dem file or parsed by stim)
    """
    try:
        import stim
        return _to_stim_dem_via_stim(code, p, rounds,
                                      before_round_data_depolarization,
                                      before_measure_flip_probability)
    except ImportError:
        return _to_stim_dem_manual(code, p, rounds)


def _to_stim_dem_via_stim(code, p, rounds, p_idle, p_meas):
    """Use the stim package to build a memory circuit and extract DEM."""
    import stim
    n, n2 = code.n, code.n2
    HX, HZ = code.hx, code.hz

    circuit = stim.Circuit()
    # Data qubits: 0..n-1, ancilla qubits: n..n+2*n2-1
    for q in range(n):
        circuit.append("H", [q])
    circuit.append("TICK")

    # X-type parity checks
    for check_idx in range(n2):
        targets = [j for j in range(n) if HX[check_idx, j]]
        anc = n + check_idx
        circuit.append("H", [anc])
        for tgt in targets:
            circuit.append("CNOT", [anc, tgt])
        circuit.append("H", [anc])

    # Z-type parity checks
    for check_idx in range(n2):
        targets = [j for j in range(n) if HZ[check_idx, j]]
        anc = n + n2 + check_idx
        for tgt in targets:
            circuit.append("CNOT", [tgt, anc])

    if p > 0:
        circuit.append("DEPOLARIZE1", list(range(n)), p)
    circuit.append("TICK")
    circuit.append("M", list(range(n + 2*n2)))
    for i in range(n + 2*n2):
        circuit.append("DETECTOR", [stim.target_rec(-n-2*n2+i)])

    try:
        dem = circuit.detector_error_model(decompose_errors=True)
        return str(dem)
    except Exception:
        return _to_stim_dem_manual(code, p, rounds)


def _to_stim_dem_manual(code, p, rounds):
    """
    Manually construct a minimal DEM text representation without stim.
    Uses chain decomposition for weight-3 columns.
    """
    n, n2 = code.n, code.n2
    HX = code.hx
    lines = [f"# DEM for BB[[{n},{code.k},?]] p={p:.4g} rounds={rounds}"]

    for j in range(n):
        checks = [i for i in range(n2) if HX[i, j]]
        q = p
        if len(checks) == 2:
            lines.append(f"error({q:.6g}) D{checks[0]} D{checks[1]}")
        elif len(checks) == 3:
            # Chain decomposition: {a,b,c} → {a,b} + {b,c}
            a, b, c = checks
            lines.append(f"error({q:.6g}) D{a} D{b}")
            lines.append(f"error({q:.6g}) D{b} D{c}")
        elif len(checks) == 1:
            lines.append(f"error({q:.6g}) D{checks[0]}")

    return "\n".join(lines)


def to_stim_circuit(code: BBCode, p: float) -> str:
    """
    Generate a basic Stim circuit for BB code memory experiment.
    Returns a Stim circuit string.
    """
    try:
        import stim
    except ImportError:
        raise ImportError("stim is required: pip install stim")

    n, n2 = code.n, code.n2
    HX, HZ = code.hx, code.hz
    circuit_lines = [
        f"# BB[[{n},{code.k},?]] memory circuit, p={p}",
    ]
    # X-type stabilisers
    for i in range(n2):
        targets = [j for j in range(n) if HX[i, j]]
        anc = n + i
        circuit_lines.append(f"H {anc}")
        for t in targets:
            circuit_lines.append(f"CNOT {anc} {t}")
        circuit_lines.append(f"H {anc}")
    # Z-type stabilisers
    for i in range(n2):
        targets = [j for j in range(n) if HZ[i, j]]
        anc = n + n2 + i
        for t in targets:
            circuit_lines.append(f"CNOT {t} {anc}")
    if p > 0:
        circuit_lines.append(f"DEPOLARIZE1({p}) " + " ".join(str(q) for q in range(n)))
    circuit_lines.append("M " + " ".join(str(q) for q in range(n+2*n2)))
    return "\n".join(circuit_lines)


# ---------------------------------------------------------------------------
# NetworkX interop
# ---------------------------------------------------------------------------

def to_networkx_tanner(code: BBCode, which: str = "X") -> "networkx.Graph":  # noqa: F821
    """
    Build the Tanner graph of the BB code as a NetworkX bipartite graph.

    Parameters
    ----------
    code  : BBCode
    which : "X" for HX | "Z" for HZ

    Returns
    -------
    networkx.Graph with nodes labelled as ("check", i) and ("qubit", j),
    plus a node attribute 'bipartite' (0=check, 1=qubit).
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx is required: pip install networkx")

    H = code.hx if which == "X" else code.hz
    n2, n = H.shape
    G = nx.Graph()
    G.graph["code"] = f"BB[[{code.n},{code.k},?]]"
    G.graph["which"] = which

    for i in range(n2):
        G.add_node(("check", i), bipartite=0, label=f"c{i}")
    for j in range(n):
        G.add_node(("qubit", j), bipartite=1, label=f"q{j}")
    for i in range(n2):
        for j in range(n):
            if H[i, j]:
                G.add_edge(("check", i), ("qubit", j))
    return G


def to_networkx_factor(code: BBCode, which: str = "X") -> "networkx.Graph":  # noqa: F821
    """
    Build the factor graph used by PyMatching: only qubit nodes,
    with edges between pairs of checks that share a qubit.
    Weight-3 columns are chain-decomposed.
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx is required: pip install networkx")

    H  = code.hx if which == "X" else code.hz
    n2, n = H.shape
    G = nx.Graph()
    G.graph["code"]  = f"BB[[{code.n},{code.k},?]]"
    G.graph["which"] = which
    G.graph["type"]  = "factor_graph_chain_decomposed"

    for i in range(n2):
        G.add_node(i, label=f"check_{i}")
    # Add a boundary node for degree-1 columns
    G.add_node("boundary", label="boundary")

    for j in range(n):
        checks = [i for i in range(n2) if H[i, j]]
        if len(checks) == 2:
            a, b = checks
            G.add_edge(a, b, qubit=j, weight=1.0)
        elif len(checks) == 3:
            a, b, c = checks
            G.add_edge(a, b, qubit=j, weight=1.0, chain_part="ab")
            G.add_edge(b, c, qubit=j, weight=1.0, chain_part="bc")
        elif len(checks) == 1:
            G.add_edge(checks[0], "boundary", qubit=j, weight=1.0)
    return G


def networkx_to_pymatching(G: "networkx.Graph", p: float = 0.01) -> Any:  # noqa: F821
    """
    Convert a NetworkX factor graph to a PyMatching matching object.

    Parameters
    ----------
    G : networkx factor graph from to_networkx_factor()
    p : edge error probability

    Returns
    -------
    pymatching.Matching instance
    """
    try:
        import pymatching
    except ImportError:
        raise ImportError("pymatching is required: pip install pymatching")

    m = pymatching.Matching()
    try:
        for u, v, data in G.edges(data=True):
            if u == "boundary" or v == "boundary":
                node = v if u == "boundary" else u
                m.add_boundary_edge(node, error_probability=p,
                                    qubit_id=data.get('qubit', None))
            else:
                m.add_edge(u, v, error_probability=p,
                           qubit_id=data.get('qubit', None))
        return m
    except Exception as e:
        raise RuntimeError(f"PyMatching conversion failed: {e}")


# ---------------------------------------------------------------------------
# Check-matrix format interop
# ---------------------------------------------------------------------------

def to_alist(H: np.ndarray) -> str:
    """
    Export check matrix H as MacKay alist format.
    Standard format for LDPC code exchange.
    """
    m, n = H.shape
    row_weights = [sum(H[i, :]) for i in range(m)]
    col_weights = [sum(H[:, j]) for j in range(n)]
    max_rw = max(row_weights) if row_weights else 0
    max_cw = max(col_weights) if col_weights else 0
    lines = [
        f"{n} {m}",
        f"{max_cw} {max_rw}",
        " ".join(str(w) for w in col_weights),
        " ".join(str(w) for w in row_weights),
    ]
    for j in range(n):
        nbrs = [i+1 for i in range(m) if H[i, j]]
        nbrs += [0] * (max_cw - len(nbrs))
        lines.append(" ".join(str(x) for x in nbrs))
    for i in range(m):
        nbrs = [j+1 for j in range(n) if H[i, j]]
        nbrs += [0] * (max_rw - len(nbrs))
        lines.append(" ".join(str(x) for x in nbrs))
    return "\n".join(lines)


def from_alist(text: str) -> np.ndarray:
    """Parse alist format and return binary check matrix."""
    lines = [ln.strip() for ln in text.strip().split('\n') if ln.strip()]
    n, m   = map(int, lines[0].split())
    max_cw, max_rw = map(int, lines[1].split())
    # skip weight lines
    col_blocks = lines[4:4+n]
    H = np.zeros((m, n), dtype=np.uint8)
    for j, row in enumerate(col_blocks):
        for nbr_str in row.split():
            nbr = int(nbr_str)
            if nbr > 0:
                H[nbr-1, j] = 1
    return H


def to_sparse_dict(H: np.ndarray) -> Dict:
    """Export check matrix as sparse dict {row: [col, ...]}."""
    m, n = H.shape
    return {
        "shape": [m, n],
        "rows":  {str(i): [j for j in range(n) if H[i,j]]
                  for i in range(m) if any(H[i,:])},
    }


def from_sparse_dict(d: Dict) -> np.ndarray:
    """Reconstruct dense matrix from sparse dict."""
    m, n = d["shape"]
    H = np.zeros((m, n), dtype=np.uint8)
    for i_str, cols in d["rows"].items():
        for j in cols:
            H[int(i_str), j] = 1
    return H


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def interop_summary(code: BBCode) -> Dict:
    """Return a dict summarising available interop paths for a code."""
    ok_tests = {}
    try:
        to_networkx_tanner(code, "X")
        ok_tests["networkx_tanner"] = True
    except Exception as e:
        ok_tests["networkx_tanner"] = str(e)
    try:
        to_networkx_factor(code, "X")
        ok_tests["networkx_factor"] = True
    except Exception as e:
        ok_tests["networkx_factor"] = str(e)
    try:
        import stim
        ok_tests["stim_available"] = True
        to_stim_detector_error_model(code, 0.01)
        ok_tests["stim_dem"] = True
    except ImportError:
        ok_tests["stim_available"] = False
    except Exception as e:
        ok_tests["stim_dem"] = str(e)
    try:
        import pymatching
        ok_tests["pymatching_available"] = True
    except ImportError:
        ok_tests["pymatching_available"] = False

    al = to_alist(code.hx)
    H_back = from_alist(al)
    ok_tests["alist_roundtrip"] = bool(np.array_equal(H_back, code.hx))

    return ok_tests
