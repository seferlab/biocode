"""Graph statistics used to score grown networks.

Port of ``SlimGraphAlgorithms.scala`` together with its helpers
``LaplacianCalculator.scala`` and ``PowerIteration.scala``.  The Scala project
also has ``GraphAlgorithms.scala``, a duplicate of the same functions written
for the scala-graph library; since the Python port has a single graph type,
both are served by this module.

The three statistics that drive the fitness function are :func:`shape`,
:func:`assortativity` and :func:`avg_clustering`.
"""

from __future__ import annotations

import math
import random as _random
from collections import deque
from itertools import combinations

import numpy as np

from . import metrics
from .histogram import Histogram
from .slimgraph import sample

INT_MAX = 2 ** 31 - 1   # Scala's Int.MaxValue marks "unreachable"


def _fmt(x):
    """Format a double the way Scala/Java prints it (close enough for logs)."""
    if isinstance(x, float):
        if math.isnan(x):
            return "NaN"
        if math.isinf(x):
            return "Infinity" if x > 0 else "-Infinity"
    return repr(float(x))


def print_properties(graph, out=print):
    out("# Nodes: %d # Edges: %d" % (graph.order(), graph.size()))
    out("Shape (deg dist)\t" + ",".join(_fmt(x) for x in shape(graph)))
    out("Assortativity\t" + _fmt(assortativity(graph)))
    out("AVG Clustering\t" + _fmt(avg_clustering(graph)))


# ---------------------------------------------------------------------------
# Degree distribution "shape", assortativity, clustering
# ---------------------------------------------------------------------------

def degrees(graph):
    return [graph.degree(v) for v in graph.vertices]


def shape(graph):
    """Cumulative, normalised 20-bin histogram of ``degree / (max_degree + 1)``.

    The last entry is always 20.  Two graphs are compared with the L1 distance
    between their shapes.
    """
    ds = degrees(graph) or [0]
    sumds = max(ds) + 1
    h = Histogram([d / sumds for d in ds], drange=(0.0, 1.0), nbins=20, normed=True)
    return np.cumsum(h.hist)


def assortativity(graph):
    """Degree assortativity exactly as used by the fitness function."""
    return metrics.assortativity(graph, pad=True)


def avg_clustering(graph):
    return metrics.avg_clustering(graph)


def clustering(graph):
    """Global clustering: closed / (1 + connected) triples."""
    closed, connected = 0, 1
    for x in graph.vertices:
        seq = list(graph.neighbor_vertices(x))
        uniq = list(dict.fromkeys(seq))
        pairs = list(combinations(uniq, 2))
        if len(uniq) != len(seq):          # Scala's combinations on a multiset
            counts = {}
            for v in seq:
                counts[v] = counts.get(v, 0) + 1
            pairs += [(v, v) for v, c in counts.items() if c >= 2]
        for v, w in pairs:
            connected += 1
            if graph.edge_option(v, w) is not None:
                closed += 1
    return closed / connected


# ---------------------------------------------------------------------------
# Shortest paths
# ---------------------------------------------------------------------------

class ShortestPathOperations:
    """Unweighted single-source shortest paths (breadth first search)."""

    def __init__(self, graph):
        self.graph = graph
        self.dists = {}

    def single_source_shortest_paths(self, src):
        g = self.graph
        dists = {v: INT_MAX for v in g.vertices}
        dists[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            du = dists[u] + 1
            for w in g.neighbor_vertices(u):
                if dists[w] == INT_MAX:
                    dists[w] = du
                    q.append(w)
        self.dists = dists
        return dists


def diameter(graph):
    sp = ShortestPathOperations(graph)
    best = 0
    for v in list(graph.vertices):
        d = sp.single_source_shortest_paths(v)
        best = max(best, max(x for x in d.values() if x < INT_MAX))
    return best


def _all_path_lengths(graph):
    sp = ShortestPathOperations(graph)
    out = []
    for v in list(graph.vertices):
        out.extend(float(x) for x in sp.single_source_shortest_paths(v).values() if x < INT_MAX)
    return out


def characteristic_path_length(graph):
    """Mean finite distance over all ordered pairs (self pairs included, as in Scala)."""
    pl = _all_path_lengths(graph)
    return float(np.float32(sum(pl))) / len(pl)


def path_length_dist(graph):
    return Histogram(_all_path_lengths(graph), drange=(0, 40), nbins=40, normed=True).hist


def effective_diameter(graph, cutoff, rng=_random):
    """Interpolated distance below which ``cutoff`` of the sampled pairs fall."""
    if graph.size() == 0:
        return 0.0
    num_samp = min(graph.order(), 200)
    counts = {}
    for v in sample(graph.vertices, num_samp, rng):
        d = ShortestPathOperations(graph).single_source_shortest_paths(v)
        for x in d.values():
            if 0 < x < INT_MAX:
                counts[x] = counts.get(x, 0.0) + 1.0
    cum, c = 0.0, {}
    for k in sorted(counts):
        cum += counts[k]
        c[k] = cum
    for k in c:
        c[k] /= cum
    eff = next(k for k in sorted(c) if c[k] >= cutoff)
    if eff - 1 == 0:
        return 0.0
    x1 = eff - 1
    y1 = c[x1]
    dy = c[eff] - c[x1]
    return (dy * x1 + cutoff - y1) / dy


# ---------------------------------------------------------------------------
# Spectral statistics
# ---------------------------------------------------------------------------

def _adjacency_index(graph):
    nodes = list(graph.vertices)
    idx = {u: i for i, u in enumerate(nodes)}
    pairs = set()
    for u in nodes:
        ui = idx[u]
        for v in graph.neighbor_vertices(u):
            pairs.add((ui, idx[v]))
    if pairs:
        rows, cols = map(np.array, zip(*sorted(pairs)))
    else:
        rows = cols = np.zeros(0, dtype=int)
    return nodes, idx, rows, cols


def leading_eig(rows, cols, n, iters=50, thresh=1e-3):
    """Power iteration on the 0/1 matrix with ones at ``(rows, cols)``
    (port of ``PowerIteration.leadingEig``)."""
    bk = np.ones(n)
    i, err = 0, float("inf")
    while i < iters and err > thresh:
        bk1 = np.bincount(rows, weights=bk[cols], minlength=n)
        with np.errstate(divide="ignore", invalid="ignore"):
            bk1 = bk1 / np.linalg.norm(bk1)
        err = np.linalg.norm(bk1 - bk)
        bk = bk1
        i += 1
    return bk


def eigenvector_centrality(graph):
    """Leading eigenvector of the adjacency matrix, in ``graph.vertices`` order."""
    nodes, _, rows, cols = _adjacency_index(graph)
    return leading_eig(rows, cols, len(nodes))


def laplacian_matrix(graph):
    """Normalised Laplacian ``I - D^-1/2 W D^-1/2`` and the degree vector
    (port of ``LaplacianCalculator.compute``; dense, so for modest graphs)."""
    nodes = list(graph.vertices)
    idx = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)
    d = np.zeros(n)
    for u in nodes:
        d[idx[u]] = sum(1 for _ in graph.neighbor_vertices(u))
    L = np.zeros((n, n))
    for u in nodes:
        i = idx[u]
        for v in graph.neighbor_vertices(u):
            j = idx[v]
            L[i, j] = -1.0 / math.sqrt(d[i] * d[j])
        L[i, i] = 1.0
    return L, d


def laplacian(graph):
    """Sorted eigenvalues of the normalised Laplacian."""
    n = graph.order()
    L, _ = laplacian_matrix(graph)
    try:
        return np.sort(np.linalg.eigvalsh(L))
    except np.linalg.LinAlgError:
        return np.zeros(max(n - 1, 0))


def lch(lap):
    return Histogram(lap, drange=(0.0, 2.0), nbins=20, normed=True).hist


def ech(eig_centrality):
    return Histogram(eig_centrality, drange=(0.0, 1.0), nbins=20, normed=True).hist


# ---------------------------------------------------------------------------
# Forest-fire sub-sampling
# ---------------------------------------------------------------------------

class GeometricRV:
    """``floor(log(U) / log(1 - p))`` -- number of extra neighbours to burn."""

    def __init__(self, p, rng=None):
        self.p = p
        self.rng = rng or _random.Random()

    def next(self):
        u = 1.0 - self.rng.random()          # in (0, 1]
        return int(math.log(u) / math.log(1 - self.p))


def sub_sample(source, make_edge, frac, rng=None):
    """Forest-fire sample of about ``frac * |V|`` vertices of ``source``."""
    assert frac < 1.0
    rng = rng or _random.Random()
    target = source.empty()
    geom = GeometricRV(0.7, rng)
    nodes = list(source.vertices)

    def burn(src):
        seen = {src}
        to_visit = deque([src])
        while to_visit:
            other = to_visit.popleft()
            x = min(source.degree(other), geom.next())
            burned = []
            for w in sample(list(source.neighbor_vertices(other)), x, rng):
                if w not in seen:
                    seen.add(w)
                    burned.insert(0, w)
            for w in burned:
                target.insert_edge(make_edge(other, w))
                to_visit.append(w)

    target_order = int(frac * source.order())
    used = set()
    while target.order() < target_order and len(used) < len(nodes):
        seed = nodes[rng.randrange(len(nodes))]
        if seed not in used:
            burn(seed)
            used.add(seed)
    return target
