"""Basic network statistics (port of ``Metrics.scala``).

These functions follow the Scala semantics exactly, including how self loops
are treated:

* a self loop adds 2 to the degree of its vertex;
* in the local clustering coefficient, self loops are ignored both in the
  neighbour count and in the triangle count.
"""

from __future__ import annotations

import math

import numpy as np


def corrcoef(a, b):
    """Pearson correlation of two equal-length sequences (NaN if undefined)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = a.size
    if n == 0:
        return float("nan")
    da = a - a.mean()
    db = b - b.mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.dot(da, db) / (math.sqrt(np.dot(da, da)) * math.sqrt(np.dot(db, db))))


def degree_pairs(graph):
    """Arrays of (source degree, target degree) for both edge orientations.

    Self loops are listed once.  Returns ``(src, tgt, n_entries)``.
    """
    deg = {u: graph.degree(u) for u in graph.vertices}
    edges = list(graph.edges)
    s = [deg[e[0]] for e in edges]
    t = [deg[e[1]] for e in edges]
    for e in edges:
        if e[0] != e[1]:
            s.append(deg[e[1]])
            t.append(deg[e[0]])
    return s, t, len(s)


def assortativity(graph, pad=False):
    """Degree assortativity.

    ``pad=False`` is ``Metrics.assortativity``.  ``pad=True`` reproduces
    ``SlimGraphAlgorithms.assortativity`` (the one used in the fitness), which
    adds one extra observation ``(0.5, 0.5)`` -- or ``(0, 0)`` when the graph
    has self loops -- because it slices its padded arrays with an inclusive
    range.
    """
    s, t, n = degree_pairs(graph)
    if pad:
        val = 0.5 if n == 2 * graph.size() else 0.0
        s.append(val)
        t.append(val)
    cc = corrcoef(s, t)
    return 0.0 if math.isnan(cc) else cc


def clustering(graph, u, nbrs=None):
    """Local clustering coefficient of ``u`` (self loops ignored)."""
    if nbrs is None:
        su = set(graph.neighbor_vertices(u))
        get = lambda v: set(graph.neighbor_vertices(v))
    else:
        su = nbrs[u]
        get = nbrs.__getitem__
    S = su - {u} if u in su else su
    nn = len(S)
    if nn <= 1:
        return 0.0
    twice_links = 0
    for v in S:
        nv = get(v)
        twice_links += len(nv & S)
        if v in nv:            # a self loop on v is not a triangle edge
            twice_links -= 1
    return twice_links / (nn * (nn - 1))


def avg_clustering(graph):
    """Mean of the local clustering coefficients over all vertices."""
    nbrs = {u: set(io.incoming).union(io.outgoing) for u, io in graph.adj_list.items()}
    nvert = len(nbrs)
    if nvert == 0:
        return float("nan")
    return sum(clustering(graph, u, nbrs) for u in nbrs) / nvert
