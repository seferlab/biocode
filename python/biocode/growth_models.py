"""Classic parametric growth models (port of ``GrowthModels.scala``).

``forest_fire`` and ``dmc`` (duplication-mutation-complementation) are kept as
baselines for comparison with learned GrowCode programs.
"""

from __future__ import annotations

import random as _random
from collections import deque

from .graph_algorithms import GeometricRV
from .slimgraph import SlimGraphUnDirected, UndirectedEdge, sample


def forest_fire(n, p, f=None, samp=25, rng=None):
    """Forest-fire model with ``n`` vertices and burning parameter ``p``."""
    rng = rng or _random.Random()
    graph = SlimGraphUnDirected()
    graph.insert_edge(UndirectedEdge(0, 1))
    geom = GeometricRV(p, rng)

    def do_step(new_vert):
        u = int(rng.random() * graph.order())
        seen = {u}
        to_visit = deque([u])
        while to_visit:
            other = to_visit.popleft()
            x = min(graph.degree(other), geom.next())
            for w in sample(list(graph.neighbor_vertices(other)), x, rng):
                if w not in seen:
                    seen.add(w)
                    to_visit.append(w)
            # connect the burned vertex to the new vertex
            graph.insert_edge(UndirectedEdge(other, new_vert))

    for v in range(2, n):
        do_step(v)
        if f is not None and v % samp == 0:
            f(graph)
    return graph


def dmc(n, qmod, qcon, rng=None):
    """Duplication-mutation-complementation model (Vazquez et al.)."""
    rng = rng or _random.Random()
    graph = SlimGraphUnDirected()
    graph.insert_edge(UndirectedEdge(0, 1))
    for u in range(2, n):
        v = int(rng.random() * graph.order())
        graph.insert_vertex(u)
        # duplicate v
        for x in list(graph.neighbor_vertices(v)):
            graph.insert_edge(UndirectedEdge(u, x))
        # remove one edge of each duplicated pair with probability qmod
        for x in list(graph.neighbor_vertices(u)):
            if rng.random() < qmod:
                if rng.random() < 0.5:
                    graph.remove_edge(UndirectedEdge(v, x))
                else:
                    graph.remove_edge(u, x)
        # complementation: link the copy to the original with probability qcon
        if rng.random() < qcon:
            graph.insert_edge(UndirectedEdge(u, v))
        if graph.size() > 3 * n:
            return graph
    return graph


# Scala-style aliases
ForestFire = forest_fire
DMC = dmc
