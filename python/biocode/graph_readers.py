"""Graph file readers (port of ``FileReaders.scala`` and the
``EdgeListReader`` object in ``GCInitializer.scala``).

All readers ignore lines whose first non-blank character is ``#``.  Unlike the
Scala versions they also skip blank lines and tolerate leading whitespace.
"""

from __future__ import annotations

from .slimgraph import (DirectedEdge, SlimGraphDirected, SlimGraphUnDirected,
                        UndirectedEdge, WeightedDirectedEdge, WeightedUndirectedEdge)


def _data_lines(fname):
    with open(fname) as fh:
        for line in fh:
            s = line.strip()
            if s and not s.startswith("#"):
                yield s.split()


def _new_graph(directed):
    return SlimGraphDirected() if directed else SlimGraphUnDirected()


class EdgeListReader:
    """Whitespace separated edge lists: ``u v`` or ``u v weight``."""

    @staticmethod
    def from_file_sg(fname):
        """Undirected integer graph -- what the GrowCode tools use for targets."""
        g = SlimGraphUnDirected()
        for toks in _data_lines(fname):
            g.insert_edge(UndirectedEdge(int(toks[0]), int(toks[1])))
        return g

    @staticmethod
    def from_file(fname, directed=False, weighted=False, node_type=int,
                  weight_type=float, default_weight=None, edge_factory=None):
        """General reader.

        * ``weighted=True`` reads the weight from the third column.
        * ``default_weight`` attaches a constant weight to every edge.
        * ``edge_factory(u, v[, w])`` overrides the edge type.
        """
        g = _new_graph(directed)
        if edge_factory is None:
            if weighted or default_weight is not None:
                edge_factory = WeightedDirectedEdge if directed else WeightedUndirectedEdge
            else:
                edge_factory = DirectedEdge if directed else UndirectedEdge
        for toks in _data_lines(fname):
            u, v = node_type(toks[0]), node_type(toks[1])
            if weighted:
                g.insert_edge(edge_factory(u, v, weight_type(toks[2])))
            elif default_weight is not None:
                g.insert_edge(edge_factory(u, v, default_weight))
            else:
                g.insert_edge(edge_factory(u, v))
        return g


class AdjListReader:
    """Each line: ``u v1 v2 v3 ...``."""

    @staticmethod
    def from_file(fname, directed=False, node_type=int, edge_factory=None):
        g = _new_graph(directed)
        edge_factory = edge_factory or (DirectedEdge if directed else UndirectedEdge)
        for toks in _data_lines(fname):
            u = node_type(toks[0])
            g.insert_vertex(u)
            for vs in toks[1:]:
                g.insert_edge(edge_factory(u, node_type(vs)))
        return g


class MultiAdjListReader:
    """Blocks of the form::

        u  k
        v1 w1
        ...
        vk wk
    """

    @staticmethod
    def from_file(fname, directed=False, weighted=False, node_type=int,
                  weight_type=float, edge_factory=None):
        g = _new_graph(directed)
        if edge_factory is None:
            if weighted:
                edge_factory = WeightedDirectedEdge if directed else WeightedUndirectedEdge
            else:
                edge_factory = DirectedEdge if directed else UndirectedEdge
        with open(fname) as fh:
            lines = iter(fh)
            for line in lines:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                toks = s.split()
                u, k = node_type(toks[0]), int(toks[1])
                g.insert_vertex(u)
                for _ in range(k):
                    t = next(lines).split()
                    v = node_type(t[0])
                    if weighted:
                        g.insert_edge(edge_factory(u, v, weight_type(t[1])))
                    else:
                        g.insert_edge(edge_factory(u, v))
        return g


def write_edge_list(graph, fname, sep="\t"):
    """Write ``source<sep>target`` lines (format used by GrowEnsemble)."""
    with open(fname, "w") as out:
        for e in graph.edges:
            out.write("%s%s%s\n" % (e.source, sep, e.target))
