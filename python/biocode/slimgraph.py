"""Light-weight graph containers (port of ``SlimGraph.scala``).

The Scala code stores, for every vertex, two maps: ``incoming`` (neighbour ->
edge) and ``outgoing`` (neighbour -> edge).  Undirected edges are normalised so
that ``source <= target``; the edge then lives in ``outgoing[source]`` and in
``incoming[target]``.  A self loop ``(u, u)`` appears in both maps of ``u`` and
therefore contributes 2 to ``degree(u)``.  This module keeps exactly that
layout because several metrics (degree, assortativity, clustering) depend on
it.

The only structural addition is :class:`IndexedSet`, which backs the edge set
so that a uniformly random edge can be drawn in O(1) (``Machine.opRedge``).
"""

from __future__ import annotations

import heapq
import random as _random
from collections import deque
from itertools import chain
from operator import itemgetter


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------

class SimpleEdge(tuple):
    """Base class: an edge is an immutable ``(source, target)`` pair.

    Equality and hashing only look at ``(source, target)``, matching the
    ``UnorderedEdgeLike`` / ``OrderedEdgeLike`` traits in the Scala code.
    """

    __slots__ = ()
    source = property(itemgetter(0))
    target = property(itemgetter(1))

    def __repr__(self):
        return "%s(%r, %r)" % (type(self).__name__, self[0], self[1])


class UndirectedEdge(SimpleEdge):
    """Undirected edge; the smaller endpoint is always stored as ``source``."""

    __slots__ = ()

    def __new__(cls, u, v):
        return tuple.__new__(cls, (v, u) if u > v else (u, v))

    @staticmethod
    def factory():
        return UndirectedEdge


class DirectedEdge(SimpleEdge):
    __slots__ = ()

    def __new__(cls, source, target):
        return tuple.__new__(cls, (source, target))

    @staticmethod
    def factory():
        return DirectedEdge


class WeightedUndirectedEdge(UndirectedEdge):
    """Undirected edge carrying a mutable ``weight`` (not part of equality)."""

    def __new__(cls, u, v, weight=None):
        e = UndirectedEdge.__new__(cls, u, v)
        e.weight = weight
        return e

    def __getnewargs__(self):
        return (self[0], self[1], self.weight)

    @staticmethod
    def factory():
        return WeightedUndirectedEdge

    @staticmethod
    def factory_with_default(w):
        return lambda u, v: WeightedUndirectedEdge(u, v, w)


class WeightedDirectedEdge(DirectedEdge):
    def __new__(cls, source, target, weight=None):
        e = DirectedEdge.__new__(cls, source, target)
        e.weight = weight
        return e

    def __getnewargs__(self):
        return (self[0], self[1], self.weight)

    @staticmethod
    def factory():
        return WeightedDirectedEdge

    @staticmethod
    def factory_with_default(w):
        return lambda u, v: WeightedDirectedEdge(u, v, w)


class MMSEdge(UndirectedEdge):
    """Edge type read by :class:`MMSGraphReader` (``u v weight freq q``)."""

    def __new__(cls, u, v, weight, freq, q):
        e = UndirectedEdge.__new__(cls, u, v)
        e.weight, e.freq, e.q = weight, freq, q
        return e

    def __getnewargs__(self):
        return (self[0], self[1], self.weight, self.freq, self.q)

    @staticmethod
    def from_line(s):
        toks = s.split()
        return MMSEdge(toks[0], toks[1], float(toks[2]), float(toks[3]), float(toks[4]))


# ---------------------------------------------------------------------------
# Indexed set (O(1) add / remove / contains / random access)
# ---------------------------------------------------------------------------

class IndexedSet:
    __slots__ = ("_items", "_index")

    def __init__(self, items=()):
        self._items = []
        self._index = {}
        for x in items:
            self.add(x)

    def add(self, x):
        if x in self._index:
            return False
        self._index[x] = len(self._items)
        self._items.append(x)
        return True

    def discard(self, x):
        i = self._index.pop(x, None)
        if i is None:
            return False
        last = self._items.pop()
        if i < len(self._items):
            self._items[i] = last
            self._index[last] = i
        return True

    remove = discard

    def __contains__(self, x):
        return x in self._index

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, i):
        return self._items[i]

    def head(self):
        return self._items[0]

    def copy(self):
        s = IndexedSet()
        s._items = list(self._items)
        s._index = dict(self._index)
        return s

    def __repr__(self):
        return "IndexedSet(%r)" % (self._items,)


class InOutEdges:
    __slots__ = ("incoming", "outgoing")

    def __init__(self, incoming=None, outgoing=None):
        self.incoming = {} if incoming is None else incoming
        self.outgoing = {} if outgoing is None else outgoing


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------

class SimpleSlimGraphLike:
    """Shared behaviour of the directed and undirected slim graphs."""

    def __init__(self):
        self.adj_list = {}          # N -> InOutEdges
        self.edge_set = IndexedSet()

    # -- size / order ------------------------------------------------------
    def degree(self, v):
        io = self.adj_list[v]
        return len(io.incoming) + len(io.outgoing)

    def size(self):
        return len(self.edge_set)

    def order(self):
        return len(self.adj_list)

    @property
    def vertices(self):
        return self.adj_list.keys()

    @property
    def edges(self):
        return self.edge_set

    def edge_at(self, i):
        """The i-th edge of the edge set (used to draw a random edge)."""
        return self.edge_set[i]

    def has_vertex(self, u):
        return u in self.adj_list

    def empty(self):
        return type(self)()

    # -- mutation ------------------------------------------------------------
    def insert_edge(self, e):
        if e in self.edge_set:
            return False
        self.edge_set.add(e)
        s, t = e[0], e[1]
        io = self.adj_list.get(s)
        if io is None:
            io = self.adj_list[s] = InOutEdges()
        io.outgoing[t] = e
        io = self.adj_list.get(t)
        if io is None:
            io = self.adj_list[t] = InOutEdges()
        io.incoming[s] = e
        return True

    def remove_edge(self, e, v=None):
        """``remove_edge(e)`` or ``remove_edge(u, v)`` (as in the Scala overloads)."""
        if v is not None:
            eo = self.edge_option(e, v)
            return self.remove_edge(eo) if eo is not None else False
        if e not in self.edge_set:
            return False
        self.edge_set.discard(e)
        del self.adj_list[e[0]].outgoing[e[1]]
        del self.adj_list[e[1]].incoming[e[0]]
        return True

    def insert_vertex(self, u):
        if u in self.adj_list:
            return False
        self.adj_list[u] = InOutEdges()
        return True

    def remove_vertex(self, u):
        io = self.adj_list.get(u)
        if io is None:
            return False
        for v, e in list(io.outgoing.items()):
            self.adj_list[v].incoming.pop(u, None)
            self.edge_set.discard(e)
        for v, e in list(io.incoming.items()):
            self.adj_list[v].outgoing.pop(u, None)
            self.edge_set.discard(e)
        del self.adj_list[u]
        return True

    # -- neighbourhoods ----------------------------------------------------
    def neighbor_edges(self, u):
        io = self.adj_list[u]
        return chain(io.incoming.values(), io.outgoing.values())

    def neighbor_vertices(self, u):
        io = self.adj_list[u]
        return chain(io.incoming.keys(), io.outgoing.keys())

    def neighbors(self, u):
        """Iterator over ``(neighbour, edge)`` pairs."""
        io = self.adj_list[u]
        return chain(io.incoming.items(), io.outgoing.items())

    # subclasses implement: is_directed, has_edge(u, v), edge_option(u, v), copy

    def __repr__(self):
        return "%s(|V|=%d, |E|=%d)" % (type(self).__name__, self.order(), self.size())


class SlimGraphUnDirected(SimpleSlimGraphLike):

    def is_directed(self):
        return False

    def copy(self):
        g = type(self)()
        for u in self.adj_list:
            g.insert_vertex(u)
        for e in self.edge_set:
            g.insert_edge(e)
        return g

    def has_edge(self, u, v):
        io = self.adj_list.get(u)
        if io is None:
            return False
        return (v in io.outgoing) if u < v else (v in io.incoming)

    def edge_option(self, u, v):
        """The edge between u and v, or ``None``."""
        io = self.adj_list.get(u)
        if io is None:
            return None
        return io.outgoing.get(v) if u < v else io.incoming.get(v)

    def neighbor_sets(self):
        """``{u: set(neighbours)}`` -- a convenience snapshot for metrics."""
        return {u: set(io.incoming).union(io.outgoing) for u, io in self.adj_list.items()}


class SlimGraphDirected(SimpleSlimGraphLike):

    def is_directed(self):
        return True

    def has_edge(self, u, v):
        io = self.adj_list.get(u)
        return io is not None and v in io.outgoing

    def edge_option(self, u, v):
        io = self.adj_list.get(u)
        return None if io is None else io.outgoing.get(v)

    def copy(self):
        g = SlimGraphDirected()
        for u in self.vertices:
            g.insert_vertex(u)
        for e in self.edges:
            g.insert_edge(e)
        return g

    def incoming_edges(self, u):
        return iter(self.adj_list[u].incoming.values())

    def outgoing_edges(self, u):
        return iter(self.adj_list[u].outgoing.values())

    def predecessors(self, u):
        return iter(self.adj_list[u].incoming.keys())

    def successors(self, u):
        return iter(self.adj_list[u].outgoing.keys())

    def topological_ordering(self):
        """Vertices in topological order (DFS, as in the Scala code).

        Raises ``AssertionError`` if the graph has a cycle that leaves some
        vertex unreachable from a sink.
        """
        order, visited = [], set()
        sinks = [n for n in self.vertices if not self.adj_list[n].outgoing]
        for s in sinks:
            stack = [(s, iter(list(self.predecessors(s))))]
            if s in visited:
                continue
            visited.add(s)
            while stack:
                node, it = stack[-1]
                nxt = next(it, None)
                if nxt is None:
                    stack.pop()
                    order.append(node)
                elif nxt not in visited:
                    visited.add(nxt)
                    stack.append((nxt, iter(list(self.predecessors(nxt)))))
        assert len(order) == self.order(), (
            "Size of topo order list is %d, but there are %d nodes" % (len(order), self.order()))
        return order


# ---------------------------------------------------------------------------
# Observable graph + neighbour cache
# ---------------------------------------------------------------------------

class EdgeAddedEvent:
    __slots__ = ("e",)

    def __init__(self, e):
        self.e = e


class EdgeRemovedEvent(EdgeAddedEvent):
    __slots__ = ()


class VertexAddedEvent:
    __slots__ = ("v",)

    def __init__(self, v):
        self.v = v


class VertexRemovedEvent(VertexAddedEvent):
    __slots__ = ()


class ObservableSlimGraphUndirected(SlimGraphUnDirected):
    """Undirected slim graph that notifies listeners on every change."""

    def __init__(self):
        super().__init__()
        self.edge_callbacks = {}
        self.vertex_callbacks = {}

    def observe_edge(self, callback):
        self.edge_callbacks[callback] = callback
        return callback

    def observe_vertex(self, callback):
        self.vertex_callbacks[callback] = callback
        return callback

    def unobserve_edge(self, handle):
        self.edge_callbacks.pop(handle, None)

    def unobserve_vertex(self, handle):
        self.vertex_callbacks.pop(handle, None)

    def _notify_edge(self, ev):
        for cb in list(self.edge_callbacks.values()):
            cb(ev)

    def _notify_vertex(self, ev):
        for cb in list(self.vertex_callbacks.values()):
            cb(ev)

    def insert_edge(self, e):
        r = super().insert_edge(e)
        if r:
            self._notify_edge(EdgeAddedEvent(e))
        return r

    def remove_edge(self, e, v=None):
        if v is not None:
            eo = self.edge_option(e, v)
            return self.remove_edge(eo) if eo is not None else False
        r = super().remove_edge(e)
        if r:
            self._notify_edge(EdgeRemovedEvent(e))
        return r

    def insert_vertex(self, n):
        r = super().insert_vertex(n)
        if r:
            self._notify_vertex(VertexAddedEvent(n))
        return r

    def remove_vertex(self, n):
        r = super().remove_vertex(n)
        if r:
            self._notify_vertex(VertexRemovedEvent(n))
        return r


class NeighborCache:
    """Caches neighbour sets of an observable graph and keeps them in sync."""

    def __init__(self, graph):
        self.graph = graph
        self.neighbors = {}
        self._vh = graph.observe_vertex(self.modified_vertex)
        self._eh = graph.observe_edge(self.modified_edge)

    def stop_observing(self):
        self.graph.unobserve_edge(self._eh)
        self.graph.unobserve_vertex(self._vh)

    def modified_vertex(self, ve):
        if isinstance(ve, VertexRemovedEvent) and ve.v in self.neighbors:
            self.neighbors[ve.v] = set()

    def modified_edge(self, ee):
        e = ee.e
        if isinstance(ee, EdgeRemovedEvent):
            if e.source in self.neighbors:
                self.neighbors[e.source].discard(e.target)
            if e.target in self.neighbors:
                self.neighbors[e.target].discard(e.source)
        else:
            if e.source in self.neighbors:
                self.neighbors[e.source].add(e.target)
            if e.target in self.neighbors:
                self.neighbors[e.target].add(e.source)

    def neighbor_vertex_set(self, v):
        if v not in self.neighbors:
            self.neighbors[v] = set(self.graph.neighbor_vertices(v))
        return self.neighbors[v]


# ---------------------------------------------------------------------------
# Union-find and edge-based algorithms
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self):
        self._parent = {}
        self._rank = {}

    def add_node(self, x):
        self._parent[x] = x
        self._rank[x] = 0

    def find(self, x):
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:           # path compression
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return
        if self._rank[xr] < self._rank[yr]:
            self._parent[xr] = yr
        elif self._rank[xr] > self._rank[yr]:
            self._parent[yr] = xr
        else:
            self._parent[yr] = xr
            self._rank[xr] += 1


def _other_vertex(u, e):
    return e.target if e.source == u else e.source


class EdgeGraphAlgorithms:

    @staticmethod
    def minimum_spanning_tree(graph, wf):
        """Kruskal's algorithm; ``wf(edge)`` gives the weight to minimise."""
        tree = graph.empty()
        uf = UnionFind()
        for n in graph.vertices:
            uf.add_node(n)
        heap = [(wf(e), i, e) for i, e in enumerate(graph.edges)]
        heapq.heapify(heap)
        order, spanned = graph.order(), 0
        while heap and spanned < order:
            _, _, e = heapq.heappop(heap)
            if uf.find(e.source) != uf.find(e.target):
                uf.union(e.source, e.target)
                tree.insert_edge(e)
                spanned += 1
        return tree

    @staticmethod
    def triangles(graph, f=lambda t: True):
        """All triangles ``(v, w, x)``, each reported once."""
        tris = []
        gc = graph.copy()
        for v in list(gc.vertices):
            for e in list(gc.neighbor_edges(v)):
                gc.remove_edge(e)
                w = _other_vertex(v, e)
                common = set(gc.neighbor_vertices(w)) & set(gc.neighbor_vertices(v))
                for x in common:
                    t = (v, w, x)
                    if f(t):
                        tris.append(t)
        return tris

    @staticmethod
    def triangles_by_edge(graph, f=lambda t: True, verbose=False):
        """All triangles as edge triples ``(e1, e2, e3)``."""
        tris = []
        if verbose:
            print("copying the graph")
        gc = graph.copy()
        if verbose:
            print("done")
        pnum = num = 0
        while gc.size() > 0:
            e1 = gc.edges.head()
            gc.remove_edge(e1)
            u, v = e1.source, e1.target
            for w, e2 in list(gc.neighbors(u)):
                e3 = gc.edge_option(v, w)
                if e3 is not None:
                    num += 1
                    t = (e1, e2, e3)
                    if f(t):
                        tris.append(t)
            if verbose and num - pnum > 10000:
                pnum = num
                print("\r%d total and %d valid triangles " % (num, len(tris)), end="")
        if verbose:
            print()
        return tris

    @staticmethod
    def fast_triangles(graph, f=lambda t: True):
        """Same result as :meth:`triangles`, using a :class:`NeighborCache`."""
        tris = []
        gc = ObservableSlimGraphUndirected()
        for e in graph.edges:
            gc.insert_edge(e)
        for n in graph.vertices:
            gc.insert_vertex(n)
        nc = NeighborCache(gc)
        for v in list(gc.vertices):
            for e in list(gc.neighbor_edges(v)):
                gc.remove_edge(e)
                w = _other_vertex(v, e)
                for x in nc.neighbor_vertex_set(v) & nc.neighbor_vertex_set(w):
                    t = (v, w, x)
                    if f(t):
                        tris.append(t)
        nc.stop_observing()
        return tris


class MMSGraphReader:
    @staticmethod
    def from_file(fname):
        g = SlimGraphUnDirected()
        with open(fname) as fh:
            for line in fh:
                if line.strip():
                    g.insert_edge(MMSEdge.from_line(line))
        return g


# ---------------------------------------------------------------------------
# Random helpers shared by several modules (``RichImplicits`` in Scala)
# ---------------------------------------------------------------------------

def sample(seq, k, rng=_random):
    """Order-preserving selection sampling of ``k`` items (Knuth's Algorithm S).

    Port of ``RichIterable.sample`` from ``GrowthModels.scala``.
    """
    seq = list(seq)
    n = len(seq)
    assert k <= n, "k = %d > n = %d" % (k, n)
    out, need = [], k
    i = 0
    while need > 0:
        if rng.random() < need / (n - i):
            out.append(seq[i])
            need -= 1
        i += 1
    return out


def choice(seq, rng=_random):
    seq = seq if isinstance(seq, (list, tuple)) else list(seq)
    return seq[rng.randrange(len(seq))]


def bfs_levels(graph, root):
    """Breadth first distances from ``root`` (helper used by tests)."""
    dist = {root: 0}
    q = deque([root])
    while q:
        u = q.popleft()
        for w in graph.neighbor_vertices(u):
            if w not in dist:
                dist[w] = dist[u] + 1
                q.append(w)
    return dist
