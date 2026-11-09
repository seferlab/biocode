"""The GrowCode virtual machine (port of ``Machine.scala``).

A machine owns an undirected graph that starts as the single edge ``0-1``.
:meth:`Machine.grow` runs the program ``num_iter`` times; each run
(:meth:`Machine.execute`) loads three random vertices into the registers and
steps through the instructions, executing at most ``TIME_LIMIT`` (1100)
instructions.  All per-run state (registers, label memory, loop counters) is
cleared after every run; only the graph persists.

Registers
    ``r0`` and ``r1`` usually hold vertices, ``r2`` holds a saved vertex or a
    small integer (``set``) used as the BFS depth of ``infl``; ``clear`` sets it
    to -1 (meaning depth 5).

Label memory
    ``infl p`` labels vertices reached from ``r0`` by a probabilistic BFS
    (a vertex at depth L is labelled with probability p**L).  ``attach`` links
    ``r0`` to every vertex labelled with ``r1``; ``detach`` drops the edges
    between ``r0`` and the vertices it labelled.

Two optional guards were added for the Python port: ``deadline`` (absolute
``time.monotonic()``) halts a run that takes too long -- this is the
``runOrDie``/``halt`` mechanism of the Scala code -- and ``max_order`` halts
growth once the graph has that many vertices.
"""

from __future__ import annotations

import random as _random
import time

from .console import ProgressBar
from .slimgraph import SlimGraphUnDirected, UndirectedEdge


class Machine:
    TIME_LIMIT = 1100

    def __init__(self, rng=None):
        self.register = [0, 0, 0]
        self.pc = 0
        self.attr = {}
        self.graph = SlimGraphUnDirected()
        self.random = rng if rng is not None else _random
        self._rand = self.random.random
        self.step_ctrs = {}
        self.rewind_ctrs = {}
        self.step_stack = []
        self.label_mem = {}
        self.halt = False
        self.deadline = None
        self.max_order = None
        self.timed_out = False

    # ------------------------------------------------------------------
    # Driving the machine
    # ------------------------------------------------------------------
    def get_random_node(self):
        # vertices of a grown graph are always 0..order-1
        return int(self._rand() * self.graph.order())

    def grow(self, prog, num_iter, verbose=False, f=None, samp=25,
             timeout=None, max_order=None):
        """Run ``prog`` ``num_iter`` times, growing ``self.graph``.

        ``f(graph)`` is called every ``samp`` iterations.  ``timeout`` (seconds)
        and ``max_order`` stop growth early by setting ``halt``.
        """
        if timeout is not None:
            self.deadline = time.monotonic() + timeout
        if max_order is not None:
            self.max_order = max_order
        self.graph.insert_edge(UndirectedEdge(0, 1))
        pb = ProgressBar(num_iter, "*") if verbose else None
        x = 0
        while x < num_iter and not self.halt:
            if pb:
                pb.update(x)
            self.execute(prog)
            if f is not None and x % samp == 0:
                f(self.graph)
            if self.max_order is not None and self.graph.order() >= self.max_order:
                self.halt = True
            x += 1
        if pb:
            pb.done()
        return self.graph

    def execute(self, p):
        g = self.graph
        if g.order() == 0:
            g.insert_vertex(0)
        n = g.order()
        r = self._rand
        reg = self.register
        reg[0] = int(r() * n)
        reg[1] = int(r() * n)
        reg[2] = int(r() * n)

        code = p.pcode
        plen = len(code)
        count = 0
        limit = self.TIME_LIMIT
        deadline = self.deadline
        monotonic = time.monotonic
        while self.pc < plen and count < limit and not self.halt:
            code[self.pc].apply(self)
            self.pc += 1
            count += 1
            if deadline is not None and not (count & 7) and monotonic() > deadline:
                self.halt = True
                self.timed_out = True
        self.reset()

    def reset(self):
        self.register[0] = self.register[1] = self.register[2] = 0
        self.pc = 0
        self.attr.clear()
        self.step_ctrs.clear()
        self.step_stack.clear()
        self.label_mem.clear()
        self.rewind_ctrs.clear()

    def node_in(self, i):
        x = self.register[i]
        return x if x in self.graph.adj_list else self.get_random_node()

    def inc_pc(self):
        self.pc += 1

    def state_str(self):
        r = self.register
        return "r=[%4d %4d %4d] pc=[%4d] n=[%4d] at=[%d]" % (
            r[0], r[1], r[2], self.pc, self.graph.order(), self.attr.get(r[0], 0))

    def _check_deadline(self):
        if self.deadline is not None and time.monotonic() > self.deadline:
            self.halt = True
            self.timed_out = True

    # ------------------------------------------------------------------
    # Instructions
    # ------------------------------------------------------------------
    def op_nop(self):
        pass

    def op_detach(self):
        """Remove edges between r0 and the vertices r0 has labelled."""
        u = self.node_in(0)
        lm = self.label_mem
        g = self.graph
        victims = [v for v in g.neighbor_vertices(u) if lm.get(v) == u]
        for v in victims:
            g.remove_edge(UndirectedEdge(u, v))

    def op_attach(self):
        """Connect r0 to every vertex labelled with r1."""
        u = self.node_in(0)
        v = self.node_in(1)
        g = self.graph
        for w, label in self.label_mem.items():
            if v == label:
                g.insert_edge(UndirectedEdge(u, w))

    def op_infl(self, p):
        """Probabilistic BFS from r0 labelling reached vertices with r0."""
        k = self.register[2]
        u = self.node_in(0)
        adj = self.graph.adj_list
        r = self._rand
        lm = self.label_mem
        seen = {u}
        level = 1
        io = adj[u]
        next_level = set(io.incoming)
        next_level.update(io.outgoing)
        if k == -1:
            k = 5
        while next_level and level <= k and not self.halt:
            this_level = next_level
            next_level = set()
            pl = p ** level
            # same as testing "v not in seen" inside the loop, since this_level
            # has no duplicates: one random draw per unseen vertex
            for v in this_level - seen:
                if r() < pl:
                    lm[v] = u
                    io = adj[v]
                    next_level.update(io.incoming)
                    next_level.update(io.outgoing)
            seen |= this_level
            level += 1
            self._check_deadline()

    def op_clear(self):
        self.register[2] = -1

    def op_ron_paul(self):
        """``clearinf``: forget all labels."""
        self.label_mem.clear()

    def op_copy(self):
        g = self.graph
        n = g.order()
        g.insert_vertex(n)
        for v in list(g.neighbor_vertices(self.node_in(0))):
            g.insert_edge(UndirectedEdge(n, v))
        self.register[1] = self.node_in(0)
        self.register[0] = n

    def op_new(self):
        g = self.graph
        n = g.order()
        g.insert_vertex(n)
        self.register[0] = n

    def op_save(self):
        self.register[2] = self.register[0]

    def op_load(self):
        if self.register[2] >= 0:
            self.register[0] = self.register[2]

    def op_swap(self):
        reg = self.register
        reg[0], reg[1] = reg[1], reg[0]

    def op_create(self):
        # nodeIn is evaluated four times, exactly like the Scala code
        if self.node_in(0) != self.node_in(1):
            self.graph.insert_edge(UndirectedEdge(self.node_in(0), self.node_in(1)))

    def op_rnode(self):
        g = self.graph
        if g.order() > 1:
            x = self.register[0]
            while x == self.register[0]:
                self.register[0] = self.get_random_node()

    def op_redge(self):
        g = self.graph
        m = g.size()
        if m > 0:
            e = g.edge_at(int(self._rand() * m))
            if int(self._rand() * 2) == 0:
                self.register[0], self.register[1] = e.source, e.target
            else:
                self.register[1], self.register[0] = e.source, e.target

    def op_neigh(self):
        nlist = list(self.graph.neighbor_vertices(self.node_in(0)))
        if nlist:
            self.register[0] = nlist[int(self._rand() * len(nlist))]

    def op_walk(self, n):
        self.register[1] = self.register[0]
        u = self.node_in(0)
        g = self.graph
        i = 0
        nlist = list(g.neighbor_vertices(u))
        while i < n and nlist:
            nlist = list(g.neighbor_vertices(u))
            u = nlist[int(self._rand() * len(nlist))]
            i += 1
        self.register[0] = u

    def op_set(self, n):
        self.register[2] = n

    def op_step(self, step_op):
        if step_op in self.step_ctrs:
            self.step_ctrs[step_op] -= 1
        else:
            self.step_ctrs[step_op] = step_op.d
            self.step_stack.append((step_op, self.pc))

    def op_loop(self):
        while self.step_stack:
            closest, pc_loc = self.step_stack.pop()
            if self.step_ctrs[closest] > 0:
                self.pc = pc_loc
                self.step_stack.append((closest, pc_loc))
                break

    def op_rewind(self, n_instr, n_loop):
        ctr = self.rewind_ctrs
        pc = self.pc
        if pc not in ctr:
            ctr[pc] = n_loop
        if ctr[pc] > 0:
            ctr[pc] -= 1
            self.pc = max(pc - n_instr, 0)

    def op_truth(self, p1, p2, p3, p4):
        n0, n1 = self.node_in(0), self.node_in(1)
        g = self.graph
        r = self._rand
        for _ in range(10):
            n = self.get_random_node()
            if n != n0 and n != n1:
                if g.edge_option(n0, n) is not None:
                    if g.edge_option(n1, n) is not None:
                        if r() >= p4:
                            g.remove_edge(n0, n)
                    elif r() >= p3:
                        g.remove_edge(n0, n)
                else:
                    if g.edge_option(n1, n) is not None:
                        if r() < p2:
                            g.insert_edge(UndirectedEdge(n0, n))
                    elif r() < p1:
                        g.insert_edge(UndirectedEdge(n0, n))

    def op_scoin(self, p):
        """Skip the next instruction with probability p."""
        if self._rand() < p:
            self.pc += 1
