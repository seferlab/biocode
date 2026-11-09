"""Target statistics and the fitness function.

Port of ``GCInitializer.scala`` (target graph statistics) and
``BestProgramProblem.scala`` (scoring a GrowCode program).

A program is grown ``replicates`` (5) times for ``numGrowIter`` iterations.
For each grown graph, if its vertex and edge counts are both within 50 % of
the target's, three objectives are computed (all to be maximised):

* assortativity   ``1 - |r - r_target| / 2``
* degree shape    ``1 / (L1(shape - shape_target) + 0.0001)``
* avg clustering  ``1 - |C - C_target|``

otherwise the replicate scores ``(0, 0, 0)``.  The fitness is the mean over
replicates, in the order ``(assortativity, shape, avg clustering)``.
"""

from __future__ import annotations

import sys

import numpy as np

from .graph_algorithms import assortativity, avg_clustering, eigenvector_centrality, shape
from .graph_readers import EdgeListReader
from .machine import Machine
from .operations import f32

OBJECTIVE_NAMES = ("assortativity", "shape", "avg_clustering")


class TargetStats:
    """Statistics of the target network (``GCInitializer``)."""

    def __init__(self, graph, compute_eig=False):
        self.order = graph.order()
        self.size = graph.size()
        self.shape = np.asarray(shape(graph))
        self.diameter = 10                     # hard-coded in the Scala code
        self.assortativity = assortativity(graph)
        self.avg_clustering = avg_clustering(graph)
        self.clustering = 0.0                  # not used (commented out in Scala)
        # computed by GCInitializer but never used in the fitness
        self.eig_centrality = eigenvector_centrality(graph) if compute_eig else None

    @classmethod
    def from_file(cls, path, verbose=True, compute_eig=False):
        g = EdgeListReader.from_file_sg(path)
        t = cls(g, compute_eig=compute_eig)
        if verbose:
            print("|V|=%d |E|=%d" % (t.order, t.size))
            print("AVG Clustering\t%r" % t.avg_clustering)
        return t


def score_graph(graph, target):
    """The three objective values of one grown graph (float32-rounded)."""
    fit_order = abs(target.order - graph.order())
    fit_size = abs(target.size - graph.size())
    if fit_order < target.order / 2.0 and fit_size < target.size / 2.0:
        diff = float(np.sum(np.abs(target.shape - shape(graph))))
        f_shape = f32(1.0 / abs(-diff - 0.0001))
        f_assort = 1.0 - f32(abs(assortativity(graph) + 1 - (target.assortativity + 1))) / 2.0
        f_clust = 1.0 - f32(abs(avg_clustering(graph) - target.avg_clustering))
        return f32(f_assort), f_shape, f32(f_clust)
    return 0.0, 0.0, 0.0


def evaluate_program(prog, target, num_iter, replicates=5, timeout=120.0,
                     rng=None, log=None):
    """Fitness tuple ``(assortativity, shape, avg_clustering)`` of ``prog``.

    ``timeout`` (seconds) bounds each replicate; a replicate that runs out of
    time is scored on the graph grown so far (as in the Scala code).  Growth
    also stops as soon as the graph has 1.5x the target's vertices, because
    from then on the replicate can only score zero.
    """
    max_order = 1.5 * target.order
    scores = []
    for _ in range(replicates):
        m = Machine(rng)
        m.grow(prog, num_iter, timeout=timeout, max_order=max_order)
        if m.timed_out and log is not None:
            log("Grow timed out after %ss; scoring the partial graph of:\n%s"
                % (timeout, prog.shortStr))
        scores.append(score_graph(m.graph, target))
    mean = np.mean(np.asarray(scores, dtype=np.float32), axis=0, dtype=np.float32)
    return tuple(float(x) for x in mean)


def _stderr(msg):
    print(msg, file=sys.stderr)
