"""GrowEnsemble: grow many graphs from one GrowCode program (port of
``GrowEnsemble.scala``).

Usage::

    python -m biocode ensemble --input examples/PrefAttach/PrefAttach.grw \\
        --numGraphs 20 --iterations 498 --outDir ensemble/ --threads 4

Writes ``<outDir>/<i>.edg`` (tab separated edge lists).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random

from .graph_readers import write_edge_list
from .machine import Machine
from .program import Program


def _grow_one(job):
    i, prog, num_it, out_dir, verbose = job
    random.seed()            # independent stream per worker
    print("graph %d" % i, flush=True)
    m = Machine()
    m.grow(prog, num_it, verbose=verbose)
    path = os.path.join(out_dir, "%d.edg" % i)
    write_edge_list(m.graph, path)
    return path


def grow_ensemble(prog, num_graphs, num_it, out_dir, threads=2):
    os.makedirs(out_dir, exist_ok=True)
    verbose = threads == 1
    jobs = [(i, prog, num_it, out_dir, verbose) for i in range(num_graphs)]
    if threads > 1:
        with mp.get_context().Pool(threads) as pool:
            return list(pool.imap_unordered(_grow_one, jobs))
    return [_grow_one(j) for j in jobs]


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="GrowEnsemble",
        description="GrowEnsemble lets you grow an ensemble of graphs from a given GrowCode program.")
    ap.add_argument("--input", required=True, help="file containing the input program")
    ap.add_argument("--numGraphs", type=int, required=True, help="The size of the ensemble to generate")
    ap.add_argument("--iterations", type=int, required=True,
                    help="The number of iterations to grow the graphs")
    ap.add_argument("--outDir", required=True, help="directory where the output will be written")
    ap.add_argument("--threads", type=int, default=2, help="number of processes to run in parallel")
    args = ap.parse_args(argv)

    prog = Program.from_file(args.input)
    print("using program " + prog.shortStr)
    print("Generating an ensemble of %d random graphs" % args.numGraphs)
    grow_ensemble(prog, args.numGraphs, args.iterations, args.outDir, max(1, args.threads))


if __name__ == "__main__":
    main()
