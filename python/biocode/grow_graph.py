"""GrowGraph: analyse an evolved population or a single GrowCode program.

Port of ``GrowGraph.scala``.

* Given a population file written by LearnGraph (binary), it normalises the
  objectives, picks the program with the best mean/stddev trade-off (or the
  one given with ``-p``), writes it as a text ``.grw`` program, grows a graph
  from it and prints its statistics next to the original graph's.
* Given a text program (``; GROWCODE`` header), it grows one graph, saves
  degree-distribution plots, prints statistics and optionally runs a
  power-law goodness-of-fit test on the degrees.

Usage::

    python -m biocode grow -g examples/PrefAttach/PrefAttach.edg -i 498 \\
        -o examples/PrefAttach/PrefAttach.grw examples/PrefAttach/PrefAttach.grw
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

from . import marshall
from .graph_algorithms import clustering, print_properties
from .graph_readers import EdgeListReader
from .histogram import Histogram
from .machine import Machine
from .plfit import plfit
from .program import Program

LOGO = r"""
   ___                    ___           _
  / _ \_ __ _____      __/ __\ ___   __| | ___
 / /_\/ '__/ _ \ \ /\ / / /   / _ \ / _` |/ _ \
/ /_\\| | | (_) \ V  V / /___| (_) | (_| |  __/
\____/|_|  \___/ \_/\_/\____/ \___/ \__,_|\___|
  """


# ---------------------------------------------------------------------------
# Degree distribution helpers
# ---------------------------------------------------------------------------

def analyze_degree_dist(degs, timeout):
    """Power-law test.  Returns ``(is_scale_free, alpha)`` with p >= 0.1 as the
    acceptance threshold (as in the Scala code)."""
    data = [int(d + 1) for d in degs]
    try:
        alpha, xmin, ll, p, gof = plfit(data, timeout=timeout)
    except ValueError:
        return False, -1.0
    print("alpha = %s, mxin = %s, L = %s, P = %s G = %s" % (alpha, xmin, ll, p, gof))
    return (p >= 0.1), alpha


def _pyplot():
    try:
        import matplotlib
        if not os.environ.get("DISPLAY") and sys.platform not in ("darwin", "win32"):
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("(matplotlib is not installed; skipping plots)")
        return None


def plot_degree_dist(degs, fname=None):
    plt = _pyplot()
    if plt is None:
        return
    h = Histogram(degs, nbins=20).hist
    y = np.log(h + 1)
    x = np.log(np.arange(len(y)) + 1)
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=12, color="tab:blue")
    ax.set_xlabel("Histogram bin # (low => high)")
    ax.set_ylabel("log(frequency)")
    _finish(fig, plt, fname)


def rank_plot_degrees(degs, fname=None):
    plt = _pyplot()
    if plt is None:
        return
    y = np.log10(np.asarray(degs, dtype=float) + 1)
    x = np.log10(np.arange(len(y)) + 1)
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=6, color="tab:blue")
    ax.set_xlabel("log10(node rank)")
    ax.set_ylabel("log10(degree)")
    _finish(fig, plt, fname)


def _finish(fig, plt, fname):
    if fname:
        fig.savefig(fname, dpi=120, bbox_inches="tight")
        print("saved plot %s" % fname)
        plt.close(fig)
    else:
        plt.show()


def query_continue(assume_yes=False):
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        return False
    ans = input("Continue with P-value analysis? [y/N] ").strip().lower()
    return ans in ("y", "yes")


# ---------------------------------------------------------------------------
# Single program / population analysis
# ---------------------------------------------------------------------------

def analyze_single_program(prog, num_it, timeout, plot_prefix="growgraph", assume_yes=False):
    machine = Machine()
    machine.grow(prog, num_it, verbose=True)
    graph = machine.graph
    print("Graph stats: order=%d size=%d" % (graph.order(), graph.size()))
    degs = sorted((float(graph.degree(v)) for v in graph.vertices), reverse=True)
    if plot_prefix:
        plot_degree_dist(degs, plot_prefix + "_degree_hist.png")
        rank_plot_degrees(degs, plot_prefix + "_degree_rank.png")
    print_properties(graph)
    print("Clustering: %r" % clustering(graph))
    if query_continue(assume_yes):
        return analyze_degree_dist(degs, timeout)[0]
    return False


def _java_double_key(x):
    """Sort key ordering doubles like java.lang.Double.compare (NaN last)."""
    return (1, 0.0) if math.isnan(x) else (0, x)


def select_program(progs, prog_ind=-1, out=print):
    """Choose a program from an evolved population (the Scala selection rule).

    Programs are sorted by the first objective (descending); each objective is
    normalised by ``min(v, 1) / min(max_v, 1)`` and the program maximising
    ``sum / stddev`` of its normalised objectives is chosen.  Returns
    ``(sorted_programs, chosen_index)``.
    """
    def objs(p):
        return [float(o) for o in p.properties["fitness"].split(",")]

    sorted_progs = sorted(progs, key=lambda p: objs(p)[0], reverse=True)
    out(str(len(progs)))
    for i, p in enumerate(sorted_progs):
        out("[%d] %d , %s\n" % (i, len(p), p.properties["fitness"]))

    objectives = np.array([objs(p) for p in sorted_progs], dtype=float)
    nobj = objectives.shape[1]
    out("Number of objectives: %d" % nobj)
    max_obj = objectives.max(axis=0)
    out("Max objectives: " + ", ".join("%d -> %r" % (i, float(v)) for i, v in enumerate(max_obj)))
    with np.errstate(divide="ignore", invalid="ignore"):
        normed = np.minimum(objectives, 1.0) / np.minimum(max_obj, 1.0)
    out("Normalized objectives:")
    for row in normed:
        out(",".join(repr(float(v)) for v in row))

    with np.errstate(divide="ignore", invalid="ignore"):
        sums = normed.sum(axis=1)
        std = np.sqrt(((normed - normed.mean(axis=1, keepdims=True)) ** 2).mean(axis=1))
        score = -sums / std
    order = sorted(range(len(sorted_progs)), key=lambda i: _java_double_key(float(score[i])))
    max_id = order[0]
    out("Maximum objective: %s; Program ID: %d" % (",".join(repr(float(v)) for v in normed[max_id]), max_id))
    return sorted_progs, (max_id if prog_ind == -1 else prog_ind)


def analyze_population(progs, num_it, timeout, prog_ind, orig_graph, prog_out_file):
    sorted_progs, idx = select_program(progs, prog_ind)
    prog = sorted_progs[idx]
    prog.to_file(prog_out_file)

    machine = Machine()
    machine.grow(prog, num_it)
    print("Selected Program:")
    print(prog.shortStr)
    print("Its fitness characteristics: " + prog.properties["fitness"])
    print("A graph from best model:")
    print_properties(machine.graph)
    print("-----")
    print("Original graph:")
    if orig_graph:
        print_properties(EdgeListReader.from_file_sg(orig_graph))
    else:
        print("(no original graph given; use -g)")
    return prog


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="GrowGraph",
        description="GrowGraph: Version 1.0 Copyright(c) 2011, Geet Duggal & Rob Patro" + LOGO,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--iterations", type=int, default=100, metavar="n",
                    help="Number of iterations to grow (default is 100)")
    ap.add_argument("-t", "--timeout", type=int, default=50, metavar="t",
                    help="Time, in seconds, to wait for each p-value test (default is 50)")
    ap.add_argument("-p", "--program", type=int, default=-1, metavar="p",
                    help="Program index to select from the population")
    ap.add_argument("-o", "--output", default="growgraph.grw", metavar="o",
                    help="Output file for selected GrowCode program")
    ap.add_argument("-g", "--origGraph", default="", metavar="g",
                    help="Original graph for printing statistics")
    ap.add_argument("--plots", default="growgraph", metavar="PREFIX",
                    help="(single program) prefix for the degree plots; '' disables them")
    ap.add_argument("-y", "--yes", action="store_true",
                    help="(single program) run the power-law test without asking")
    ap.add_argument("growfile", help="Grow file from which to read graph")
    args = ap.parse_args(argv)

    if marshall.is_ascii(args.growfile):
        analyze_single_program(Program.from_file(args.growfile), args.iterations,
                               args.timeout, plot_prefix=args.plots, assume_yes=args.yes)
    else:
        analyze_population(marshall.read(args.growfile), args.iterations, args.timeout,
                           args.program, args.origGraph, args.output)


if __name__ == "__main__":
    main()
